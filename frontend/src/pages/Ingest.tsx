import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import {
  ApiError,
  api,
  detectLink,
  formatBytes,
  formatDuration,
  LAYOUTS,
  LINK_LABELS,
  type DriveFolderListing,
  type Job,
  type LinkKind,
  type LinkProgress,
  type JobSettingsOverrides,
  type Layout,
  type ProviderStatus,
  type Settings,
} from '../api'
import { ErrorNote } from '../components/ErrorNote'

export function Ingest() {
  const navigate = useNavigate()
  // Explore's "Clip this video" lands here as /?url=…, prefilled.
  const [searchParams] = useSearchParams()
  const [url, setUrl] = useState(() => searchParams.get('url') ?? '')
  const [busy, setBusy] = useState<'url' | 'file' | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [overrides, setOverrides] = useState<JobSettingsOverrides>({})
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [defaults, setDefaults] = useState<Settings | null>(null)
  // A pasted Drive folder: its videos, and which of them to clip.
  const [folder, setFolder] = useState<DriveFolderListing | null>(null)
  const [picked, setPicked] = useState<Set<string>>(new Set())
  const [linkProgress, setLinkProgress] = useState<LinkProgress | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [jobsVersion, setJobsVersion] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const linkKind = detectLink(url)
  // A ref, not the `busy` state: a file dropped while a URL fetch is in
  // flight must be refused even if the drop handler closed over a stale value.
  const busyRef = useRef(false)

  useEffect(() => {
    api.providerStatus().then(setProviders).catch(() => undefined)
    api.getSettings().then(setDefaults).catch(() => undefined)
  }, [])

  // Recent jobs: refreshed while any of them is still queued or running, so
  // the list doesn't keep showing "12%" for a job that finished long ago.
  // The poll stops by itself once every listed job has finished.
  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    const load = async () => {
      try {
        const next = await api.listJobs(8)
        if (stopped) return
        setJobs(next)
        if (next.some((job) => job.status === 'queued' || job.status === 'running')) {
          timer = window.setTimeout(load, 3000)
        }
      } catch {
        /* The list is a convenience; the job pages stream their own state. */
      }
    }
    void load()
    return () => {
      stopped = true
      window.clearTimeout(timer)
    }
  }, [jobsVersion])

  const start = useCallback(
    async (kind: 'url' | 'file', run: () => Promise<{ id: string }>) => {
      if (busyRef.current) return
      // Checked before ingesting: a bad override used to surface only after
      // the whole video had been downloaded or uploaded.
      const invalid = validateOverrides(overrides, defaults)
      if (invalid) {
        setError(new Error(invalid))
        setAdvancedOpen(true)
        return
      }
      busyRef.current = true
      setBusy(kind)
      setError(null)
      try {
        const source = await run()
        const job = await api.createJob(source.id, overrides)
        navigate(`/jobs/${job.id}`)
      } catch (err) {
        setError(err as Error)
      } finally {
        busyRef.current = false
        setBusy(null)
      }
    },
    [navigate, overrides, defaults],
  )

  /**
   * Download a file link, polling the server for byte progress meanwhile
   * (the POST itself only answers once the file is in).
   */
  const ingestLinkWithProgress = useCallback(async (link: string) => {
    const progressId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
    setLinkProgress(null)
    const timer = window.setInterval(() => {
      api.linkProgress(progressId).then(setLinkProgress).catch(() => undefined)
    }, 700)
    try {
      return await api.ingestLink(link, progressId)
    } finally {
      window.clearInterval(timer)
      setLinkProgress(null)
    }
  }, [])

  const submitUrl = async (event: React.FormEvent) => {
    event.preventDefault()
    const link = url.trim()
    if (!link) return
    setNotice(null)
    if (linkKind === 'gdrive_folder') {
      if (busyRef.current) return
      busyRef.current = true
      setBusy('url')
      setError(null)
      try {
        const listing = await api.driveFolder(link)
        setFolder(listing)
        setPicked(new Set(listing.files.map((f) => f.id)))
      } catch (err) {
        setError(err as Error)
      } finally {
        busyRef.current = false
        setBusy(null)
      }
      return
    }
    if (linkKind && linkKind !== 'youtube') {
      void start('url', () => ingestLinkWithProgress(link))
      return
    }
    void start('url', () => api.ingestYouTube(link))
  }

  /** Ingest the picked folder videos one by one, queueing a job for each. */
  const clipFolder = async () => {
    if (!folder || busyRef.current) return
    const files = folder.files.filter((f) => picked.has(f.id))
    if (files.length === 0) return
    const invalid = validateOverrides(overrides, defaults)
    if (invalid) {
      setError(new Error(invalid))
      setAdvancedOpen(true)
      return
    }
    busyRef.current = true
    setBusy('url')
    setError(null)
    let firstJob: string | null = null
    let queued = 0
    try {
      for (const file of files) {
        setNotice(`Downloading ${file.name} (${queued + 1} of ${files.length})…`)
        const source = await ingestLinkWithProgress(file.url)
        const job = await api.createJob(source.id, overrides)
        firstJob ??= job.id
        queued += 1
        setJobsVersion((v) => v + 1)
      }
      if (files.length === 1 && firstJob) {
        navigate(`/jobs/${firstJob}`)
        return
      }
      setNotice(`Queued ${queued} video${queued === 1 ? '' : 's'} from “${folder.name}”.`)
      setFolder(null)
      setUrl('')
    } catch (err) {
      setNotice(queued ? `Queued ${queued} of ${files.length} before the error below.` : null)
      setError(err as Error)
    } finally {
      busyRef.current = false
      setBusy(null)
    }
  }

  const submitFile = (file: File) => void start('file', () => api.uploadSource(file))

  const usableProvider = providers.find((p) => p.available)

  return (
    <div className="pt-14">
      {/* Masthead. Left-aligned and asymmetric — the field is the subject, not
          a centred hero card. */}
      <div className="rise max-w-3xl">
        <p className="eyebrow">Local · No accounts · No watermarks</p>
        <h1 className="mt-5 font-display text-[clamp(2.75rem,7vw,5.5rem)] leading-[0.95] text-ink-100">
          Long video in.
          <br />
          <span className="italic text-sodium-500">Shorts</span> out.
        </h1>
      </div>

      <div className="mt-16 grid gap-x-16 gap-y-12 lg:grid-cols-[1.35fr_1fr]">
        {/* URL */}
        <section className="rise" style={{ animationDelay: '90ms' }}>
          <form onSubmit={submitUrl}>
            <label htmlFor="url" className="eyebrow">
              Paste a link
            </label>
            <div className="mt-3 flex items-end gap-4">
              <input
                id="url"
                className="field font-display text-xl md:text-2xl"
                placeholder="YouTube, Google Drive, Dropbox, OneDrive or .mp4 link"
                value={url}
                onChange={(e) => {
                  setUrl(e.target.value)
                  setFolder(null)
                }}
                autoComplete="off"
                spellCheck={false}
                disabled={busy !== null}
              />
              <button
                type="submit"
                className="btn btn-primary shrink-0"
                disabled={busy !== null || !url.trim()}
              >
                {busy === 'url' ? 'Fetching…' : linkKind === 'gdrive_folder' ? 'List' : 'Start'}
              </button>
            </div>
          </form>

          {linkKind && <LinkBadge kind={linkKind} />}
          {busy === 'url' && linkProgress && <DownloadProgress progress={linkProgress} />}
          {folder && (
            <FolderPicker
              folder={folder}
              picked={picked}
              onChange={setPicked}
              onClip={() => void clipFolder()}
              busy={busy !== null}
            />
          )}
          {notice && <p className="mt-3 text-xs text-ink-300">{notice}</p>}

          <p className="mt-3 text-xs leading-relaxed text-ink-500">
            Only download video you own or have the rights to process.
          </p>

          <AdvancedOptions
            open={advancedOpen}
            onToggle={() => setAdvancedOpen((v) => !v)}
            overrides={overrides}
            onChange={setOverrides}
            providers={providers}
            defaults={defaults}
          />
        </section>

        {/* Upload */}
        <section className="rise" style={{ animationDelay: '160ms' }}>
          <p className="eyebrow">Or drop a file</p>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              const file = e.dataTransfer.files[0]
              if (file) submitFile(file)
            }}
            className={[
              'mt-3 flex min-h-52 cursor-pointer flex-col items-center justify-center gap-2 border border-dashed px-6 text-center transition-colors duration-200',
              dragging
                ? 'border-sodium-500 bg-sodium-700/10'
                : 'border-ink-700 hover:border-ink-600',
            ].join(' ')}
            onClick={() => {
              if (busy === null) fileInput.current?.click()
            }}
            role="button"
            tabIndex={0}
            aria-disabled={busy !== null}
            aria-label="Choose a video or audio file to upload"
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                // Space would otherwise scroll the page as well.
                e.preventDefault()
                if (busy === null) fileInput.current?.click()
              }
            }}
          >
            <span className="font-display text-2xl text-ink-200">
              {busy === 'file' ? 'Uploading…' : 'Drop video or audio'}
            </span>
            <span className="text-xs text-ink-500">mp4 · mov · mkv · webm · mp3 · wav · m4a</span>
          </div>
          <input
            ref={fileInput}
            type="file"
            className="hidden"
            accept="video/*,audio/*"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) submitFile(file)
              e.target.value = ''
            }}
          />
        </section>
      </div>

      {error && (
        <div className="mt-10 max-w-3xl">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {providers.length > 0 && !usableProvider && (
        <p className="mt-10 max-w-3xl border-l-2 border-sodium-600 pl-4 text-sm text-ink-300">
          No AI provider is reachable yet, so clip selection will fail. Add an API key or
          start Ollama in{' '}
          <a href="/settings" className="text-sodium-500 underline underline-offset-4">
            Settings
          </a>
          .
        </p>
      )}

      <RecentJobs jobs={jobs} retentionDays={defaults?.housekeeping?.retention_days ?? 0} />
    </div>
  )
}

function AdvancedOptions({
  open,
  onToggle,
  overrides,
  onChange,
  providers,
  defaults,
}: {
  open: boolean
  onToggle: () => void
  overrides: JobSettingsOverrides
  onChange: (next: JobSettingsOverrides) => void
  providers: ProviderStatus[]
  defaults: Settings | null
}) {
  const set = <K extends keyof JobSettingsOverrides>(key: K, value: JobSettingsOverrides[K]) =>
    onChange({ ...overrides, [key]: value })

  return (
    <div className="mt-8">
      <button
        type="button"
        onClick={onToggle}
        className="btn btn-quiet -ml-1"
        aria-expanded={open}
        aria-controls="job-options"
      >
        <span
          className="inline-block transition-transform duration-300"
          style={{ transform: open ? 'rotate(90deg)' : 'none' }}
          aria-hidden
        >
          ›
        </span>
        {open ? 'Hide options' : 'Options'}
      </button>

      {/* grid-template-rows rather than height, so the reveal animates without
          touching layout properties. */}
      <div
        className="grid transition-[grid-template-rows] duration-400 ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ gridTemplateRows: open ? '1fr' : '0fr' }}
      >
        {/* `inert` while collapsed: the 0fr row hides the fields visually,
            but without it they stayed in the tab order, invisible. */}
        <div
          id="job-options"
          className="overflow-hidden"
          // React 18 has no typed `inert` prop; an empty string sets the attribute.
          {...(open ? {} : { inert: '' })}
        >
          <div className="grid gap-x-8 gap-y-5 pt-5 sm:grid-cols-2">
            <Selector
              label="Provider"
              value={overrides.provider ?? ''}
              onChange={(v) => set('provider', v || undefined)}
              options={[
                { value: '', label: 'Use default' },
                ...providers.map((p) => ({
                  value: p.name,
                  label: p.available ? p.name : `${p.name} — unavailable`,
                  disabled: !p.available,
                })),
              ]}
            />
            <Selector
              label="Whisper model"
              value={overrides.whisper_model ?? ''}
              onChange={(v) => set('whisper_model', v || undefined)}
              options={[
                { value: '', label: 'Use default' },
                { value: 'tiny', label: 'tiny — fastest, roughest' },
                { value: 'base', label: 'base' },
                { value: 'small', label: 'small — balanced' },
                { value: 'medium', label: 'medium' },
                { value: 'large-v3', label: 'large-v3 — slowest, best' },
              ]}
            />
            {/* Streams with a facecam overlay: stack the cam above the
                gameplay instead of cropping it out of the 9:16 frame. */}
            <Selector
              label="Layout (9:16)"
              value={overrides.layout ?? ''}
              onChange={(v) => set('layout', (v || undefined) as Layout | undefined)}
              options={[
                {
                  value: '',
                  label: `Use default${
                    defaults
                      ? ` — ${LAYOUTS.find((l) => l.value === defaults.export.layout)?.label ?? defaults.export.layout}`
                      : ''
                  }`,
                },
                { value: 'auto', label: 'Auto — facecam + gameplay when a cam is found' },
                { value: 'split', label: 'Facecam + gameplay' },
                { value: 'crop', label: 'Crop — follow the speaker' },
              ]}
            />
            <NumberField
              label="Max clips"
              value={overrides.max_clips}
              placeholder={String(defaults?.clips.max_clips ?? 10)}
              min={1}
              max={50}
              onChange={(v) => set('max_clips', v)}
            />
            <div className="grid grid-cols-2 gap-4">
              <NumberField
                label="Min length (s)"
                value={overrides.min_duration_s}
                placeholder={String(defaults?.clips.min_duration_s ?? 20)}
                min={5}
                max={300}
                onChange={(v) => set('min_duration_s', v)}
              />
              <NumberField
                label="Max length (s)"
                value={overrides.max_duration_s}
                placeholder={String(defaults?.clips.max_duration_s ?? 90)}
                min={5}
                max={300}
                onChange={(v) => set('max_duration_s', v)}
              />
            </div>
            {/* Shows the saved default until touched, and sends an explicit
                true/false once it is: "unchecked" used to mean "use the
                default", so a job could not opt out when the default was on. */}
            <label className="flex items-center gap-3 text-sm text-ink-200 sm:col-span-2">
              <input
                type="checkbox"
                checked={overrides.diarization ?? defaults?.whisper.diarization ?? false}
                onChange={(e) => set('diarization', e.target.checked)}
                className="size-4 accent-sodium-500"
              />
              Multiple speakers — label who is talking, and cut to them
            </label>
          </div>
        </div>
      </div>
    </div>
  )
}

/**
 * Mirror the server's checks on per-job overrides (JobSettingsIn and the
 * min < max rule in create_job), so a typo is caught before the download.
 * Returns the problem, or null when the overrides are acceptable.
 */
function validateOverrides(
  overrides: JobSettingsOverrides,
  defaults: Settings | null,
): string | null {
  const { max_clips, min_duration_s, max_duration_s } = overrides
  if (max_clips !== undefined && (!Number.isInteger(max_clips) || max_clips < 1 || max_clips > 50)) {
    return 'Max clips must be a whole number from 1 to 50.'
  }
  for (const [label, value] of [
    ['Min length', min_duration_s],
    ['Max length', max_duration_s],
  ] as const) {
    if (value !== undefined && !(value > 0)) return `${label} must be more than 0 seconds.`
  }
  const min = min_duration_s ?? defaults?.clips.min_duration_s
  const max = max_duration_s ?? defaults?.clips.max_duration_s
  if (min !== undefined && max !== undefined && min >= max) {
    return `Min length (${min}s) must be below max length (${max}s).`
  }
  return null
}

function Selector({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string; disabled?: boolean }[]
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="field mt-1 cursor-pointer text-sm"
      >
        {options.map((option) => (
          <option
            key={option.value}
            value={option.value}
            disabled={option.disabled}
            className="bg-ink-850"
          >
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function NumberField({
  label,
  value,
  placeholder,
  min,
  max,
  onChange,
}: {
  label: string
  value: number | undefined
  placeholder: string
  min: number
  max: number
  onChange: (value: number | undefined) => void
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric mt-1 text-sm"
        placeholder={placeholder}
        value={value ?? ''}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
      />
    </label>
  )
}

function LinkIcon({ kind }: { kind: LinkKind }) {
  // Tiny monochrome glyphs: a play button, a cloud, a folder, a chain link.
  const path =
    kind === 'youtube'
      ? 'M2 4.5A2.5 2.5 0 0 1 4.5 2h7A2.5 2.5 0 0 1 14 4.5v7a2.5 2.5 0 0 1-2.5 2.5h-7A2.5 2.5 0 0 1 2 11.5zM6.5 5.5v5l4-2.5z'
      : kind === 'gdrive_folder'
        ? 'M1.5 4A1.5 1.5 0 0 1 3 2.5h3l1.5 1.5H13A1.5 1.5 0 0 1 14.5 5.5v6A1.5 1.5 0 0 1 13 13H3a1.5 1.5 0 0 1-1.5-1.5z'
        : kind === 'direct'
          ? 'M6.5 9.5l3-3M5 7.5 3.8 8.7a2.3 2.3 0 0 0 3.3 3.3L8.5 11M11 8.5l1.2-1.2a2.3 2.3 0 0 0-3.3-3.3L7.5 5'
          : 'M4.5 12.5a3 3 0 0 1-.4-6 4 4 0 0 1 7.7-1A3.3 3.3 0 0 1 11.5 12.5z'
  const stroke = kind === 'direct'
  return (
    <svg
      viewBox="0 0 16 16"
      className="size-3.5 shrink-0"
      aria-hidden
      fill={stroke ? 'none' : 'currentColor'}
      stroke={stroke ? 'currentColor' : 'none'}
      strokeWidth={stroke ? 1.6 : 0}
      strokeLinecap="round"
    >
      <path d={path} />
    </svg>
  )
}

function LinkBadge({ kind }: { kind: LinkKind }) {
  return (
    <p className="mt-3 flex items-center gap-2 text-xs text-ink-400">
      <LinkIcon kind={kind} />
      <span>
        {LINK_LABELS[kind]}
        {kind === 'gdrive_folder' && ' · pick the videos to clip'}
      </span>
    </p>
  )
}

function DownloadProgress({ progress }: { progress: LinkProgress }) {
  const { done_bytes: done, total_bytes: total } = progress
  const fraction = total ? Math.min(1, done / total) : null
  return (
    <div className="mt-3 max-w-md" aria-live="polite">
      <div className="h-1 bg-ink-800">
        <div
          className="h-1 bg-sodium-500 transition-[width] duration-500"
          style={{ width: `${Math.round((fraction ?? 0) * 100)}%` }}
        />
      </div>
      <p className="numeric mt-1.5 text-xs text-ink-500">
        {fraction !== null
          ? `${Math.round(fraction * 100)}% of ${formatBytes(total ?? 0)}`
          : done > 0
            ? `${formatBytes(done)} downloaded`
            : 'Connecting…'}
      </p>
    </div>
  )
}

function FolderPicker({
  folder,
  picked,
  onChange,
  onClip,
  busy,
}: {
  folder: DriveFolderListing
  picked: Set<string>
  onChange: (next: Set<string>) => void
  onClip: () => void
  busy: boolean
}) {
  const toggle = (id: string) => {
    const next = new Set(picked)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    onChange(next)
  }
  const all = folder.files.length > 0 && picked.size === folder.files.length

  return (
    <div className="mt-6 border border-ink-800 p-4">
      <div className="flex items-baseline justify-between gap-4">
        <p className="truncate text-sm text-ink-100">{folder.name}</p>
        <span className="numeric shrink-0 text-xs text-ink-500">
          {folder.files.length} video{folder.files.length === 1 ? '' : 's'}
        </span>
      </div>
      {folder.files.length === 0 ? (
        <p className="mt-3 text-xs text-ink-500">No video or audio files in this folder.</p>
      ) : (
        <>
          <label className="mt-3 flex items-center gap-3 text-xs text-ink-400">
            <input
              type="checkbox"
              checked={all}
              onChange={() => onChange(all ? new Set() : new Set(folder.files.map((f) => f.id)))}
              className="size-4 accent-sodium-500"
              disabled={busy}
            />
            Select all
          </label>
          <ul className="mt-2 max-h-64 overflow-y-auto">
            {folder.files.map((file) => (
              <li key={file.id}>
                <label className="flex items-center gap-3 border-b border-ink-850 py-2 text-sm text-ink-200">
                  <input
                    type="checkbox"
                    checked={picked.has(file.id)}
                    onChange={() => toggle(file.id)}
                    className="size-4 shrink-0 accent-sodium-500"
                    disabled={busy}
                  />
                  <span className="truncate">{file.name}</span>
                  {file.folder && (
                    <span className="ml-auto shrink-0 text-xs text-ink-600">{file.folder}</span>
                  )}
                </label>
              </li>
            ))}
          </ul>
        </>
      )}
      {folder.truncated && (
        <p className="mt-2 text-xs text-ink-500">
          The folder is large; only the first part of it is listed.
        </p>
      )}
      <button
        type="button"
        className="btn btn-primary mt-4"
        disabled={busy || picked.size === 0}
        onClick={onClip}
      >
        {busy ? 'Working…' : `Clip ${picked.size} video${picked.size === 1 ? '' : 's'}`}
      </button>
    </div>
  )
}

/** "Deletes in 3 days" for a job, given the retention setting (0 = off). */
function deletesIn(job: Job, retentionDays: number): string | null {
  if (retentionDays <= 0 || job.status === 'queued' || job.status === 'running') return null
  const created = Date.parse(job.created_at)
  if (Number.isNaN(created)) return null
  const left = Math.ceil((created + retentionDays * 86_400_000 - Date.now()) / 86_400_000)
  if (left <= 0) return 'Deleting soon'
  return `Deletes in ${left} day${left === 1 ? '' : 's'}`
}

function RecentJobs({ jobs, retentionDays }: { jobs: Job[]; retentionDays: number }) {
  if (jobs.length === 0) return null

  return (
    <section className="rise mt-24" style={{ animationDelay: '240ms' }}>
      <div className="flex items-baseline justify-between border-b border-ink-800 pb-3">
        <h2 className="eyebrow">Recent</h2>
        <span className="numeric text-xs text-ink-600">{jobs.length}</span>
      </div>

      <ul>
        {jobs.map((job) => (
          <li key={job.id}>
            <a
              href={job.status === 'done' ? `/jobs/${job.id}/clips` : `/jobs/${job.id}`}
              className="group grid grid-cols-[1fr_auto] items-baseline gap-4 border-b border-ink-850 py-4 transition-colors duration-200 hover:bg-ink-850/40 sm:grid-cols-[1fr_7rem_6rem_5rem]"
            >
              <span className="min-w-0">
                <span className="block truncate text-[0.9375rem] text-ink-200 group-hover:text-ink-100">
                  {job.source?.title || 'Untitled'}
                </span>
                {deletesIn(job, retentionDays) && (
                  <span className="mt-0.5 block text-xs text-ink-600">
                    {deletesIn(job, retentionDays)}
                  </span>
                )}
              </span>
              <span className="numeric hidden text-xs text-ink-500 sm:block">
                {job.source ? formatDuration(job.source.duration_s) : '—'}
              </span>
              <span className="hidden text-xs text-ink-500 sm:block">{job.provider}</span>
              <StatusTag job={job} />
            </a>
            {job.status === 'done' && (
              <a
                href={`/jobs/${job.id}/pick`}
                className="-mt-3 block pb-3 text-right text-[0.6875rem] text-ink-500 transition-colors hover:text-sodium-500"
              >
                Pick parts on the rewatch graph →
              </a>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

function StatusTag({ job }: { job: Job }) {
  const tone: Record<string, string> = {
    done: 'text-signal-good',
    failed: 'text-signal-bad',
    running: 'text-sodium-500',
    queued: 'text-ink-400',
    cancelled: 'text-ink-500',
  }
  const label =
    job.status === 'running' ? `${Math.round(job.progress * 100)}%` : job.status

  return (
    <span className={`numeric justify-self-end text-xs ${tone[job.status] ?? 'text-ink-400'}`}>
      {label}
    </span>
  )
}
