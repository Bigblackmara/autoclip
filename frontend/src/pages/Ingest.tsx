import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'

import {
  ApiError,
  api,
  detectLink,
  formatBytes,
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
import { JobList } from '../components/JobList'
import {
  ArrowRightIcon,
  Button,
  ButtonLink,
  Callout,
  Card,
  Checkbox,
  ChevronDownIcon,
  CloudIcon,
  CompassIcon,
  Field,
  FolderIcon,
  Input,
  LinkIcon,
  PageHeader,
  ProgressBar,
  Segmented,
  Select,
  SlidersIcon,
  SparklesIcon,
  UploadIcon,
  cx,
} from '../ui'

type SourceTab = 'link' | 'file' | 'drive'

/** Short labels on phones so the three choices fit on one row. */
const SOURCE_TABS = [
  { value: 'link', label: <Short full="Paste a link" short="Link" /> },
  { value: 'file', label: <Short full="Upload a file" short="Upload" /> },
  { value: 'drive', label: <Short full="Google Drive folder" short="Drive folder" /> },
] as const

function Short({ full, short }: { full: string; short: string }) {
  return (
    <>
      <span className="sm:hidden">{short}</span>
      <span className="hidden sm:inline">{full}</span>
    </>
  )
}

export function Ingest() {
  const navigate = useNavigate()
  // Explore's "Clip this video" lands here as /?url=…, prefilled.
  const [searchParams] = useSearchParams()
  const [url, setUrl] = useState(() => searchParams.get('url') ?? '')
  const [tab, setTab] = useState<SourceTab>('link')
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
  const isFolder = linkKind === 'gdrive_folder'

  return (
    <div>
      <PageHeader
        title="Make new clips"
        description="Paste a link or upload a long video. AutoClip finds the moments worth sharing and turns them into captioned vertical clips."
      />

      {providers.length > 0 && !usableProvider && (
        <Callout
          tone="warn"
          title="No AI provider is set up yet"
          className="mb-6"
          action={
            <ButtonLink to="/settings" size="sm" variant="secondary">
              Open Settings
            </ButtonLink>
          }
        >
          Clip picking needs one. Add an API key or start Ollama in Settings → General &amp; AI.
        </Callout>
      )}

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_19rem]">
        {/* The start card. Dropping a file anywhere on it uploads that file. */}
        <Card
          padding="lg"
          className={cx('relative', dragging && 'ring-2 ring-accent')}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDragging(false)
          }}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            const file = e.dataTransfer.files[0]
            if (file) submitFile(file)
          }}
        >
          <Segmented
            label="Where is your video?"
            options={SOURCE_TABS}
            value={tab}
            onChange={(next) => {
              setTab(next)
              setError(null)
            }}
          />

          {tab !== 'file' ? (
            <form onSubmit={submitUrl} className="mt-6">
              <label htmlFor="url" className="text-sm font-medium text-fg">
                {tab === 'drive' ? 'Google Drive folder link' : 'Video link'}
              </label>
              <div className="mt-2 flex flex-col gap-3 sm:flex-row">
                <div className="relative min-w-0 flex-1">
                  <span className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-fg-subtle">
                    {tab === 'drive' ? <FolderIcon size={18} /> : <LinkIcon size={18} />}
                  </span>
                  <Input
                    id="url"
                    className="py-3 pl-11 text-base"
                    placeholder={
                      tab === 'drive'
                        ? 'https://drive.google.com/drive/folders/…'
                        : 'https://www.youtube.com/watch?v=…'
                    }
                    value={url}
                    onChange={(e) => {
                      setUrl(e.target.value)
                      setFolder(null)
                    }}
                    autoComplete="off"
                    spellCheck={false}
                    disabled={busy !== null}
                  />
                </div>
                <Button
                  type="submit"
                  variant="primary"
                  size="lg"
                  icon={isFolder ? FolderIcon : SparklesIcon}
                  loading={busy === 'url'}
                  disabled={busy !== null || !url.trim()}
                >
                  {busy === 'url' ? 'Fetching…' : isFolder ? 'List videos' : 'Find clips'}
                </Button>
              </div>
              <p className="mt-2 text-xs text-fg-muted">
                {linkKind ? (
                  <span className="inline-flex items-center gap-1.5 font-medium text-accent-text">
                    <LinkKindIcon kind={linkKind} />
                    {LINK_LABELS[linkKind]}
                    {isFolder && ' · pick the videos to clip'}
                  </span>
                ) : tab === 'drive' ? (
                  'Share the folder as "Anyone with the link". You can pick which videos to clip.'
                ) : (
                  'Works with YouTube, Google Drive, Dropbox, OneDrive and direct .mp4 links.'
                )}
              </p>

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
            </form>
          ) : (
            <div className="mt-6">
              <div
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
                className={cx(
                  'flex min-h-48 cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-8 text-center transition-colors duration-200',
                  dragging ? 'border-accent bg-accent-soft' : 'border-line-strong hover:border-accent hover:bg-subtle',
                )}
              >
                <span className="mb-1 grid size-12 place-items-center rounded-2xl bg-accent-soft text-accent-text">
                  <UploadIcon size={22} />
                </span>
                <span className="text-base font-semibold text-fg">
                  {busy === 'file' ? 'Uploading…' : 'Drop a video or audio file here'}
                </span>
                <span className="text-sm text-fg-muted">
                  or <span className="font-medium text-accent-text">browse your computer</span>
                </span>
                <span className="mt-1 text-xs text-fg-subtle">mp4 · mov · mkv · webm · mp3 · wav · m4a</span>
              </div>
            </div>
          )}
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

          {notice && <p className="mt-4 text-sm text-fg-muted">{notice}</p>}

          {error && (
            <div className="mt-5">
              <ErrorNote error={error} onDismiss={() => setError(null)} />
            </div>
          )}

          <AdvancedOptions
            open={advancedOpen}
            onToggle={() => setAdvancedOpen((v) => !v)}
            overrides={overrides}
            onChange={setOverrides}
            providers={providers}
            defaults={defaults}
          />

          <p className="mt-6 border-t border-line pt-4 text-xs leading-relaxed text-fg-subtle">
            Only download video you own or have the rights to process. Everything runs on this computer.
          </p>
        </Card>

        <div className="grid gap-6 sm:grid-cols-2 xl:flex xl:flex-col">
          <Card padding="md" title="Not sure what to clip?" icon={CompassIcon}>
            <p className="-mt-2 text-sm leading-relaxed text-fg-muted">
              Search YouTube and Twitch, peek at where viewers rewatch, then send a video here.
            </p>
            <ButtonLink to="/explore" variant="secondary" className="mt-4" block iconRight={ArrowRightIcon}>
              Open Explore
            </ButtonLink>
          </Card>
          <Card padding="md" title="How it works">
            <ol className="-mt-1 space-y-3 text-sm">
              {[
                ['Add a video', 'A link, a file or a whole Drive folder.'],
                ['AutoClip finds the moments', 'It transcribes, picks highlights and reframes to 9:16.'],
                ['Review and export', 'Trim, edit captions, then download your clips.'],
              ].map(([title, text], i) => (
                <li key={title} className="flex gap-3">
                  <span className="numeric grid size-6 shrink-0 place-items-center rounded-full bg-accent-soft text-xs font-semibold text-accent-text">
                    {i + 1}
                  </span>
                  <span>
                    <span className="font-medium text-fg">{title}</span>
                    <span className="block text-fg-muted">{text}</span>
                  </span>
                </li>
              ))}
            </ol>
          </Card>
        </div>
      </div>

      {jobs.length > 0 && (
        <Card
          padding="none"
          className="mt-8 overflow-hidden"
          title="Recent projects"
          description="Open one to review its clips."
          actions={
            <Link to="/projects" className="text-sm font-medium text-accent-text hover:underline">
              View all
            </Link>
          }
        >
          <div className="border-t border-line">
            <JobList jobs={jobs} retentionDays={defaults?.housekeeping?.retention_days ?? 0} />
          </div>
        </Card>
      )}
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
  const changed = Object.values(overrides).filter((v) => v !== undefined).length

  return (
    <div className="mt-6">
      <button
        type="button"
        onClick={onToggle}
        className="inline-flex items-center gap-2 rounded-lg px-1 py-1 text-sm font-medium text-fg-muted transition-colors hover:text-fg"
        aria-expanded={open}
        aria-controls="job-options"
      >
        <SlidersIcon size={16} />
        Advanced options
        {changed > 0 && <span className="rounded-full bg-accent-soft px-1.5 text-xs text-accent-text">{changed} changed</span>}
        <ChevronDownIcon size={16} className={cx('transition-transform duration-200', open && 'rotate-180')} />
      </button>

      {/* grid-template-rows rather than height, so the reveal animates without
          touching layout properties. */}
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]"
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
          <p className="pt-3 text-xs text-fg-muted">
            Just for this video. Leave anything on “Use default” to use your Settings.
          </p>
          <div className="grid gap-x-6 gap-y-5 pt-4 pb-1 sm:grid-cols-2">
            <Field label="AI provider" hint="Which AI picks the clips.">
              <Select
                value={overrides.provider ?? ''}
                onChange={(e) => set('provider', e.target.value || undefined)}
                options={[
                  { value: '', label: 'Use default' },
                  ...providers.map((p) => ({
                    value: p.name,
                    label: p.available ? p.name : `${p.name} — unavailable`,
                    disabled: !p.available,
                  })),
                ]}
              />
            </Field>
            <Field label="Transcription quality" hint="Bigger models are slower but more accurate.">
              <Select
                value={overrides.whisper_model ?? ''}
                onChange={(e) => set('whisper_model', e.target.value || undefined)}
                options={[
                  { value: '', label: 'Use default' },
                  { value: 'tiny', label: 'tiny — fastest, roughest' },
                  { value: 'base', label: 'base' },
                  { value: 'small', label: 'small — balanced' },
                  { value: 'medium', label: 'medium' },
                  { value: 'large-v3', label: 'large-v3 — slowest, best' },
                ]}
              />
            </Field>
            {/* Streams with a facecam overlay: stack the cam above the
                gameplay instead of cropping it out of the 9:16 frame. */}
            <Field label="Layout (9:16)" hint="Gameplay streams: stack the facecam above the game.">
              <Select
                value={overrides.layout ?? ''}
                onChange={(e) => set('layout', (e.target.value || undefined) as Layout | undefined)}
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
            </Field>
            <NumberField
              label="How many clips (at most)"
              value={overrides.max_clips}
              placeholder={String(defaults?.clips.max_clips ?? 10)}
              min={1}
              max={50}
              onChange={(v) => set('max_clips', v)}
            />
            <NumberField
              label="Shortest clip (seconds)"
              value={overrides.min_duration_s}
              placeholder={String(defaults?.clips.min_duration_s ?? 20)}
              min={5}
              max={300}
              onChange={(v) => set('min_duration_s', v)}
            />
            <NumberField
              label="Longest clip (seconds)"
              value={overrides.max_duration_s}
              placeholder={String(defaults?.clips.max_duration_s ?? 90)}
              min={5}
              max={300}
              onChange={(v) => set('max_duration_s', v)}
            />
            {/* Shows the saved default until touched, and sends an explicit
                true/false once it is: "unchecked" used to mean "use the
                default", so a job could not opt out when the default was on. */}
            <Checkbox
              className="sm:col-span-2"
              label="Multiple speakers"
              hint="Label who is talking, and cut to them."
              checked={overrides.diarization ?? defaults?.whisper.diarization ?? false}
              onChange={(checked) => set('diarization', checked)}
            />
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
    <Field label={label} hint={`Default: ${placeholder}`}>
      <Input
        type="number"
        placeholder={placeholder}
        value={value ?? ''}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
      />
    </Field>
  )
}

function LinkKindIcon({ kind }: { kind: LinkKind }) {
  if (kind === 'gdrive_folder') return <FolderIcon size={14} />
  if (kind === 'direct') return <LinkIcon size={14} />
  if (kind === 'youtube') return <SparklesIcon size={14} />
  return <CloudIcon size={14} />
}

function DownloadProgress({ progress }: { progress: LinkProgress }) {
  const { done_bytes: done, total_bytes: total } = progress
  const fraction = total ? Math.min(1, done / total) : null
  return (
    <div className="mt-4 max-w-md" aria-live="polite">
      <ProgressBar value={fraction ?? 0} label="Download progress" />
      <p className="numeric mt-1.5 text-xs text-fg-muted">
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
    <div className="mt-5 rounded-xl border border-line bg-subtle/60 p-4">
      <div className="flex items-baseline justify-between gap-4">
        <p className="flex min-w-0 items-center gap-2 text-sm font-semibold text-fg">
          <FolderIcon size={16} className="text-fg-subtle" />
          <span className="truncate">{folder.name}</span>
        </p>
        <span className="numeric shrink-0 text-xs text-fg-muted">
          {folder.files.length} video{folder.files.length === 1 ? '' : 's'}
        </span>
      </div>
      {folder.files.length === 0 ? (
        <p className="mt-3 text-sm text-fg-muted">No video or audio files in this folder.</p>
      ) : (
        <>
          <label className="mt-3 flex items-center gap-3 text-sm font-medium text-fg-muted">
            <input
              type="checkbox"
              checked={all}
              onChange={() => onChange(all ? new Set() : new Set(folder.files.map((f) => f.id)))}
              className="size-4"
              disabled={busy}
            />
            Select all
          </label>
          <ul className="mt-2 max-h-64 overflow-y-auto rounded-lg border border-line bg-surface">
            {folder.files.map((file) => (
              <li key={file.id} className="border-b border-line last:border-b-0">
                <label className="flex items-center gap-3 px-3 py-2 text-sm text-fg">
                  <input
                    type="checkbox"
                    checked={picked.has(file.id)}
                    onChange={() => toggle(file.id)}
                    className="size-4 shrink-0"
                    disabled={busy}
                  />
                  <span className="truncate">{file.name}</span>
                  {file.folder && <span className="ml-auto shrink-0 text-xs text-fg-subtle">{file.folder}</span>}
                </label>
              </li>
            ))}
          </ul>
        </>
      )}
      {folder.truncated && (
        <p className="mt-2 text-xs text-fg-muted">The folder is large; only the first part of it is listed.</p>
      )}
      <Button variant="primary" className="mt-4" disabled={busy || picked.size === 0} loading={busy} onClick={onClip}>
        {busy ? 'Working…' : `Clip ${picked.size} video${picked.size === 1 ? '' : 's'}`}
      </Button>
    </div>
  )
}
