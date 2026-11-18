import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import {
  ApiError,
  api,
  formatDuration,
  LAYOUTS,
  type Job,
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
  const fileInput = useRef<HTMLInputElement>(null)
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
  }, [])

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

  const submitUrl = (event: React.FormEvent) => {
    event.preventDefault()
    if (!url.trim()) return
    void start('url', () => api.ingestYouTube(url.trim()))
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
                placeholder="https://youtube.com/watch?v=…"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                autoComplete="off"
                spellCheck={false}
                disabled={busy !== null}
              />
              <button
                type="submit"
                className="btn btn-primary shrink-0"
                disabled={busy !== null || !url.trim()}
              >
                {busy === 'url' ? 'Fetching…' : 'Start'}
              </button>
            </div>
          </form>

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

      <RecentJobs jobs={jobs} />
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

function RecentJobs({ jobs }: { jobs: Job[] }) {
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
              <span className="truncate text-[0.9375rem] text-ink-200 group-hover:text-ink-100">
                {job.source?.title || 'Untitled'}
              </span>
              <span className="numeric hidden text-xs text-ink-500 sm:block">
                {job.source ? formatDuration(job.source.duration_s) : '—'}
              </span>
              <span className="hidden text-xs text-ink-500 sm:block">{job.provider}</span>
              <StatusTag job={job} />
            </a>
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
