/**
 * Typed client for the AutoClip API.
 *
 * Same-origin throughout: the server hands out the built bundle, so there is no
 * base URL to configure and no CORS to negotiate.
 */

export interface Source {
  id: string
  type: string
  title: string
  url: string | null
  filename: string | null
  channel: string | null
  duration_s: number
  width: number | null
  height: number | null
  fps: number | null
  has_audio: boolean
  has_video: boolean
  created_at: string
}

export type JobStatus = 'queued' | 'running' | 'failed' | 'done' | 'cancelled'

export interface Job {
  id: string
  source_id: string
  status: JobStatus
  current_stage: string
  progress: number
  error: string | null
  provider: string
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  source: Source | null
}

export interface ExportRecord {
  id: string
  clip_id: string
  ratio: string
  style: string
  size_bytes: number
  created_at: string
  download_url: string
}

export type ClipStatus = 'candidate' | 'kept' | 'discarded' | 'exported'

export interface Clip {
  id: string
  job_id: string
  rank: number
  start_s: number
  end_s: number
  duration_s: number
  start_word: number
  end_word: number
  title: string
  hook: string
  score: number
  reason: string
  status: ClipStatus
  user_trimmed: boolean
  caption_style: string
  ratio: string
  /** The job's layout setting; what a re-export uses unless told otherwise. */
  layout: Layout
  exports: ExportRecord[]
  /** Up to three on-screen headline options from detection, best first. */
  hook_options: string[]
  /** The chosen headline as stored ('' = none chosen). */
  hook_text: string
  /** What the render shows: the chosen hook, else the first option, else the title. */
  headline: string
  description: string
  hashtags: string[]
  /** The clip's own flash-forward toggle; null follows `flash_forward_default`. */
  flash_forward: boolean | null
  flash_forward_default: boolean
  /** Whether the model flagged a tease line (otherwise one is picked by rule). */
  has_teaser: boolean
  teaser_start_s: number | null
  teaser_end_s: number | null
  /** The original clip this one re-cuts (null for an original). */
  variant_of: string | null
  motion_seed: number
}

export interface ClipPatch {
  start_s?: number
  end_s?: number
  title?: string
  status?: ClipStatus
  hook_text?: string
  description?: string
  hashtags?: string[]
  /** null goes back to following the export setting. */
  flash_forward?: boolean | null
}

/**
 * Frame layout for 9:16 clips. `auto` stacks a detected facecam overlay above
 * the gameplay (streamer clips) and crops everything else; `split` asks for
 * the stacked layout whenever a facecam is found; `crop` never stacks.
 */
export type Layout = 'auto' | 'split' | 'crop'

export const LAYOUTS: { value: Layout; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'split', label: 'Facecam + gameplay' },
  { value: 'crop', label: 'Crop' },
]

export interface Word {
  text: string
  start: number
  end: number
  speaker: string | null
}

export interface CropKeyframe {
  t: number
  x: number
  y: number
}

export interface CropSegment {
  start_s: number
  end_s: number
  width: number
  height: number
  keyframes: CropKeyframe[]
  strategy: 'track' | 'wide' | 'general'
  zoom: number
  fit: boolean
}

/** Mirrors autoclip.pipeline.reframe.croppath.CropPath. */
export interface CropPath {
  source_width: number
  source_height: number
  segments: CropSegment[]
}

export interface CaptionStyle {
  key: string
  label: string
  description: string
  preview: {
    font: string
    primary: string
    accent: string | null
    outline: string
    outlineWidth: number
    allCaps: boolean
    sizeRatio: number
    marginRatio: number
    boxed: boolean
    animation: string
    maxWords: number
  }
}

export interface ProviderStatus {
  name: string
  available: boolean
  detail: string
  models: string[]
  requires_key: boolean
  has_key: boolean
}

export type Pace = 'off' | 'natural' | 'fast'
/**
 * Language of burned captions, the hook and the post text. `as_spoken` keeps
 * each script except Hindi (written as Hinglish), `roman` writes everything
 * in Latin letters, `english` translates.
 */
export type CaptionLanguage = 'as_spoken' | 'roman' | 'english'
export type Motion = 'off' | 'subtle' | 'dynamic'
export type WatermarkMode = 'off' | 'text' | 'image'
export type WatermarkCorner = 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'

export interface Settings {
  active_provider: string
  providers: Record<string, { model: string; base_url: string | null }>
  whisper: { model: string; compute_type: string; language: string; diarization: boolean }
  clips: { min_duration_s: number; max_duration_s: number; max_clips: number }
  ingest: {
    ytdlp_format: string
    cookies_from_browser: string
    prefer_youtube_captions: boolean
  }
  export: {
    ratio: string
    caption_style: string
    loudness_lufs: number
    prefer_hardware_encoder: boolean
    crf: number
    write_srt: boolean
    layout: Layout
    // Short-form editing (normal-mode exports).
    pace: Pace
    hook_title: boolean
    motion: Motion
    keyword_highlight: boolean
    music_enabled: boolean
    music_path: string
    music_volume: number
    music_duck: boolean
    watermark: WatermarkMode
    watermark_text: string
    watermark_image: string
    watermark_corner: WatermarkCorner
    watermark_opacity: number
    fps: number
    two_pass_loudnorm: boolean
    flash_forward: boolean
    caption_language: CaptionLanguage
  }
  insecure_secret_storage: boolean
  keys_present: Record<string, boolean>
}

/**
 * A settings update. Sections are partial on purpose: the server merges each
 * section field by field, and sending only the changed field means two quick
 * edits in one section can't overwrite each other with stale values.
 */
export interface SettingsPatch {
  active_provider?: string
  providers?: Record<string, Partial<Settings['providers'][string]>>
  whisper?: Partial<Settings['whisper']>
  clips?: Partial<Settings['clips']>
  ingest?: Partial<Settings['ingest']>
  export?: Partial<Settings['export']>
}

export interface SystemStatus {
  ready: boolean
  python_version: string
  platform: string
  ffmpeg_version: string | null
  has_libass: boolean
  nvenc_works: boolean
  accel: string
  gpu_name: string | null
  compute_type: string
  diarization_available: boolean
}

export interface JobSettingsOverrides {
  provider?: string
  whisper_model?: string
  language?: string
  diarization?: boolean
  min_duration_s?: number
  max_duration_s?: number
  max_clips?: number
  caption_style?: string
  ratio?: string
  layout?: Layout
}

/** An API error carrying the server's message and its actionable hint. */
export class ApiError extends Error {
  hint: string
  status: number

  constructor(message: string, status: number, hint = '') {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.hint = hint
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    let hint = ''
    try {
      const body = await response.json()
      const detail = body.detail
      if (typeof detail === 'string') {
        message = detail
      } else if (Array.isArray(detail)) {
        // FastAPI request validation (422): [{loc: ['body', 'settings',
        // 'max_clips'], msg: '...'}]. Without this the user saw only
        // "422 Unprocessable Entity" with no hint of which field was wrong.
        message = detail
          .map((item: { loc?: unknown[]; msg?: string }) => {
            const field = (item.loc ?? []).filter((part) => part !== 'body').join('.')
            return field ? `${field}: ${item.msg ?? 'invalid'}` : (item.msg ?? 'invalid')
          })
          .join('\n')
      } else if (detail && typeof detail === 'object') {
        // The ingest endpoints return {message, hint} so the UI can show the
        // fix alongside the failure.
        message = detail.message ?? message
        hint = detail.hint ?? ''
      }
    } catch {
      /* Response had no JSON body; the status line is the best we have. */
    }
    throw new ApiError(message, response.status, hint)
  }

  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; version: string }>('/api/health'),
  system: () => request<SystemStatus>('/api/system'),
  fetchModels: () => request<void>('/api/system/models', { method: 'POST' }),

  listSources: () => request<Source[]>('/api/sources'),

  ingestYouTube: (url: string, cookiesFromBrowser?: string) =>
    request<Source>('/api/sources/youtube', {
      method: 'POST',
      body: JSON.stringify({ url, cookies_from_browser: cookiesFromBrowser || null }),
    }),

  uploadSource: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Source>('/api/sources/upload', { method: 'POST', body: form })
  },

  listJobs: (limit = 25) => request<Job[]>(`/api/jobs?limit=${limit}`),
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),

  createJob: (sourceId: string, settings: JobSettingsOverrides = {}) =>
    request<Job>('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ source_id: sourceId, settings }),
    }),

  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  retryJob: (id: string) => request<Job>(`/api/jobs/${id}/retry`, { method: 'POST' }),

  listClips: (jobId: string) => request<Clip[]>(`/api/jobs/${jobId}/clips`),
  getClip: (clipId: string) => request<Clip>(`/api/clips/${clipId}`),

  getCropPath: (clipId: string) => request<CropPath>(`/api/clips/${clipId}/crop-path`),
  getClipWords: (clipId: string) => request<Word[]>(`/api/clips/${clipId}/words`),

  patchClip: (clipId: string, patch: ClipPatch) =>
    request<Clip>(`/api/clips/${clipId}`, { method: 'PATCH', body: JSON.stringify(patch) }),

  /** Same moment, next caption style / zoom pattern / hook; rendered unless `exportNow` is false. */
  makeVariant: (clipId: string, options: { exportNow?: boolean; layout?: Layout } = {}) =>
    request<Clip>(`/api/clips/${clipId}/variant`, {
      method: 'POST',
      body: JSON.stringify({ export: options.exportNow ?? true, layout: options.layout }),
    }),

  patchCaptions: (
    clipId: string,
    patch: { words?: Word[]; caption_style?: string; ratio?: string },
  ) =>
    request<Clip>(`/api/clips/${clipId}/captions`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  exportClip: (
    clipId: string,
    ratio: string,
    style: string,
    writeSrt = false,
    layout?: Layout,
  ) =>
    request<ExportRecord>(`/api/clips/${clipId}/export`, {
      method: 'POST',
      body: JSON.stringify({ ratio, style, write_srt: writeSrt, layout }),
    }),

  captionStyles: () => request<CaptionStyle[]>('/api/caption-styles'),
  providerStatus: () => request<ProviderStatus[]>('/api/providers/status'),

  getSettings: () => request<Settings>('/api/settings'),
  putSettings: (patch: SettingsPatch) =>
    request<Settings>('/api/settings', { method: 'PUT', body: JSON.stringify(patch) }),

  putSecret: (key: string, value: string) =>
    request<void>('/api/settings/secrets', {
      method: 'PUT',
      body: JSON.stringify({ key, value }),
    }),

  deleteSecret: (key: string) =>
    request<void>(`/api/settings/secrets/${key}`, { method: 'DELETE' }),

  mediaUrl: (jobId: string) => `/api/jobs/${jobId}/media`,
}

/** Format seconds as m:ss, or h:mm:ss past an hour. */
export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const s = total % 60
  const m = Math.floor(total / 60) % 60
  const h = Math.floor(total / 3600)
  const pad = (n: number) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`
}

/** Format seconds as timecode with centiseconds, for trim handles. */
export function formatTimecode(seconds: number): string {
  const total = Math.max(0, seconds)
  const cs = Math.floor((total % 1) * 100)
  const s = Math.floor(total) % 60
  const m = Math.floor(total / 60)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(m)}:${pad(s)}.${pad(cs)}`
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  const units = ['KB', 'MB', 'GB']
  let value = bytes / 1024
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`
}
