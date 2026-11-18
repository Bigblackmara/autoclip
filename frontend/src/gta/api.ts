/** GTA Mode API client and types (backend: autoclip/api/gta.py). */

import { request } from '../api'

export type StreamStatus =
  | 'queued'
  | 'probing'
  | 'downloading_audio'
  | 'fetching_signals'
  | 'transcribing'
  | 'building_hype'
  | 'judging'
  | 'analyzed'
  | 'done'
  | 'failed'
  | 'cancelled'
  | 'live_connecting'
  | 'live_watching'
  | 'live_ended'
  | 'live_stopped'

export type ClipStatus =
  | 'candidate'
  | 'rejected'
  | 'approved'
  | 'downloading'
  | 'rendering'
  | 'ready'
  | 'qa_failed'
  | 'failed'

export interface GtaStream {
  id: string
  url: string
  platform: string
  title: string
  channel: string
  duration_s: number
  range_start_s: number | null
  range_end_s: number | null
  status: StreamStatus
  stage: string
  progress: number
  detail: string
  error: string | null
  signals: Record<string, unknown>
  is_live: boolean
  clip_counts: Partial<Record<ClipStatus, number>>
  created_at: string
  updated_at: string
}

export interface QaCheck {
  ok: boolean
  detail: string
  value: unknown
}

export interface Answer {
  answer: 'yes' | 'no'
  why: string
}

export interface GtaClip {
  id: string
  stream_id: string
  status: ClipStatus
  start_s: number
  end_s: number
  duration_s: number
  hook_title: string
  hook_quote: string
  moment_type: string
  summary: string
  hype_score: number
  score: number
  rank: number | null
  verdict: Partial<Record<'scroll_stop' | 'context_free' | 'payoff', Answer>>
  reject_reason: string
  caption: string
  hashtags: string[]
  post_text: string
  path: string
  has_video: boolean
  qa: {
    passed?: boolean
    length?: QaCheck
    loudness?: QaCheck
    black_frames?: QaCheck
    facecam?: QaCheck
    format?: QaCheck
    own_value?: QaCheck
    watermark?: QaCheck
    notes?: string[]
    cut_dead_air_s?: number
    duration_s?: number
    turnaround_s?: number
    turnaround_ok?: boolean
  }
  error: string | null
  uploads: Partial<Record<Platform, GtaUpload>>
}

export type Platform = 'youtube' | 'instagram'

export const PLATFORMS: readonly Platform[] = ['youtube', 'instagram']

export const PLATFORM_NAME: Record<Platform, string> = { youtube: 'YouTube', instagram: 'Instagram' }

export const PLATFORM_SHORT: Record<Platform, string> = { youtube: 'YT', instagram: 'IG' }

/**
 * checking: sent to Zernio, outcome not confirmed yet (AutoClip keeps asking
 * and never posts it twice). posted: confirmed live on Instagram. Instagram
 * "scheduled" = scheduled at Zernio, goes live at slot_at.
 */
export type UploadStatus =
  | 'uploading'
  | 'scheduled'
  | 'checking'
  | 'posted'
  | 'failed'
  | 'interrupted'
  | 'skipped'

export interface GtaUpload {
  id: string
  clip_id: string
  platform: Platform
  status: UploadStatus
  /** UTC ISO time the post goes public. */
  slot_at: string | null
  video_id: string
  url: string
  media_url: string
  title: string
  error: string | null
  attempts: number
  updated_at: string
}

export interface GtaPosting {
  enabled: boolean
  youtube: boolean
  instagram: boolean
  /** Optional Instagram username to post as (when several are connected at Zernio). */
  instagram_account: string
  slots: string[]
  lead_minutes: number
  category_id: string
  shorts_tag: boolean
}

export interface PlanEntry {
  slot_at: string
  platform: Platform
  status: 'open' | 'missed' | 'planned' | UploadStatus
  clip_id: string | null
  upload_id?: string
  title: string
}

export interface PlatformState {
  message: string
  paused_until: string | null
  uploading_clip_id: string | null
  last_run: string | null
}

export interface InstagramStatus {
  key_saved: boolean
  reelrepost_key_available: boolean
}

export interface PostingOverview {
  states: Record<Platform, PlatformState>
  running: boolean
  posting: GtaPosting
  categories: Record<string, string>
  plan: PlanEntry[]
  uploads: (GtaUpload & { clip_title: string })[]
}

/** Max upload attempts per clip (backend gta/poster.py MAX_ATTEMPTS). */
export const MAX_UPLOAD_ATTEMPTS = 3

export type GtaCaptionStyle = 'gta' | 'pop' | 'neon' | 'comic' | 'minimal' | 'karaoke_fill'
export type GtaMotion = 'off' | 'subtle' | 'dynamic'

export const GTA_CAPTION_STYLES: { value: GtaCaptionStyle; label: string; hint: string }[] = [
  { value: 'gta', label: 'GTA (classic)', hint: 'White with a black outline; the spoken word turns yellow.' },
  { value: 'pop', label: 'Pop', hint: 'Big three-word bursts; spoken word gold, key words green.' },
  { value: 'neon', label: 'Neon', hint: 'Glowing cyan outline; key words magenta.' },
  { value: 'comic', label: 'Comic', hint: 'Yellow comic lettering, two words at a time; key words red.' },
  { value: 'minimal', label: 'Minimal', hint: 'Clean sentence case with a soft shadow; only key words take colour.' },
  { value: 'karaoke_fill', label: 'Karaoke Fill', hint: 'Words fill with colour as they are spoken.' },
]

export const GTA_MOTION_LEVELS: { value: GtaMotion; label: string }[] = [
  { value: 'off', label: 'Off (punch-in only)' },
  { value: 'subtle', label: 'Subtle (zoom every ~4.5 s)' },
  { value: 'dynamic', label: 'Dynamic (zoom every ~2 s)' },
]

export interface GtaSettings {
  chunk_minutes: number
  chunk_tail_s: number
  min_clip_s: number
  target_clip_s: number
  max_clip_s: number
  max_clips_per_stream: number
  section_pad_s: number
  dead_air_min_s: number
  dead_air_keep_s: number
  hook_title_s: number
  caption_style: GtaCaptionStyle
  keyword_highlight: boolean
  motion: GtaMotion
  comment_signal: boolean
  youtube_captions: boolean
  loudness_lufs: number
  loudness_tolerance_lu: number
  true_peak_db: number
  face_min_ratio: number
  analysis_format: string
  clip_format: string
  network_workers: number
  heavy_workers: number
  output_dir: string
  punch_in: boolean
  punch_in_zoom: number
  punch_in_s: number
  reaction_text: boolean
  watermark_path: string
  watermark_text: string
  watermark_font: string
  watermark_corner: 'top_left' | 'top_right'
  watermark_width_px: number
  watermark_opacity: number
  live: { listen_interval_s: number; context_s: number; turnaround_s: number }
  posting: GtaPosting
  bias: {
    moment_types: Record<string, number>
    streamers: Record<string, number>
    summary: string
    updated_at: string
    channel_url: string
  }
  twitch_credentials_set: boolean
}

export interface GroupStat {
  count: number
  average_views: number
  vs_channel: number
}

export interface FeedbackReport {
  channel_average: number
  count: number
  matched: number
  moment_types: Record<string, GroupStat>
  streamers: Record<string, GroupStat>
  bias: GtaSettings['bias']
  shorts: { title: string; views: number; url: string; moment_type: string; streamer: string }[]
}

export const gtaApi = {
  streams: () => request<GtaStream[]>('/api/gta/streams'),
  addStreams: (urls: string[], range: { start?: number | null; end?: number | null }) =>
    request<{ added: GtaStream[]; skipped: { url: string; reason: string }[] }>('/api/gta/streams', {
      method: 'POST',
      body: JSON.stringify({
        urls,
        range_start_s: range.start ?? null,
        range_end_s: range.end ?? null,
      }),
    }),
  cancel: (id: string) => request<GtaStream>(`/api/gta/streams/${id}/cancel`, { method: 'POST' }),
  retry: (id: string) => request<GtaStream>(`/api/gta/streams/${id}/retry`, { method: 'POST' }),
  remove: (id: string) => request<void>(`/api/gta/streams/${id}`, { method: 'DELETE' }),
  clips: () => request<GtaClip[]>('/api/gta/clips'),
  videoUrl: (clipId: string) => `/api/gta/clips/${clipId}/video`,
  logoUrl: (version: string) => `/api/gta/logo?v=${encodeURIComponent(version)}`,
  startLive: (url: string) =>
    request<GtaStream>('/api/gta/live', { method: 'POST', body: JSON.stringify({ url }) }),
  stopLive: (id: string) => request<GtaStream>(`/api/gta/live/${id}/stop`, { method: 'POST' }),
  settings: () => request<GtaSettings>('/api/gta/settings'),
  saveSettings: (patch: Partial<GtaSettings>) =>
    request<GtaSettings>('/api/gta/settings', { method: 'PUT', body: JSON.stringify(patch) }),
  saveTwitch: (client_id: string, client_secret: string) =>
    request<{ twitch_credentials_set: boolean }>('/api/gta/twitch-credentials', {
      method: 'PUT',
      body: JSON.stringify({ client_id, client_secret }),
    }),
  feedback: () =>
    request<{ id: string; channel_url: string; created_at: string; report: FeedbackReport } | null>(
      '/api/gta/feedback',
    ),
  runFeedback: (channel_url: string) =>
    request<FeedbackReport>('/api/gta/feedback', {
      method: 'POST',
      body: JSON.stringify({ channel_url }),
    }),
  posting: () => request<PostingOverview>('/api/gta/posting'),
  runPosting: () => request<{ running: boolean }>('/api/gta/posting/run', { method: 'POST' }),
  setClipPosting: (clipId: string, skip: boolean) =>
    request<{ uploads: Partial<Record<Platform, GtaUpload>> }>(`/api/gta/clips/${clipId}/posting`, {
      method: 'POST',
      body: JSON.stringify({ skip }),
    }),
  retryUpload: (uploadId: string) =>
    request<{ retried: boolean }>(`/api/gta/uploads/${uploadId}/retry`, { method: 'POST' }),
  /** Instagram posting through Zernio: is an API key saved (or importable from ReelRepost)? */
  instagram: () => request<InstagramStatus>('/api/gta/instagram'),
  /** Save the Zernio API key to the OS keyring; an empty string removes it. */
  saveInstagramKey: (api_key: string) =>
    request<InstagramStatus>('/api/gta/instagram/key', {
      method: 'PUT',
      body: JSON.stringify({ api_key }),
    }),
  importReelRepostKey: () =>
    request<InstagramStatus>('/api/gta/instagram/import-reelrepost', { method: 'POST' }),
  testInstagram: (account: string) =>
    request<{ ok: boolean; message: string; username: string }>('/api/gta/instagram/test', {
      method: 'POST',
      body: JSON.stringify({ account }),
    }),
  /** Google consent for the shared YouTube connection, returning to GTA Mode. */
  connectYoutube: () =>
    request<{ redirect_url: string }>('/api/live/uploads/youtube/start?return_to=/gta', {
      method: 'POST',
    }),
}

/** "Tue 17:00" in this PC's local time. */
export function slotLabel(iso: string | null): string {
  if (!iso) return ''
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return date.toLocaleString([], { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export const STATUS_LABEL: Record<StreamStatus, string> = {
  queued: 'Queued',
  probing: 'Reading link',
  downloading_audio: 'Downloading audio',
  fetching_signals: 'Chat & signals',
  transcribing: 'Transcribing',
  building_hype: 'Hype Curve',
  judging: 'Judging',
  analyzed: 'Rendering clips',
  done: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
  live_connecting: 'Connecting',
  live_watching: 'Watching live',
  live_ended: 'Stream ended',
  live_stopped: 'Stopped',
}

export const MOMENT_LABEL: Record<string, string> = {
  first_reaction: 'First reaction',
  crazy_discovery: 'Crazy discovery',
  freakout: 'Freakout',
  funny_fail: 'Funny fail',
  hot_take: 'Hot take',
  other: 'Other',
}

/** "1:02:03" / "4:05" / "90" → seconds; empty → null; anything else → NaN. */
export function parseClock(text: string): number | null {
  const value = text.trim()
  if (!value) return null
  const parts = value.split(':').map((p) => p.trim())
  if (parts.length > 3 || parts.some((p) => !/^\d+(\.\d+)?$/.test(p))) return Number.NaN
  return parts.reduce((total, part) => total * 60 + Number(part), 0)
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = String(s % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`
}

/** Statuses during which the VOD queue is still working on a stream. */
export const ACTIVE: ReadonlySet<StreamStatus> = new Set<StreamStatus>([
  'queued',
  'probing',
  'downloading_audio',
  'fetching_signals',
  'transcribing',
  'building_hype',
  'judging',
  'analyzed',
])

export const LIVE_RUNNING: ReadonlySet<StreamStatus> = new Set<StreamStatus>([
  'live_connecting',
  'live_watching',
])
