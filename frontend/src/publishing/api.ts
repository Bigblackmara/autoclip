/**
 * Posting, Schedule, Autopilot and Health API client.
 * Backend: autoclip/api/publishing.py and autoclip/api/autopilot.py.
 */

import { request } from '../api'

export type Platform = 'youtube' | 'instagram'

export type PostStatus =
  | 'queued'
  | 'uploading'
  | 'scheduled'
  | 'posted'
  | 'failed'
  | 'interrupted'
  | 'checking'
  | 'canceled'
  | 'planned'
  | 'skipped'

export interface PostRecord {
  id: string
  clip_id: string
  platform: Platform
  status: PostStatus
  job_id: string
  scheduled_at: string | null
  privacy: 'public' | 'unlisted' | 'private'
  title: string
  caption: string
  hashtags: string[]
  remote_id: string
  url: string
  media_url: string
  error: string | null
  attempts: number
  autopilot_id: string
  slot_key: string
  sent_at: string | null
  created_at: string
  updated_at: string
}

export interface ScheduleItem {
  key: string
  source: 'normal' | 'autopilot' | 'gta'
  id: string
  clip_id: string
  job_id: string
  autopilot_id: string
  platform: Platform
  status: PostStatus
  when: string | null
  scheduled_at: string | null
  title: string
  url: string
  error: string | null
  attempts: number
  upcoming: boolean
  can_cancel: boolean
  cancel_options: ('private' | 'delete')[]
  cancel_note: string
  can_retry: boolean
  retry_confirm: boolean
}

export interface ScheduleOut {
  now: string
  youtube_can_edit: boolean
  upcoming: ScheduleItem[]
  past: ScheduleItem[]
  counts: { upcoming: number; posted: number; failed: number }
}

export interface PostableClip {
  clip_id: string
  job_id: string
  title: string
  text: string
  hashtags: string[]
  source_title: string
  export_id: string
  download_url: string | null
  active: Record<Platform, PostRecord | null>
}

/** GET /api/publishing/clips/{clip_id}: one clip's post text and its posts. */
export interface ClipPosting {
  clip_id: string
  title: string
  text: string
  hashtags: string[]
  /** False until the clip has an exported file to upload. */
  exported: boolean
  posts: PostRecord[]
  active: Record<Platform, PostRecord | null>
}

/** What the Post panel needs about a clip (a PostableClip or a ClipPosting both fit). */
export type PostTarget = Pick<PostableClip, 'clip_id' | 'title' | 'text' | 'hashtags' | 'active'>

export interface PublishSettings {
  category_id: string
  shorts_tag: boolean
  lead_minutes: number
  instagram_account: string
}

export interface PosterState {
  message: string
  paused_until: string | null
  uploading_clip_id: string | null
  last_run: string | null
}

export interface PublishingStatus {
  running: boolean
  states: Record<Platform, PosterState>
  ready: { youtube: string; instagram: string; youtube_can_edit: boolean }
  settings: PublishSettings
}

export interface NewPost {
  clip_id: string
  platforms: Platform[]
  scheduled_at: string | null
  privacy?: 'public' | 'unlisted' | 'private'
  title?: string
  caption?: string
}

// ---------------------------------------------------------------- Autopilot

export interface AutopilotSource {
  kind: 'channel' | 'topic'
  value: string
  country: string
  language: string
}

export interface AutopilotFilters {
  min_views: number
  min_minutes: number
  max_minutes: number
  max_age_days: number
}

export interface AutopilotInput {
  name: string
  sources: AutopilotSource[]
  filters: AutopilotFilters
  clips_per_video: number
  slots: string[]
  timezone: string
  platforms: Platform[]
  approve_before_posting: boolean
  heatmap_wait_hours: number
  rights_confirmed: boolean
}

export interface AutopilotCounts {
  videos_waiting: number
  videos_clipping: number
  videos_seen: number
  clips_review: number
  clips_ready: number
  clips_booked: number
  clips_posted: number
  stock: number
}

export interface Autopilot extends AutopilotInput {
  id: string
  enabled: boolean
  created_at: string
  updated_at: string
  state: {
    last_scan_at: string | null
    last_run_at: string | null
    last_error: string | null
    message: string | null
  }
  counts: AutopilotCounts
}

export interface AutopilotVideo {
  id: string
  url: string
  title: string
  channel: string
  views: number | null
  duration_s: number | null
  upload_date: string | null
  source: string
  status: 'waiting' | 'ready' | 'clipping' | 'done' | 'skipped' | 'failed'
  reason: string
  has_heatmap: boolean
  first_seen: string
  last_checked: string | null
  job_id: string
}

export interface AutopilotClip {
  clip_id: string
  job_id: string
  video_id: string
  title: string
  score: number
  status: 'review' | 'ready' | 'booked' | 'posted' | 'rejected' | 'failed'
  slot_key: string
  post_ids: string[]
  reason: string
  created_at: string
}

export interface AutopilotRun {
  at: string
  scanned: boolean
  found: number
  checked: number
  started: string | null
  harvested: number
  booked: number
  errors: string[]
  message: string
}

export interface AutopilotDetail extends Autopilot {
  runs: AutopilotRun[]
  videos: AutopilotVideo[]
  clips: AutopilotClip[]
  posts: PostRecord[]
  plan: { slot_at: string; clip_id: string | null; title: string | null; status: string }[]
}

// ---------------------------------------------------------------- Health

export type CheckStatus = 'ok' | 'warn' | 'fail'

export interface HealthCheck {
  id: string
  group: string
  label: string
  status: CheckStatus
  detail: string
  fix: string
}

export interface DoctorReport {
  generated_at: string
  live: boolean
  summary: { status: CheckStatus; ok: number; warn: number; fail: number }
  checks: HealthCheck[]
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const publishingApi = {
  status: () => request<PublishingStatus>('/api/publishing/status'),
  saveSettings: (settings: PublishSettings) =>
    request<PublishSettings>('/api/publishing/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
  clips: (limit = 40) => request<PostableClip[]>(`/api/publishing/clips?limit=${limit}`),
  clip: (clipId: string) => request<ClipPosting>(`/api/publishing/clips/${clipId}`),
  createPosts: (post: NewPost) =>
    request<{ posts: PostRecord[]; warnings: string[] }>('/api/publishing/posts', json(post)),
  cancel: (id: string, action: 'private' | 'delete' = 'private') =>
    request<PostRecord>(`/api/publishing/posts/${id}/cancel`, json({ action })),
  retry: (id: string) => request<PostRecord>(`/api/publishing/posts/${id}/retry`, json({})),
  retryGta: (id: string) => request<{ retried: boolean }>(`/api/gta/uploads/${id}/retry`, json({})),
  schedule: () => request<ScheduleOut>('/api/schedule'),
  doctor: (live = false, refresh = false) =>
    request<DoctorReport>(`/api/system/doctor?live=${live}&refresh=${refresh}`),
}

export const autopilotApi = {
  list: () => request<{ running: boolean; autopilots: Autopilot[] }>('/api/autopilots'),
  get: (id: string) => request<AutopilotDetail>(`/api/autopilots/${id}`),
  create: (input: AutopilotInput) => request<AutopilotDetail>('/api/autopilots', json(input)),
  update: (id: string, input: AutopilotInput) =>
    request<AutopilotDetail>(`/api/autopilots/${id}`, {
      method: 'PUT',
      body: JSON.stringify(input),
    }),
  remove: (id: string) => request<void>(`/api/autopilots/${id}`, { method: 'DELETE' }),
  pause: (id: string) => request<AutopilotDetail>(`/api/autopilots/${id}/pause`, json({})),
  resume: (id: string) => request<AutopilotDetail>(`/api/autopilots/${id}/resume`, json({})),
  run: (id: string) => request<{ queued: boolean }>(`/api/autopilots/${id}/run`, json({})),
  clipAction: (id: string, clipId: string, action: 'approve' | 'reject') =>
    request<AutopilotClip>(`/api/autopilots/${id}/clips/${clipId}`, json({ action })),
}

// ---------------------------------------------------------------- helpers

export const PLATFORM_NAME: Record<Platform, string> = { youtube: 'YouTube', instagram: 'Instagram' }

/** "Today 18:00" / "Tomorrow 12:00" / "Wed 19 Nov 12:00", in this browser's time zone. */
export function when(iso: string | null): string {
  if (!iso) return 'as soon as possible'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  const today = new Date()
  const tomorrow = new Date(Date.now() + 86_400_000)
  if (d.toDateString() === today.toDateString()) return `Today ${time}`
  if (d.toDateString() === tomorrow.toDateString()) return `Tomorrow ${time}`
  return `${d.toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })} ${time}`
}

export const STATUS_LABEL: Record<string, string> = {
  queued: 'Waiting',
  uploading: 'Uploading',
  scheduled: 'Scheduled',
  posted: 'Posted',
  failed: 'Failed',
  interrupted: 'Interrupted',
  checking: 'Checking',
  canceled: 'Cancelled',
  planned: 'Planned',
  skipped: 'Skipped',
}

export function statusTone(status: string): string {
  if (status === 'posted') return 'text-signal-good'
  if (status === 'failed' || status === 'interrupted') return 'text-signal-bad'
  if (status === 'canceled' || status === 'skipped') return 'text-ink-500'
  if (status === 'uploading' || status === 'checking') return 'text-sodium-400'
  return 'text-ink-200'
}

export function errorText(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}
