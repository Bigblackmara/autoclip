/**
 * Typed client for the Live Watch API, and the one shared live event feed.
 *
 * Mirrors api.ts conventions: same request() helper, same ApiError. Wire
 * shapes follow backend/autoclip/api/live.py. Fields that only newer servers
 * send are optional here and feature-detected by the panels, so an older
 * server degrades to fewer details rather than to broken cards.
 */

import { ApiError, request } from '../api'

export type LiveStreamState =
  | 'off'
  | 'waiting_for_live'
  | 'connecting'
  | 'watching'
  | 'clipping'
  | 'reconnecting'
  | 'ended'
  | 'error'

export const STREAM_STATES: readonly LiveStreamState[] = [
  'off',
  'waiting_for_live',
  'connecting',
  'watching',
  'clipping',
  'reconnecting',
  'ended',
  'error',
]

export interface LiveStream {
  id: string
  url: string
  label: string
  /** False while paused (the row then reads state "off"). Drives Pause/Resume. */
  enabled: boolean
  state: LiveStreamState
  state_detail: string
  last_error: string | null
  /** READY clips of this stream (not failed/queued ones). Absent on older servers. */
  clip_count?: number
  created_at: string
  updated_at: string
}

export type LiveClipStatus =
  | 'detecting'
  | 'queued'
  | 'rendering'
  | 'ready'
  | 'failed'
  | 'rejected'

export const CLIP_STATUSES: readonly LiveClipStatus[] = [
  'detecting',
  'queued',
  'rendering',
  'ready',
  'failed',
  'rejected',
]

/** What the clip wall shows by default: everything except rejected windows. */
export const WALL_STATUSES: readonly LiveClipStatus[] = [
  'detecting',
  'queued',
  'rendering',
  'ready',
  'failed',
]

export interface LiveClip {
  id: string
  stream_id: string
  status: LiveClipStatus
  title: string
  score: number
  reason: string
  hook: string
  trigger_start_s: number
  trigger_end_s: number
  clip_start_s: number
  clip_end_s: number
  caption_style: string
  ratio: string
  error: string | null
  created_at: string
  updated_at: string
}

export type LivePlatform = 'youtube' | 'instagram'

export const PLATFORMS: readonly { key: LivePlatform; label: string }[] = [
  { key: 'youtube', label: 'YouTube' },
  { key: 'instagram', label: 'Instagram' },
]

export function platformLabel(platform: LivePlatform): string {
  return platform === 'youtube' ? 'YouTube' : 'Instagram'
}

export type LiveUploadStatus = 'pending' | 'uploading' | 'done' | 'failed' | 'skipped'

export const UPLOAD_STATUSES: readonly LiveUploadStatus[] = [
  'pending',
  'uploading',
  'done',
  'failed',
  'skipped',
]

export interface LiveUpload {
  id: string
  clip_id: string
  platform: LivePlatform
  status: LiveUploadStatus
  upload_url: string
  platform_media_id: string
  error: string | null
  created_at: string
  updated_at: string
}

export interface LiveStatus {
  /** False while Live Watch is switched off in settings. */
  enabled: boolean
  /** False when switched on but this server runs without workers. Newer servers. */
  running?: boolean
  watching: number
  clipping: number
  caps: {
    max_concurrent_streams: number
    clip_queue_len: number
    clip_queue_max: number
    /** Disk held by clip windows still waiting to render, in MB. */
    disk_used_mb: number
  }
}

export interface LiveSettingsData {
  enabled: boolean
  rolling_buffer: {
    segment_s: number
    retain_minutes: number
    max_disk_mb: number
    pre_roll_s: number
    post_roll_s: number
    judge_context_s: number
  }
  rate_limit: {
    judge_per_minute: number
    judge_per_hour: number
    judge_per_day: number
    judge_min_score: number
    cooldown_s: number
  }
  detection: {
    audio: {
      baseline_window_s: number
      spike_z: number
      floor_rms: number
      max_sustained_s: number
    }
    chat: {
      enabled: boolean
      baseline_window_s: number
      rate_z: number
      floor_msgs_per_min: number
      emotes_threshold: number
      hypetrain_msgs_per_min: number
    }
  }
  uploads: {
    youtube: { auto_upload: boolean; title_prefix: string }
    instagram: { auto_upload: boolean; caption_prefix: string }
  }
  workers: {
    max_concurrent_streams: number
    clip_queue_workers: number
    clip_queue_max: number
    /** Advisory in this version: nothing enforces it. */
    cpu_budget_percent: number
    /** Advisory in this version: nothing enforces it. */
    ram_budget_gb: number
    /** Disk budget for everything under live/ except finished clips. Newer servers. */
    disk_budget_mb?: number
    /** New candidates are refused below this much free disk. Newer servers. */
    min_free_disk_mb?: number
  }
}

/** PUT /api/live/settings answer: the stored settings, plus whether they went live. */
export type LiveSettingsSaved = LiveSettingsData & {
  applied?: boolean
  /** Why the running feature did not take the save (``applied`` false); the save stands. */
  apply_error?: string
}

export interface LivePlatformStatus {
  connected: boolean
  account: string | null
  /** The exact OAuth callback URL to register with the platform. Newer servers. */
  redirect_uri?: string
  /** Whether your OAuth app's client id + secret are stored. Newer servers. */
  has_credentials?: boolean
}

export interface LiveAttestation {
  accepted: boolean
  accepted_at: string | null
}

export const liveApi = {
  status: () => request<LiveStatus>('/api/live/status'),

  listStreams: () => request<LiveStream[]>('/api/live/streams'),
  addStream: (url: string, label = '') =>
    request<LiveStream>('/api/live/streams', {
      method: 'POST',
      body: JSON.stringify({ url, label }),
    }),
  removeStream: (id: string) => request<void>(`/api/live/streams/${id}`, { method: 'DELETE' }),
  pauseStream: (id: string) =>
    request<LiveStream>(`/api/live/streams/${id}/pause`, { method: 'POST' }),
  resumeStream: (id: string) =>
    request<LiveStream>(`/api/live/streams/${id}/resume`, { method: 'POST' }),

  /** Newest first. `status` filters server-side; callers still filter, for older servers. */
  listClips: ({ limit = 100, status }: { limit?: number; status?: readonly LiveClipStatus[] } = {}) => {
    const params = new URLSearchParams({ limit: String(limit) })
    if (status && status.length > 0) params.set('status', status.join(','))
    return request<LiveClip[]>(`/api/live/clips?${params.toString()}`)
  },
  getClip: (id: string) => request<LiveClip>(`/api/live/clips/${id}`),
  discardClip: (id: string) =>
    request<LiveClip>(`/api/live/clips/${id}/discard`, { method: 'POST' }),
  /** Re-enqueue a failed or interrupted clip; 409 with a reason when it can't be. */
  retryClip: (id: string) => request<LiveClip>(`/api/live/clips/${id}/retry`, { method: 'POST' }),

  /** `attempt` busts the browser's cached failure after a manual retry. */
  clipMediaUrl: (id: string, attempt = 0) =>
    attempt > 0 ? `/api/live/clips/${id}/media?attempt=${attempt}` : `/api/live/clips/${id}/media`,

  // Beyond the server's default 200 rows: the wall holds up to 100 clips x 2
  // platforms plus retries, and a clip missing its "done" row offers Upload again.
  listUploads: () => request<LiveUpload[]>('/api/live/uploads?limit=1000'),
  uploadClip: (id: string, platform: LivePlatform) =>
    request<LiveUpload>(`/api/live/clips/${id}/upload`, {
      method: 'POST',
      body: JSON.stringify({ platform }),
    }),

  startAuth: (platform: LivePlatform) =>
    request<{ redirect_url: string }>(`/api/live/uploads/${platform}/start`, { method: 'POST' }),
  authStatus: (platform: LivePlatform) =>
    request<LivePlatformStatus>(`/api/live/uploads/${platform}/status`),
  /** Write-only: the server stores both in the OS keyring and never echoes them. */
  saveCredentials: (platform: LivePlatform, clientId: string, clientSecret: string) =>
    request<{ saved: boolean }>(`/api/live/uploads/${platform}/credentials`, {
      method: 'POST',
      body: JSON.stringify({ client_id: clientId, client_secret: clientSecret }),
    }),

  getAttestation: () => request<LiveAttestation>('/api/live/uploads/attestation'),
  acceptAttestation: () =>
    request<LiveAttestation>('/api/live/uploads/attestation', {
      method: 'POST',
      body: JSON.stringify({ accepted: true }),
    }),

  getSettings: () => request<LiveSettingsData>('/api/live/settings'),
  /** Full settings object on purpose: the server validates and replaces the whole section. */
  putSettings: (settings: LiveSettingsData) =>
    request<LiveSettingsSaved>('/api/live/settings', {
      method: 'PUT',
      body: JSON.stringify(settings),
    }),
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

export function asError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error))
}

/** One line for an action error: the server's message, and its fix when it sent one. */
export function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.hint ? `${error.message} — ${error.hint}` : error.message
  if (error instanceof Error) return error.message
  return String(error)
}

export function errorStatus(error: unknown): number | null {
  return error instanceof ApiError ? error.status : null
}

/** "NotLinkedError: youtube account…" → "youtube account…" */
function stripExceptionName(text: string): string {
  return text.replace(/^[A-Z][A-Za-z]*(Error|Exception|Accepted): /, '')
}

function capitalize(text: string): string {
  return text ? text[0].toUpperCase() + text.slice(1) : text
}

export interface FriendlyError {
  message: string
  /** True when the fix lives in Settings → Uploads (rights, app, connection). */
  setup: boolean
}

/**
 * Turn an upload or OAuth failure — an HTTP `detail` or a stored record's
 * error — into one sentence that says what to do next. Unknown text passes
 * through, trimmed, rather than being hidden behind a vague "failed".
 */
export function describeUploadError(
  raw: string | null | undefined,
  platform: LivePlatform,
): FriendlyError {
  const name = platformLabel(platform)
  const text = stripExceptionName((raw ?? '').trim())
  const lower = text.toLowerCase()
  if (!text) return { message: `The ${name} upload failed without a reason.`, setup: false }
  if (lower === 'interrupted' || lower.startsWith('cancelled')) {
    return { message: `The ${name} upload was interrupted when AutoClip stopped.`, setup: false }
  }
  // Order matters: the rights refusal also mentions the keyring it lives in.
  if (lower.includes('rights') || lower.includes('attestation')) {
    return {
      message: 'Uploads are locked until you accept the rights notice in Settings.',
      setup: true,
    }
  }
  if (lower.includes('keyring')) {
    return {
      message:
        'This machine has no usable OS keyring, so OAuth credentials can’t be stored and uploads stay off.',
      setup: true,
    }
  }
  if (lower.includes('client id') || lower.includes('client_id') || lower.includes('client credentials')) {
    return { message: `Save your ${name} OAuth app’s client ID and secret in Settings first.`, setup: true }
  }
  // Before "connect the": "reconnect the account" contains it.
  if (lower.includes('reconnect') || lower.includes('expired') || lower.includes('invalid_grant')) {
    return { message: `The ${name} sign-in expired — reconnect the account in Settings.`, setup: true }
  }
  if (lower.includes('not connected') || lower.includes('connect the') || lower.includes('not linked')) {
    return { message: `Connect your ${name} account in Settings first.`, setup: true }
  }
  const http = /HTTP (\d{3})/.exec(text)
  if (http) {
    const code = Number(http[1])
    const detail = /"message"\s*:\s*"([^"]{1,160})"/.exec(text)?.[1]
    if (/quota|ratelimit|rate limit|too many/i.test(text)) {
      return { message: `${name}’s upload quota is used up for now — try again later.`, setup: false }
    }
    if (code === 401 || code === 403) {
      return {
        message: `${name} refused the upload (HTTP ${code})${detail ? `: ${detail}` : ''} — reconnect the account or check your app’s permissions.`,
        setup: true,
      }
    }
    if (code === 429) return { message: `${name} is rate-limiting uploads — try again later.`, setup: false }
    if (code >= 500) return { message: `${name} had a server error (HTTP ${code}) — try again.`, setup: false }
    return { message: `${name} rejected the upload (HTTP ${code})${detail ? `: ${detail}` : '.'}`, setup: false }
  }
  if (/(connecterror|connecttimeout|readtimeout|timeoutexception|networkerror|failed: \w+error)/i.test(text)) {
    return { message: `Couldn’t reach ${name} — check the connection and try again.`, setup: false }
  }
  const trimmed = text.length > 220 ? `${text.slice(0, 217)}…` : text
  return { message: capitalize(trimmed), setup: false }
}

// --------------------------------------------------------------------------
// The shared live event feed
// --------------------------------------------------------------------------

/** One SSE frame from /api/live/events, normalized. */
export interface LiveEvent {
  /**
   * snapshot | stream_state | clip | clip_status | upload | heartbeat.
   * Older servers name stream changes "state"; those arrive as stream_state.
   */
  type: string
  data: Record<string, unknown>
}

/** open: frames flowing · connecting/retrying: on the way · paused: tab hidden or no subscribers. */
export type LiveFeedState = 'connecting' | 'open' | 'retrying' | 'paused'

const FEED_URL = '/api/live/events'
const NAMED_EVENTS = ['snapshot', 'state', 'stream_state', 'clip', 'clip_status', 'upload', 'heartbeat']
const RETRY_MIN_MS = 3_000
const RETRY_MAX_MS = 60_000

const eventListeners = new Set<(event: LiveEvent) => void>()
const feedListeners = new Set<(state: LiveFeedState) => void>()
let source: EventSource | null = null
let feedState: LiveFeedState = 'paused'
let retryTimer: number | undefined
let retryDelay = RETRY_MIN_MS

function setFeedState(next: LiveFeedState) {
  if (next === feedState) return
  feedState = next
  for (const listener of [...feedListeners]) listener(next)
}

function dispatch(event: LiveEvent) {
  for (const listener of [...eventListeners]) {
    try {
      listener(event)
    } catch (error) {
      console.error('Live event handler failed', error)
    }
  }
}

function toEvent(name: string, raw: unknown): LiveEvent {
  let parsed: unknown = {}
  if (typeof raw === 'string' && raw) {
    try {
      parsed = JSON.parse(raw)
    } catch {
      parsed = {}
    }
  }
  const data =
    typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {}
  let type = name
  if (name === 'message' && typeof data.type === 'string') type = data.type
  if (type === 'state') type = 'stream_state'
  return { type, data }
}

function openFeed() {
  if (source || typeof EventSource === 'undefined') return
  window.clearTimeout(retryTimer)
  retryTimer = undefined
  const es = new EventSource(FEED_URL)
  source = es
  if (feedState !== 'retrying') setFeedState('connecting')
  es.onopen = () => {
    retryDelay = RETRY_MIN_MS
    setFeedState('open')
  }
  es.onerror = () => {
    if (source !== es) return
    setFeedState('retrying')
    if (es.readyState !== EventSource.CLOSED) return // the browser reconnects on its own
    // The browser gave up (an HTTP error, say): retry ourselves, backing off.
    es.close()
    source = null
    retryTimer = window.setTimeout(() => {
      retryTimer = undefined
      if (eventListeners.size > 0 && !document.hidden) openFeed()
    }, retryDelay)
    retryDelay = Math.min(retryDelay * 2, RETRY_MAX_MS)
  }
  for (const name of [...NAMED_EVENTS, 'message']) {
    es.addEventListener(name, (event: MessageEvent) => dispatch(toEvent(name, event.data)))
  }
}

function closeFeed() {
  window.clearTimeout(retryTimer)
  retryTimer = undefined
  if (source) {
    source.close()
    source = null
  }
  setFeedState('paused')
}

/**
 * A hidden tab gives its connection back. Browsers allow six HTTP/1.1
 * connections per host, and a long-lived stream held by every open tab is
 * how a few forgotten tabs end up stalling every request the app makes.
 */
function onVisibilityChange() {
  if (document.hidden) {
    closeFeed()
  } else if (eventListeners.size > 0) {
    retryDelay = RETRY_MIN_MS
    openFeed()
  }
}

/**
 * Subscribe to the live feed. Every panel shares ONE EventSource per tab:
 * opened with the first subscriber, closed with the last, closed while the
 * tab is hidden. Each (re)connect starts with a `snapshot` frame, which is
 * the cue to refetch whatever was missed. Returns the unsubscribe.
 */
export function subscribeLiveEvents(listener: (event: LiveEvent) => void): () => void {
  eventListeners.add(listener)
  if (eventListeners.size === 1) {
    document.addEventListener('visibilitychange', onVisibilityChange)
    if (!document.hidden) openFeed()
  }
  return () => {
    if (!eventListeners.delete(listener)) return
    if (eventListeners.size === 0) {
      document.removeEventListener('visibilitychange', onVisibilityChange)
      closeFeed()
    }
  }
}

/** Follow the shared feed's connection state; the listener gets the current one at once. */
export function subscribeLiveFeedState(listener: (state: LiveFeedState) => void): () => void {
  feedListeners.add(listener)
  listener(feedState)
  return () => {
    feedListeners.delete(listener)
  }
}
