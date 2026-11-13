/**
 * Typed client for the Live Watch API (mirrors api.ts conventions).
 *
 * The endpoint contract is frozen in backend/autoclip/api/live.py's docstring;
 * A10 implements the server side, F1–F3 consume this file. Keep the shapes
 * here in sync with api/live.py only through that docstring.
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

export interface LiveStream {
  id: string
  url: string
  label: string
  enabled: boolean
  state: LiveStreamState
  state_detail: string
  last_error: string | null
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

export interface LiveUpload {
  id: string
  clip_id: string
  platform: 'youtube' | 'instagram'
  status: 'pending' | 'uploading' | 'done' | 'failed' | 'skipped'
  upload_url: string
  platform_media_id: string
  error: string | null
  created_at: string
  updated_at: string
}

export interface LiveStatus {
  enabled: boolean
  watching: number
  clipping: number
  caps: {
    max_concurrent_streams: number
    clip_queue_len: number
    clip_queue_max: number
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
    cpu_budget_percent: number
    ram_budget_gb: number
  }
}

/** SSE event on /api/live/events and /api/live/streams/:id/events. */
export interface LiveEvent {
  type: string
  stream_id?: string
  clip_id?: string
  data?: Record<string, unknown>
}

export class LiveApiError extends ApiError {}

export const liveApi = {
  status: () => request<LiveStatus>('/api/live/status'),

  listStreams: () => request<LiveStream[]>('/api/live/streams'),
  addStream: (url: string, label = '') =>
    request<LiveStream>('/api/live/streams', {
      method: 'POST',
      body: JSON.stringify({ url, label }),
    }),
  removeStream: (id: string) => request<void>(`/api/live/streams/${id}`, { method: 'DELETE' }),
  pauseStream: (id: string) => request<LiveStream>(`/api/live/streams/${id}/pause`, { method: 'POST' }),
  resumeStream: (id: string) => request<LiveStream>(`/api/live/streams/${id}/resume`, { method: 'POST' }),

  listClips: (limit = 100) => request<LiveClip[]>(`/api/live/clips?limit=${limit}`),
  getClip: (id: string) => request<LiveClip>(`/api/live/clips/${id}`),
  discardClip: (id: string) => request<LiveClip>(`/api/live/clips/${id}/discard`, { method: 'POST' }),

  clipMediaUrl: (id: string) => `/api/live/clips/${id}/media`,

  listUploads: () => request<LiveUpload[]>('/api/live/uploads'),
  uploadClip: (id: string, platform: 'youtube' | 'instagram') =>
    request<LiveUpload>(`/api/live/clips/${id}/upload`, {
      method: 'POST',
      body: JSON.stringify({ platform }),
    }),

  startYouTubeAuth: () =>
    request<{ redirect_url: string }>('/api/live/uploads/youtube/start', { method: 'POST' }),
  startInstagramAuth: () =>
    request<{ redirect_url: string }>('/api/live/uploads/instagram/start', { method: 'POST' }),
  youTubeAuthStatus: () =>
    request<{ connected: boolean; account: string | null }>('/api/live/uploads/youtube/status'),
  instagramAuthStatus: () =>
    request<{ connected: boolean; account: string | null }>('/api/live/uploads/instagram/status'),

  getSettings: () => request<LiveSettingsData>('/api/live/settings'),
  putSettings: (patch: Partial<LiveSettingsData>) =>
    request<LiveSettingsData>('/api/live/settings', { method: 'PUT', body: JSON.stringify(patch) }),
}

/** Subscribe to the global live SSE feed (or one stream's). Cleanup on call. */
export function subscribeLiveEvents(
  onEvent: (event: LiveEvent) => void,
  streamId?: string,
): () => void {
  const url = streamId ? `/api/live/streams/${streamId}/events` : '/api/live/events'
  const source = new EventSource(url)
  const forward = (type: string) => (e: MessageEvent) => {
    let data: Record<string, unknown> | undefined
    try {
      data = JSON.parse(e.data)
    } catch {
      data = undefined
    }
    onEvent({ type, stream_id: streamId, data })
  }
  const types = [
    'snapshot',
    'state',
    'clip',
    'clip_status',
    'upload',
    'heartbeat',
  ]
  for (const t of types) source.addEventListener(t, forward(t))
  return () => source.close()
}
