/**
 * Client for "pick your own parts" (autoclip.api.picks).
 */

import { request, type Framing } from '../api'

export interface TimeRange {
  start: number
  end: number
}

export interface AudienceClipBand {
  id: string
  title: string
  rank: number
  start_s: number
  end_s: number
  status: string
  /** Parts of a joined clip; empty for an ordinary one. */
  spans: [number, number][]
}

export interface Hotspot {
  start_s: number
  end_s: number
  peak_s: number
  value: number
  reasons: string[]
}

export interface Chapter {
  start_s: number
  end_s: number
  title: string
}

export interface Audience {
  duration_s: number
  /** "audience" (heatmap/comments), "energy" (loudness only) or "none". */
  kind: 'audience' | 'energy' | 'none'
  sources: string[]
  /** Seconds per point (curves longer than ~2000 s are max-pooled). */
  step_s: number
  curve: number[]
  components: Partial<Record<'heatmap' | 'comments' | 'energy', number[]>>
  hotspots: Hotspot[]
  chapters: Chapter[]
  clips: AudienceClipBand[]
  has_transcript: boolean
}

export interface PickedClip {
  id: string
  title: string
  start_s: number
  end_s: number
  duration_s: number
  spans: [number, number][]
  score: number
}

export interface PickTask {
  job_id: string
  clip_ids: string[]
  state: 'running' | 'done' | 'failed'
  stage: string
  progress: number
  message: string
  error: string | null
}

export const picksApi = {
  audience: (jobId: string) => request<Audience>(`/api/jobs/${jobId}/audience`),
  create: (jobId: string, ranges: TimeRange[], join: boolean, framing: Framing | null) =>
    request<{ clips: PickedClip[]; task: PickTask }>(`/api/jobs/${jobId}/picked-clips`, {
      method: 'POST',
      body: JSON.stringify({
        ranges: ranges.map((r) => ({ start_s: r.start, end_s: r.end })),
        join,
        framing,
      }),
    }),
  status: (jobId: string) => request<PickTask>(`/api/jobs/${jobId}/picked-clips`),
}

/** Most a single pick (and a joined clip in total) may run, as on the server. */
export const MAX_PICK_S = 180
export const MIN_PICK_S = 1
export const MAX_PICKS = 12
