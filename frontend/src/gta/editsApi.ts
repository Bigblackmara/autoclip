/**
 * GTA Mode clip edits (backend: autoclip/gta/edits.py via api/gta.py):
 * hook options, flash-forward, variants, caption language and picks on the
 * stream's Hype Curve.
 */

import { request } from '../api'
import type { AudienceClipBand, Hotspot, TimeRange } from '../picks/api'
import type { GtaClip, GtaSettings } from './api'

/** Fields the clip rows carry since schema v8. */
export interface GtaClipEdits {
  hook_options?: string[]
  /** null = follows the GTA setting. */
  flash_forward?: boolean | null
  teaser_start_s?: number | null
  teaser_end_s?: number | null
  variant_of?: string | null
  family_id?: string
  motion_seed?: number
  caption_style?: string
  localised?: { mode?: string; hook?: string; caption?: string; applied?: string[] }
  source?: 'judge' | 'pick' | 'variant'
}

export type EditableClip = GtaClip & GtaClipEdits

export type CaptionLanguage = 'as_spoken' | 'roman' | 'english'

export interface GtaEditSettings {
  flash_forward: boolean
  flash_forward_s: number
  caption_language: CaptionLanguage
}

export const CAPTION_LANGUAGES: { value: CaptionLanguage; label: string; hint: string }[] = [
  {
    value: 'as_spoken',
    label: 'As spoken',
    hint: 'Every language in its own script; Hindi as Hinglish (Latin letters), no AI call.',
  },
  { value: 'roman', label: 'Latin letters', hint: 'Same words, written in Latin letters.' },
  { value: 'english', label: 'English', hint: 'Captions, hook and post text translated by your AI provider.' },
]

export interface HypeGraph {
  duration_s: number
  step_s: number
  curve: number[]
  available: string[]
  hotspots: Hotspot[]
  clips: AudienceClipBand[]
  range_start_s: number
  range_end_s: number
  max_pick_s: number
  min_pick_s: number
  /** Why picks are not possible yet ('' = they are). */
  problem: string
}

export const gtaEditsApi = {
  editSettings: () => request<GtaSettings & GtaEditSettings>('/api/gta/settings'),
  saveEditSettings: (patch: Partial<GtaEditSettings>) =>
    request<GtaSettings & GtaEditSettings>('/api/gta/settings', {
      method: 'PUT',
      body: JSON.stringify(patch),
    }),
  /** Pick a hook and/or set the flash-forward (null = follow the setting); re-renders. */
  editClip: (clipId: string, patch: { hook_title?: string; flash_forward?: boolean | null }) =>
    request<EditableClip>(`/api/gta/clips/${clipId}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  makeVariant: (clipId: string) =>
    request<EditableClip>(`/api/gta/clips/${clipId}/variant`, { method: 'POST' }),
  hype: (streamId: string) => request<HypeGraph>(`/api/gta/streams/${streamId}/hype`),
  makePicks: (streamId: string, ranges: TimeRange[]) =>
    request<{ clips: EditableClip[] }>(`/api/gta/streams/${streamId}/picks`, {
      method: 'POST',
      body: JSON.stringify({ ranges: ranges.map((r) => ({ start_s: r.start, end_s: r.end })) }),
    }),
}

export function errorMessage(err: unknown): string {
  if (err instanceof Error) {
    const hint = (err as Error & { hint?: string }).hint
    return hint ? `${err.message}\n${hint}` : err.message
  }
  return String(err)
}
