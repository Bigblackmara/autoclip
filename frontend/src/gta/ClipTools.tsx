import { useState } from 'react'

import type { GtaClip } from './api'
import { errorMessage, gtaEditsApi, type EditableClip } from './editsApi'

const LOOK_LABEL: Record<string, string> = {
  gta: 'GTA',
  pop: 'Pop',
  neon: 'Neon',
  comic: 'Comic',
  minimal: 'Minimal',
  karaoke_fill: 'Karaoke',
}

/**
 * Edits on a finished GTA clip: pick one of the judge's three hook titles,
 * flip the flash-forward, or make a variant (next caption look, next hook,
 * new motion) that the poster schedules on another day or platform.
 *
 * Every change re-renders through the normal render + QA path; the page's
 * own polling shows the clip go "rendering" and come back.
 */
export function ClipTools({ clip: base }: { clip: GtaClip }) {
  const clip = base as EditableClip
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const options = clip.hook_options?.length ? clip.hook_options : [clip.hook_title]
  const editable = clip.status === 'ready' || clip.status === 'qa_failed' || clip.status === 'failed'
  const flash = clip.flash_forward
  const teaseS = Number((clip.qa as Record<string, unknown> | undefined)?.flash_forward_s ?? 0)

  const act = async (key: string, work: () => Promise<unknown>, done: string) => {
    setBusy(key)
    setError('')
    setNote('')
    try {
      await work()
      setNote(done)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="grid gap-2 text-xs text-ink-400">
      <div className="flex flex-wrap items-center gap-1.5">
        {clip.variant_of && (
          <span className="border border-ink-600 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-ink-300">
            Variant
          </span>
        )}
        {clip.source === 'pick' && (
          <span className="border border-ink-600 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-ink-300">
            Picked
          </span>
        )}
        {clip.caption_style && (
          <span className="text-[10px] uppercase tracking-wide text-ink-500">
            {LOOK_LABEL[clip.caption_style] ?? clip.caption_style} look
          </span>
        )}
        {clip.localised?.applied?.length ? (
          <span className="text-[10px] uppercase tracking-wide text-ink-500">
            captions {clip.localised.applied.join(', ')}
          </span>
        ) : null}
      </div>

      {options.length > 1 && (
        <div role="group" aria-label="Hook title options" className="flex flex-wrap gap-1.5">
          {options.map((option) => {
            const chosen = option === clip.hook_title
            return (
              <button
                key={option}
                type="button"
                aria-pressed={chosen}
                disabled={!editable || chosen || busy !== null}
                onClick={() =>
                  act(`hook:${option}`, () => gtaEditsApi.editClip(clip.id, { hook_title: option }), 'Re-rendering with that hook…')
                }
                className={`border px-2 py-1 text-left text-[11px] leading-tight transition-colors ${
                  chosen
                    ? 'border-sodium-500 text-sodium-400'
                    : 'border-ink-700 text-ink-300 hover:border-ink-500 hover:text-ink-100 disabled:opacity-50'
                }`}
              >
                {busy === `hook:${option}` ? 'Rendering…' : option}
              </button>
            )
          })}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1.5">
          Flash-forward
          <select
            className="field !w-auto !py-0.5 text-xs"
            value={flash == null ? 'default' : flash ? 'on' : 'off'}
            disabled={!editable || busy !== null}
            onChange={(e) => {
              const value = e.target.value
              const next = value === 'default' ? null : value === 'on'
              void act(
                'flash',
                () => gtaEditsApi.editClip(clip.id, { flash_forward: next }),
                'Re-rendering…',
              )
            }}
          >
            <option value="default">As in settings</option>
            <option value="on">On</option>
            <option value="off">Off</option>
          </select>
        </label>
        {teaseS > 0 && <span className="text-ink-500">opens on the peak ({teaseS.toFixed(1)} s)</span>}
        {clip.status === 'ready' && (
          <button
            type="button"
            className="btn btn-ghost !px-2 !py-1 text-xs"
            disabled={busy !== null}
            onClick={() =>
              act(
                'variant',
                () => gtaEditsApi.makeVariant(clip.id),
                'Variant queued: next look, next hook, new motion. The poster never puts it on the same day as this clip on one platform.',
              )
            }
          >
            {busy === 'variant' ? 'Queuing…' : 'Make variant'}
          </button>
        )}
      </div>
      {note && <p className="text-signal-good">{note}</p>}
      {error && <p className="whitespace-pre-line text-signal-bad">{error}</p>}
    </div>
  )
}
