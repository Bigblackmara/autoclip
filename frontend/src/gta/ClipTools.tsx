import { useState } from 'react'

import { Badge, Button, CheckIcon, SparklesIcon, cx } from '../ui'
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

  const chips = [
    clip.variant_of ? 'Variant' : null,
    clip.source === 'pick' ? 'Picked by you' : null,
    clip.caption_style ? `${LOOK_LABEL[clip.caption_style] ?? clip.caption_style} captions` : null,
    clip.localised?.applied?.length ? `Captions: ${clip.localised.applied.join(', ')}` : null,
  ].filter(Boolean) as string[]

  return (
    <div className="grid gap-3 rounded-xl bg-subtle p-3 text-xs text-fg-muted">
      {chips.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          {chips.map((chip) => (
            <Badge key={chip} tone={chip === 'Variant' || chip === 'Picked by you' ? 'info' : 'neutral'}>
              {chip}
            </Badge>
          ))}
        </div>
      )}

      {options.length > 1 && (
        <div>
          <p className="mb-1.5 font-medium text-fg">Hook title</p>
          <div role="group" aria-label="Hook title options" className="grid gap-1.5">
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
                  className={cx(
                    'flex items-start gap-2 rounded-lg border px-2.5 py-1.5 text-left text-xs leading-snug transition-colors',
                    chosen
                      ? 'border-accent bg-surface font-semibold text-accent-text ring-1 ring-accent'
                      : 'border-line bg-surface text-fg hover:border-line-strong disabled:opacity-50',
                  )}
                >
                  {chosen && <CheckIcon size={13} className="mt-0.5" />}
                  {busy === `hook:${option}` ? 'Rendering…' : option}
                </button>
              )
            })}
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-end gap-2">
        <label className="grid gap-1">
          <span className="font-medium text-fg">Flash-forward opening</span>
          <select
            className="field w-40 py-1 text-xs"
            value={flash == null ? 'default' : flash ? 'on' : 'off'}
            disabled={!editable || busy !== null}
            onChange={(e) => {
              const value = e.target.value
              const next = value === 'default' ? null : value === 'on'
              void act('flash', () => gtaEditsApi.editClip(clip.id, { flash_forward: next }), 'Re-rendering…')
            }}
          >
            <option value="default">As in settings</option>
            <option value="on">On</option>
            <option value="off">Off</option>
          </select>
        </label>
        {clip.status === 'ready' && (
          <Button
            size="sm"
            icon={SparklesIcon}
            loading={busy === 'variant'}
            disabled={busy !== null}
            title="Same moment with the next caption look, hook and motion — posted on another day"
            onClick={() =>
              act(
                'variant',
                () => gtaEditsApi.makeVariant(clip.id),
                'Variant queued: next look, next hook, new motion. The poster never puts it on the same day as this clip on one platform.',
              )
            }
          >
            {busy === 'variant' ? 'Queuing…' : 'Make a variant'}
          </Button>
        )}
      </div>
      {teaseS > 0 && <p>Opens on the peak ({teaseS.toFixed(1)} s), then plays from the start.</p>}
      {note && <p className="text-success">{note}</p>}
      {error && <p className="whitespace-pre-line text-danger">{error}</p>}
    </div>
  )
}
