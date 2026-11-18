import { useCallback, useEffect, useState } from 'react'

import { formatDuration } from '../api'
import { RewatchGraph } from '../components/RewatchGraph'
import type { TimeRange } from '../picks/api'
import type { GtaStream } from './api'
import { errorMessage, gtaEditsApi, type HypeGraph } from './editsApi'

const NO_COMPONENTS = { heatmap: false, comments: false, energy: false }

/**
 * "Pick on the Hype Curve" for one GTA stream: drag ranges on the stream's
 * Hype Curve and turn them into GTA clips. They skip the judge's reject
 * phase but still get hook titles, caption and hashtags from the judge
 * prompt (a plain title without an AI provider), then go through the normal
 * GTA render + QA.
 */
export function GtaPickPanel({ stream }: { stream: GtaStream }) {
  const [open, setOpen] = useState(false)
  const [graph, setGraph] = useState<HypeGraph | null>(null)
  const [picks, setPicks] = useState<TimeRange[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')

  const pickable =
    !stream.is_live && ['done', 'analyzed', 'failed', 'cancelled'].includes(stream.status)

  const load = useCallback(() => {
    gtaEditsApi
      .hype(stream.id)
      .then(setGraph)
      .catch((err) => setError(errorMessage(err)))
  }, [stream.id])

  useEffect(() => {
    if (open) load()
  }, [open, load])

  if (!pickable) return null

  const tooLong = graph ? picks.filter((p) => p.end - p.start > graph.max_pick_s) : []
  const tooShort = graph ? picks.filter((p) => p.end - p.start < graph.min_pick_s) : []

  const submit = async () => {
    setBusy(true)
    setError('')
    setNote('')
    try {
      const result = await gtaEditsApi.makePicks(stream.id, picks)
      setNote(
        `${result.clips.length} clip(s) queued: ${result.clips.map((c) => c.hook_title).join(' · ')}. They render below like any other clip.`,
      )
      setPicks([])
      load()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mt-3">
      <button type="button" className="btn btn-ghost" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        {open ? 'Hide Hype Curve' : 'Pick on Hype Curve'}
      </button>
      {open && (
        <div className="mt-4 grid gap-3 border-l border-ink-800 pl-4">
          {!graph ? (
            <p className="text-xs text-ink-500">{error || 'Loading the Hype Curve…'}</p>
          ) : graph.curve.length === 0 ? (
            <p className="text-xs text-ink-500">{graph.problem || 'This stream has no Hype Curve yet.'}</p>
          ) : (
            <>
              <p className="text-xs text-ink-400">
                Drag across the curve to pick a moment ({graph.min_pick_s}–{Math.round(graph.max_pick_s)} s each);
                drag an edge to adjust. Dots are hype markers, grey bars the clips already made.
                {graph.available.length > 0 && ` Signals: ${graph.available.join(', ')}.`}
              </p>
              <div className="pt-5">
                <RewatchGraph
                  curve={graph.curve}
                  components={{}}
                  visible={NO_COMPONENTS}
                  stepS={graph.step_s}
                  duration={graph.duration_s}
                  bands={graph.clips}
                  hotspots={graph.hotspots}
                  picks={picks}
                  onPicksChange={setPicks}
                  height={160}
                />
              </div>
              {picks.length > 0 && (
                <ol className="grid gap-1 text-xs">
                  {picks.map((p, i) => (
                    <li key={`${p.start}-${i}`} className="flex items-center gap-3">
                      <span className="numeric w-5 font-semibold text-sodium-500">{i + 1}</span>
                      <span className="numeric text-ink-200">
                        {formatDuration(p.start)}–{formatDuration(p.end)}
                      </span>
                      <span className="numeric text-ink-500">{Math.round(p.end - p.start)} s</span>
                      <button
                        type="button"
                        className="btn btn-quiet !px-1.5 !py-0 text-xs"
                        onClick={() => setPicks(picks.filter((_, k) => k !== i))}
                        aria-label={`Remove pick ${i + 1}`}
                      >
                        ✕
                      </button>
                    </li>
                  ))}
                </ol>
              )}
              {(tooLong.length > 0 || tooShort.length > 0) && (
                <p className="text-xs text-signal-bad">
                  {tooLong.length > 0 && `${tooLong.length} pick(s) too long for a Short. `}
                  {tooShort.length > 0 && `${tooShort.length} pick(s) shorter than ${graph.min_pick_s} s.`}
                </p>
              )}
              {graph.problem && <p className="text-xs text-signal-bad">{graph.problem}</p>}
              <div>
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={busy || !picks.length || tooLong.length > 0 || tooShort.length > 0 || !!graph.problem}
                  onClick={() => void submit()}
                >
                  {busy ? 'Writing titles…' : `Make ${picks.length || ''} GTA clip${picks.length === 1 ? '' : 's'}`}
                </button>
              </div>
            </>
          )}
          {note && <p className="text-xs text-signal-good">{note}</p>}
          {error && graph && <p className="whitespace-pre-line text-xs text-signal-bad">{error}</p>}
        </div>
      )}
    </div>
  )
}
