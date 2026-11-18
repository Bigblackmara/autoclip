import { useCallback, useEffect, useState } from 'react'

import { formatDuration } from '../api'
import { RewatchGraph } from '../components/RewatchGraph'
import type { TimeRange } from '../picks/api'
import { Button, Callout, IconButton, ScissorsIcon, Skeleton, XIcon } from '../ui'
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
      <Button size="sm" icon={ScissorsIcon} onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        {open ? 'Hide Hype Curve' : 'Pick moments on the Hype Curve'}
      </Button>
      {open && (
        <div className="mt-3 grid gap-3 rounded-xl border border-line bg-subtle/60 p-4">
          {!graph ? (
            error ? (
              <Callout tone="danger">
                <span className="whitespace-pre-line">{error}</span>
              </Callout>
            ) : (
              <Skeleton className="h-40 w-full" />
            )
          ) : graph.curve.length === 0 ? (
            <p className="text-sm text-fg-muted">{graph.problem || 'This stream has no Hype Curve yet.'}</p>
          ) : (
            <>
              <p className="text-xs leading-relaxed text-fg-muted">
                Drag across the curve to pick a moment ({graph.min_pick_s}–{Math.round(graph.max_pick_s)} s each); drag
                an edge to adjust. Dots are hype markers, grey bars the clips already made. Picked moments skip the
                judge&rsquo;s reject round but still get titles, captions and QA.
                {graph.available.length > 0 && ` Signals: ${graph.available.join(', ')}.`}
              </p>
              <div className="rounded-lg bg-surface p-3 pt-6">
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
                <ol className="grid gap-1 rounded-lg bg-surface p-2 text-xs">
                  {picks.map((p, i) => (
                    <li key={`${p.start}-${i}`} className="flex items-center gap-3 px-1">
                      <span className="numeric grid size-5 place-items-center rounded-full bg-accent-soft text-[10px] font-semibold text-accent-text">
                        {i + 1}
                      </span>
                      <span className="numeric font-medium text-fg">
                        {formatDuration(p.start)}–{formatDuration(p.end)}
                      </span>
                      <span className="numeric text-fg-muted">{Math.round(p.end - p.start)} s</span>
                      <IconButton
                        icon={XIcon}
                        size="sm"
                        className="ml-auto"
                        label={`Remove pick ${i + 1}`}
                        onClick={() => setPicks(picks.filter((_, k) => k !== i))}
                      />
                    </li>
                  ))}
                </ol>
              )}
              {(tooLong.length > 0 || tooShort.length > 0) && (
                <p className="text-xs text-danger">
                  {tooLong.length > 0 && `${tooLong.length} pick(s) too long for a Short. `}
                  {tooShort.length > 0 && `${tooShort.length} pick(s) shorter than ${graph.min_pick_s} s.`}
                </p>
              )}
              {graph.problem && <p className="text-xs text-danger">{graph.problem}</p>}
              <div>
                <Button
                  variant="primary"
                  icon={ScissorsIcon}
                  loading={busy}
                  disabled={!picks.length || tooLong.length > 0 || tooShort.length > 0 || !!graph.problem}
                  onClick={() => void submit()}
                >
                  {busy ? 'Writing titles…' : `Make ${picks.length || ''} GTA clip${picks.length === 1 ? '' : 's'}`}
                </Button>
              </div>
            </>
          )}
          {note && <Callout tone="success">{note}</Callout>}
          {error && graph && (
            <Callout tone="danger">
              <span className="whitespace-pre-line">{error}</span>
            </Callout>
          )}
        </div>
      )}
    </div>
  )
}
