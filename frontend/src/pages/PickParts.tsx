import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, formatDuration, FRAMINGS, type Framing, type Job } from '../api'
import { ErrorNote } from '../components/ErrorNote'
import { COMPONENT_STYLE, RewatchGraph } from '../components/RewatchGraph'
import { MAX_PICK_S, MAX_PICKS, MIN_PICK_S, picksApi, type Audience, type PickTask, type TimeRange } from '../picks/api'

type Component = keyof typeof COMPONENT_STYLE

/**
 * Pick your own parts on the "where viewers rewatch" graph.
 *
 * Drag across the graph to pick parts, adjust their edges, then make one clip
 * per part or join them into one clip. The clips go through the normal
 * pipeline (edges snap to sentences, reframe, captions, export) in the
 * background, and land on the review page with the detected ones.
 */
export function PickParts() {
  const { jobId = '' } = useParams()
  const [job, setJob] = useState<Job | null>(null)
  const [audience, setAudience] = useState<Audience | null>(null)
  const [loadError, setLoadError] = useState<Error | null>(null)
  const [picks, setPicks] = useState<TimeRange[]>([])
  const [visible, setVisible] = useState<Record<Component, boolean>>({
    heatmap: true,
    comments: true,
    energy: false,
  })
  const [framing, setFraming] = useState<Framing | ''>('')
  const [task, setTask] = useState<PickTask | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [actionError, setActionError] = useState<Error | null>(null)
  const [playhead, setPlayhead] = useState<number | null>(null)
  const video = useRef<HTMLVideoElement>(null)

  const reloadAudience = useCallback(() => {
    picksApi.audience(jobId).then(setAudience).catch(setLoadError)
  }, [jobId])

  useEffect(() => {
    api.getJob(jobId).then(setJob).catch(setLoadError)
    reloadAudience()
    // A render started earlier (another tab, a reload) is picked up again.
    picksApi
      .status(jobId)
      .then(setTask)
      .catch(() => setTask(null))
  }, [jobId, reloadAudience])

  // Poll while clips render.
  useEffect(() => {
    if (task?.state !== 'running') return
    const timer = setInterval(() => {
      picksApi
        .status(jobId)
        .then((next) => {
          setTask(next)
          if (next.state !== 'running') reloadAudience()
        })
        .catch(() => undefined)
    }, 1500)
    return () => clearInterval(timer)
  }, [task?.state, jobId, reloadAudience])

  if (loadError) {
    return (
      <div className="max-w-3xl pt-14">
        <ErrorNote error={loadError} />
      </div>
    )
  }
  if (!job || !audience) {
    return <p className="pt-24 text-sm text-ink-500">Loading…</p>
  }

  const duration = audience.duration_s || job.source?.duration_s || 0
  const total = picks.reduce((sum, p) => sum + (p.end - p.start), 0)
  const rendering = task?.state === 'running'
  const canPick = job.status === 'done' && !rendering && !submitting
  const tooMany = picks.length > MAX_PICKS

  const submit = async (join: boolean) => {
    setSubmitting(true)
    setActionError(null)
    try {
      const result = await picksApi.create(jobId, picks, join, framing || null)
      setTask(result.task)
      setPicks([])
      reloadAudience()
    } catch (err) {
      setActionError(err as Error)
    } finally {
      setSubmitting(false)
    }
  }

  const seek = (t: number) => {
    setPlayhead(t)
    if (video.current) video.current.currentTime = t
  }

  const updatePick = (index: number, edge: 'start' | 'end', value: number) => {
    if (!Number.isFinite(value)) return
    const next = picks.map((p) => ({ ...p }))
    const p = next[index]
    if (edge === 'start') p.start = Math.max(0, Math.min(value, p.end - MIN_PICK_S))
    else p.end = Math.min(duration, Math.max(value, p.start + MIN_PICK_S))
    setPicks(next.sort((a, b) => a.start - b.start))
  }

  const available = (Object.keys(COMPONENT_STYLE) as Component[]).filter(
    (name) => audience.components[name]?.length,
  )

  return (
    <div className="pt-14">
      <div className="rise flex flex-wrap items-baseline justify-between gap-6 border-b border-ink-800 pb-6">
        <div className="min-w-0">
          <Link to={`/jobs/${jobId}/clips`} className="eyebrow transition-colors hover:text-sodium-500">
            ← Clips
          </Link>
          <h1 className="mt-2 max-w-3xl truncate font-display text-[clamp(1.75rem,4vw,3rem)] leading-tight text-ink-100">
            Pick your own parts
          </h1>
          <p className="mt-2 truncate text-sm text-ink-400">{job.source?.title || 'Untitled'}</p>
        </div>
        <p className="max-w-sm text-xs leading-relaxed text-ink-500">
          Drag across the graph to pick a part; drag a part's edges to adjust it. Click to jump the
          preview there. Edges snap to the nearest sentence when the clip is made.
        </p>
      </div>

      {job.status !== 'done' && (
        <p className="mt-8 text-sm text-signal-bad">
          Parts can be picked once the job has finished (it is {job.status}).
        </p>
      )}

      <div className="mt-10 grid gap-10 lg:grid-cols-[1fr_20rem]">
        <section className="min-w-0">
          <div className="mb-6 flex flex-wrap items-baseline justify-between gap-4">
            <h2 className="eyebrow">
              {audience.kind === 'audience'
                ? 'Where viewers rewatch'
                : audience.kind === 'energy'
                  ? 'Audio energy (no viewer data for this video)'
                  : 'Timeline (no viewer data for this video)'}
            </h2>
            <div className="flex flex-wrap items-baseline gap-4 text-xs">
              <span className="flex items-center gap-1.5 text-ink-300">
                <span className="inline-block h-0.5 w-4 bg-sodium-500" />
                Blend
              </span>
              {available.map((name) => (
                <label key={name} className="flex cursor-pointer items-center gap-1.5 text-ink-400">
                  <input
                    type="checkbox"
                    checked={visible[name]}
                    onChange={(e) => setVisible({ ...visible, [name]: e.target.checked })}
                  />
                  <span
                    className="inline-block h-0.5 w-4"
                    style={{ background: COMPONENT_STYLE[name].color }}
                  />
                  {COMPONENT_STYLE[name].label}
                </label>
              ))}
            </div>
          </div>

          <div className="pt-5">
            <RewatchGraph
              curve={audience.curve}
              components={audience.components}
              visible={visible}
              stepS={audience.step_s}
              duration={duration}
              bands={audience.clips}
              hotspots={audience.hotspots}
              picks={picks}
              onPicksChange={canPick ? setPicks : () => undefined}
              onSeek={seek}
              playhead={playhead}
            />
          </div>
          <p className="mt-2 text-[0.6875rem] text-ink-600">
            Grey strip: clips already made from this video. Dots: the hottest moments.
          </p>

          {job.source?.has_video !== false && (
            <video
              ref={video}
              src={api.mediaUrl(jobId)}
              controls
              preload="metadata"
              className="mt-8 aspect-video w-full max-w-2xl bg-ink-850"
              onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)}
            />
          )}
        </section>

        <aside>
          <h2 className="eyebrow border-b border-ink-800 pb-3">
            Picked · <span className="numeric">{picks.length}</span>
          </h2>
          {picks.length === 0 ? (
            <p className="mt-4 text-sm text-ink-500">Nothing picked yet.</p>
          ) : (
            <ol className="mt-1">
              {picks.map((p, i) => (
                <li key={i} className="grid grid-cols-[1.5rem_1fr_auto] items-center gap-3 border-b border-ink-850 py-3">
                  <span className="numeric text-xs text-sodium-500">{i + 1}</span>
                  <div className="flex items-center gap-2 text-xs text-ink-300">
                    <EdgeInput label={`Part ${i + 1} start`} value={p.start} onCommit={(v) => updatePick(i, 'start', v)} />
                    <span className="text-ink-600">–</span>
                    <EdgeInput label={`Part ${i + 1} end`} value={p.end} onCommit={(v) => updatePick(i, 'end', v)} />
                    <span className="numeric text-ink-500">{(p.end - p.start).toFixed(1)} s</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <button className="btn btn-quiet px-1.5 py-0.5 text-xs" onClick={() => seek(p.start)} title="Play from here">
                      ▶
                    </button>
                    <button
                      className="btn btn-quiet px-1.5 py-0.5 text-xs"
                      onClick={() => setPicks(picks.filter((_, k) => k !== i))}
                      aria-label={`Remove part ${i + 1}`}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              ))}
            </ol>
          )}

          {picks.length > 0 && (
            <p className="numeric mt-3 text-xs text-ink-500">
              {formatDuration(total)} in total
              {picks.length > 1 && total > MAX_PICK_S && ' · a joined clip stops at 3:00'}
            </p>
          )}
          {tooMany && <p className="mt-2 text-xs text-signal-bad">Pick at most {MAX_PICKS} parts at a time.</p>}

          <label className="mt-6 block text-xs text-ink-400">
            Framing
            <select
              className="field mt-1 w-full text-sm"
              value={framing}
              onChange={(e) => setFraming(e.target.value as Framing | '')}
            >
              <option value="">From settings</option>
              {FRAMINGS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </label>

          <div className="mt-6 flex flex-col gap-3">
            <button
              className="btn btn-primary"
              disabled={!canPick || picks.length === 0 || tooMany}
              onClick={() => submit(false)}
            >
              {submitting ? 'Starting…' : `Make ${picks.length || ''} clip${picks.length === 1 ? '' : 's'}`.replace('  ', ' ')}
            </button>
            <button
              className="btn btn-ghost"
              disabled={!canPick || picks.length < 2 || tooMany}
              onClick={() => submit(true)}
              title="One clip of all the parts, back to back in time order"
            >
              Join into 1 clip
            </button>
          </div>

          {actionError && (
            <div className="mt-6">
              <ErrorNote error={actionError} onDismiss={() => setActionError(null)} />
            </div>
          )}

          {task && <TaskStatus task={task} jobId={jobId} />}
        </aside>
      </div>
    </div>
  )
}

function TaskStatus({ task, jobId }: { task: PickTask; jobId: string }) {
  const count = task.clip_ids.length
  return (
    <div className="mt-8 border-t border-ink-800 pt-5" aria-live="polite">
      {task.state === 'running' && (
        <>
          <p className="text-sm text-ink-200">
            Making {count} clip{count === 1 ? '' : 's'}… <span className="text-ink-500">{task.message}</span>
          </p>
          <div className="mt-3 h-px w-full bg-ink-800">
            <div
              className="h-px origin-left bg-sodium-500 transition-transform duration-500"
              style={{ transform: `scaleX(${task.progress})` }}
            />
          </div>
        </>
      )}
      {task.state === 'done' && (
        <p className="text-sm text-signal-good">
          {count} clip{count === 1 ? '' : 's'} ready.{' '}
          <Link to={`/jobs/${jobId}/clips`} className="text-sodium-500 underline-offset-4 hover:underline">
            Review clips
          </Link>
        </p>
      )}
      {task.state === 'failed' && (
        <ErrorNote error={new Error(task.error || 'Rendering the picked clips failed.')} />
      )}
    </div>
  )
}

/** Seconds field for one edge of a pick; commits on blur or Enter. */
function EdgeInput({ label, value, onCommit }: { label: string; value: number; onCommit: (v: number) => void }) {
  const [text, setText] = useState(value.toFixed(1))
  useEffect(() => setText(value.toFixed(1)), [value])
  const commit = () => {
    const parsed = parseClock(text)
    if (parsed == null) setText(value.toFixed(1))
    else onCommit(parsed)
  }
  return (
    <input
      aria-label={label}
      title={formatDuration(value)}
      className="field numeric w-16 py-0.5 text-xs"
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === 'Enter' && commit()}
    />
  )
}

/** "75.5", "1:15.5" or "1:01:15" to seconds. */
function parseClock(text: string): number | null {
  const parts = text.trim().split(':').map(Number)
  if (!parts.length || parts.some((n) => !Number.isFinite(n) || n < 0)) return null
  return parts.reduce((total, n) => total * 60 + n, 0)
}
