import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, formatDuration, FRAMINGS, type Framing, type Job } from '../api'
import { ErrorNote } from '../components/ErrorNote'
import { COMPONENT_STYLE, RewatchGraph } from '../components/RewatchGraph'
import { MAX_PICK_S, MAX_PICKS, MIN_PICK_S, picksApi, type Audience, type PickTask, type TimeRange } from '../picks/api'
import {
  Button,
  Callout,
  Card,
  Field,
  IconButton,
  PageHeader,
  PageSkeleton,
  PlayIcon,
  ProgressBar,
  ScissorsIcon,
  Select,
  XIcon,
} from '../ui'


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
      <div className="max-w-3xl">
        <PageHeader back={{ to: `/jobs/${jobId}/clips`, label: 'Back to clips' }} title="Pick your own parts" />
        <ErrorNote error={loadError} />
      </div>
    )
  }
  if (!job || !audience) {
    return <PageSkeleton cards={2} />
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
    <div>
      <PageHeader
        back={{ to: `/jobs/${jobId}/clips`, label: 'Back to clips' }}
        title="Pick your own parts"
        description={
          <>
            <span className="font-medium text-fg">{job.source?.title || 'Untitled'}</span> — drag across the graph to
            pick a part, drag its edges to adjust, click to jump the preview there.
          </>
        }
      />

      {job.status !== 'done' && (
        <Callout tone="warn" className="mb-6">
          Parts can be picked once the job has finished (it is {job.status}).
        </Callout>
      )}

      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 space-y-6">
          <Card
            title={
              audience.kind === 'audience'
                ? 'Where viewers rewatch'
                : audience.kind === 'energy'
                  ? 'Audio energy'
                  : 'Timeline'
            }
            description={
              audience.kind === 'audience'
                ? 'Peaks are the moments people replay most.'
                : 'No viewer data for this video, so this shows the timeline.'
            }
            actions={
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                <span className="flex items-center gap-1.5 font-medium text-fg-muted">
                  <span className="inline-block h-0.5 w-4 rounded bg-accent" />
                  Blend
                </span>
                {available.map((name) => (
                  <label key={name} className="flex cursor-pointer items-center gap-1.5 text-fg-muted">
                    <input
                      type="checkbox"
                      checked={visible[name]}
                      onChange={(e) => setVisible({ ...visible, [name]: e.target.checked })}
                    />
                    <span className="inline-block h-0.5 w-4" style={{ background: COMPONENT_STYLE[name].color }} />
                    {COMPONENT_STYLE[name].label}
                  </label>
                ))}
              </div>
            }
          >
            <div className="pt-2">
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
            <p className="mt-3 text-xs text-fg-muted">
              Grey strip: clips already made from this video. Dots: the hottest moments. Edges snap to the nearest
              sentence when the clip is made.
            </p>
          </Card>

          {job.source?.has_video !== false && (
            <Card padding="sm" title="Preview">
              <video
                ref={video}
                src={api.mediaUrl(jobId)}
                controls
                preload="metadata"
                className="aspect-video w-full rounded-lg bg-media"
                onTimeUpdate={(e) => setPlayhead(e.currentTarget.currentTime)}
              />
            </Card>
          )}
        </div>

        <Card as="aside" className="lg:sticky lg:top-6" title={`Picked parts (${picks.length})`}>
          {picks.length === 0 ? (
            <p className="rounded-lg border border-dashed border-line-strong px-4 py-6 text-center text-sm text-fg-muted">
              Nothing picked yet. Drag across the graph to pick a part.
            </p>
          ) : (
            <ol className="-mx-1 divide-y divide-line">
              {picks.map((p, i) => (
                <li key={i} className="grid grid-cols-[1.5rem_1fr_auto] items-center gap-2 px-1 py-2.5">
                  <span className="numeric grid size-6 place-items-center rounded-full bg-accent-soft text-xs font-semibold text-accent-text">
                    {i + 1}
                  </span>
                  <div className="flex items-center gap-1.5 text-xs text-fg-muted">
                    <EdgeInput label={`Part ${i + 1} start`} value={p.start} onCommit={(v) => updatePick(i, 'start', v)} />
                    <span className="text-fg-subtle">–</span>
                    <EdgeInput label={`Part ${i + 1} end`} value={p.end} onCommit={(v) => updatePick(i, 'end', v)} />
                    <span className="numeric whitespace-nowrap">{(p.end - p.start).toFixed(1)} s</span>
                  </div>
                  <div className="flex items-center">
                    <IconButton icon={PlayIcon} label={`Play part ${i + 1}`} size="sm" onClick={() => seek(p.start)} />
                    <IconButton
                      icon={XIcon}
                      label={`Remove part ${i + 1}`}
                      size="sm"
                      onClick={() => setPicks(picks.filter((_, k) => k !== i))}
                    />
                  </div>
                </li>
              ))}
            </ol>
          )}

          {picks.length > 0 && (
            <p className="numeric mt-3 text-xs text-fg-muted">
              {formatDuration(total)} in total
              {picks.length > 1 && total > MAX_PICK_S && ' · a joined clip stops at 3:00'}
            </p>
          )}
          {tooMany && <p className="mt-2 text-xs text-danger">Pick at most {MAX_PICKS} parts at a time.</p>}

          <Field label="Framing" className="mt-5" hint="How the picture is fitted into the vertical frame.">
            <Select value={framing} onChange={(e) => setFraming(e.target.value as Framing | '')}>
              <option value="">From settings</option>
              {FRAMINGS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </Select>
          </Field>

          <div className="mt-5 flex flex-col gap-2">
            <Button
              variant="primary"
              icon={ScissorsIcon}
              loading={submitting}
              disabled={!canPick || picks.length === 0 || tooMany}
              onClick={() => submit(false)}
            >
              {submitting
                ? 'Starting…'
                : `Make ${picks.length || ''} clip${picks.length === 1 ? '' : 's'}`.replace('  ', ' ')}
            </Button>
            <Button
              disabled={!canPick || picks.length < 2 || tooMany}
              onClick={() => submit(true)}
              title="One clip of all the parts, back to back in time order"
            >
              Join into 1 clip
            </Button>
          </div>

          {actionError && (
            <div className="mt-5">
              <ErrorNote error={actionError} onDismiss={() => setActionError(null)} />
            </div>
          )}

          {task && <TaskStatus task={task} jobId={jobId} />}
        </Card>
      </div>
    </div>
  )
}

function TaskStatus({ task, jobId }: { task: PickTask; jobId: string }) {
  const count = task.clip_ids.length
  return (
    <div className="mt-5 border-t border-line pt-4" aria-live="polite">
      {task.state === 'running' && (
        <>
          <p className="text-sm text-fg">
            Making {count} clip{count === 1 ? '' : 's'}… <span className="text-fg-muted">{task.message}</span>
          </p>
          <ProgressBar value={task.progress} className="mt-3" label="Making clips" />
        </>
      )}
      {task.state === 'done' && (
        <Callout tone="success">
          {count} clip{count === 1 ? '' : 's'} ready.{' '}
          <Link to={`/jobs/${jobId}/clips`} className="link font-medium">
            Review clips
          </Link>
        </Callout>
      )}
      {task.state === 'failed' && <ErrorNote error={new Error(task.error || 'Rendering the picked clips failed.')} />}
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
      className="field numeric w-16 px-1.5 py-1 text-xs"
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
