import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { api, formatDuration } from '../api'
import { ErrorNote } from '../components/ErrorNote'
import { useJobStream } from '../useJobStream'
import {
  Badge,
  Button,
  ButtonLink,
  Card,
  CheckIcon,
  PageHeader,
  PageSkeleton,
  ProgressBar,
  RefreshIcon,
  ScissorsIcon,
  Spinner,
  XIcon,
  cx,
  type Tone,
} from '../ui'

/** Stage order and labels, mirroring autoclip.pipeline.Stage. */
const STAGES = [
  { key: 'prepare', label: 'Getting the video ready', note: 'Extracting audio and thumbnails' },
  { key: 'transcribe', label: 'Writing down every word', note: 'Word-level timing for captions' },
  { key: 'highlights', label: 'Finding the best moments', note: 'The AI picks what is worth cutting' },
  { key: 'reframe', label: 'Reframing to vertical', note: 'Following the speaker into 9:16' },
  { key: 'captions', label: 'Timing the captions', note: 'Building subtitle timing' },
  { key: 'export', label: 'Rendering clips', note: 'Making the video files' },

] as const

export function JobProgress() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const { job, progress } = useJobStream(jobId)
  const [cancelling, setCancelling] = useState(false)
  const [retrying, setRetrying] = useState(false)
  const [actionError, setActionError] = useState<Error | null>(null)
  // Whether the job was already finished when this page opened: then the
  // page is a destination (e.g. to pick parts), not a progress screen.
  const doneOnArrival = useRef<boolean | null>(null)
  if (job && doneOnArrival.current === null) doneOnArrival.current = job.status === 'done'

  useEffect(() => {
    if (job?.status === 'done' && !doneOnArrival.current) {
      // A short hold so the completed state is legible rather than a flash.
      const timer = setTimeout(() => navigate(`/jobs/${job.id}/clips`), 900)
      return () => clearTimeout(timer)
    }
  }, [job?.status, job?.id, navigate])

  if (!job) {
    return <PageSkeleton cards={1} />
  }

  const activeIndex = STAGES.findIndex((s) => s.key === job.current_stage)
  const percent = Math.round((progress?.overall ?? job.progress) * 100)

  const cancel = async () => {
    setCancelling(true)
    setActionError(null)
    try {
      await api.cancelJob(job.id)
      // Stay "Cancelling…" on success: a running job only stops at its next
      // cancel check, and re-enabling the button meanwhile invited repeat
      // clicks on a job that was already on its way out. The button goes
      // away when the stream reports "cancelled".
    } catch (err) {
      setActionError(err as Error)
      setCancelling(false)
    }
  }

  const retry = async () => {
    setRetrying(true)
    setActionError(null)
    try {
      await api.retryJob(job.id)
      window.location.reload()
    } catch (err) {
      setActionError(err as Error)
      setRetrying(false)
    }
  }

  const statusLook: Record<string, { tone: Tone; label: string }> = {
    queued: { tone: 'neutral', label: 'Waiting to start' },
    running: { tone: 'accent', label: 'Working on it' },
    done: { tone: 'success', label: 'Finished' },
    failed: { tone: 'danger', label: 'Stopped with an error' },
    cancelled: { tone: 'neutral', label: 'Cancelled' },
  }
  const look = statusLook[job.status] ?? { tone: 'neutral' as Tone, label: job.status }

  return (
    <div>
      <PageHeader
        back={{ to: '/projects', label: 'All projects' }}
        eyebrow={
          <Badge tone={look.tone} dot pulse={job.status === 'running'}>
            {look.label}
          </Badge>
        }
        title={job.source?.title || 'Untitled'}
        description={
          job.source ? (
            <span className="numeric">
              {formatDuration(job.source.duration_s)}
              {job.source.width ? ` · ${job.source.width}×${job.source.height}` : ''} · {job.provider}
            </span>
          ) : undefined
        }
        actions={
          <>
            {(job.status === 'running' || job.status === 'queued') && (
              <Button onClick={cancel} loading={cancelling} icon={XIcon}>
                {cancelling ? 'Cancelling…' : 'Cancel'}
              </Button>
            )}
            {(job.status === 'failed' || job.status === 'cancelled') && (
              <Button variant="primary" onClick={retry} loading={retrying} icon={RefreshIcon}>
                {retrying ? 'Retrying…' : 'Try again'}
              </Button>
            )}
            {job.status === 'done' && (
              <ButtonLink to={`/jobs/${job.id}/pick`} icon={ScissorsIcon}>
                Pick parts
              </ButtonLink>
            )}
            {job.status === 'done' && (
              <Button variant="primary" onClick={() => navigate(`/jobs/${job.id}/clips`)}>
                Review clips
              </Button>
            )}
          </>
        }
      />

      {job.error && (
        <div className="mb-6 max-w-3xl">
          <ErrorNote error={new Error(job.error)} />
          <p className="mt-2 text-xs text-fg-muted">
            “Try again” resumes from the step that failed — finished steps are not redone.
          </p>
        </div>
      )}

      {actionError && (
        <div className="mb-6 max-w-3xl">
          <ErrorNote error={actionError} onDismiss={() => setActionError(null)} />
        </div>
      )}

      <Card className="max-w-3xl" padding="lg">
        <div className="flex items-end justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-fg-muted">Overall progress</p>
            <p className="numeric mt-1 text-4xl font-bold tracking-tight text-fg">
              {percent}
              <span className="text-xl text-fg-subtle">%</span>
            </p>
          </div>
          {job.status === 'running' && (
            <p className="max-w-xs text-right text-xs text-fg-muted">
              You can leave this page — the work keeps going and shows up in Projects.
            </p>
          )}
        </div>
        <ProgressBar
          value={percent / 100}
          className="mt-4 h-2"
          tone={job.status === 'failed' ? 'danger' : job.status === 'done' ? 'success' : 'accent'}
          label="Overall progress"
        />

        <ol className="mt-8 space-y-1">
          {STAGES.map((stage, index) => {
            // A failed job stopped *at* current_stage, so that stage must read as
            // failed rather than as still running — otherwise the checklist claims
            // work is in progress while the banner says it died.
            const failedHere =
              (job.status === 'failed' || job.status === 'cancelled') && index === activeIndex

            const state =
              job.status === 'done' || (activeIndex >= 0 && index < activeIndex)
                ? 'done'
                : failedHere
                  ? 'stopped'
                  : index === activeIndex && job.status === 'running'
                    ? 'active'
                    : 'pending'

            const fraction =
              state === 'done' ? 1 : state === 'active' ? (progress?.stageProgress ?? 0) : 0

            return (
              <li
                key={stage.key}
                className={cx(
                  'grid grid-cols-[2rem_1fr] items-start gap-x-4 rounded-xl px-3 py-3',
                  state === 'active' && 'bg-accent-soft/60',
                )}
              >
                <span
                  className={cx(
                    'numeric grid size-8 place-items-center rounded-full text-xs font-semibold',
                    state === 'done' && 'bg-success-soft text-success',
                    state === 'stopped' && 'bg-danger-soft text-danger',
                    state === 'active' && 'bg-accent text-accent-fg',
                    state === 'pending' && 'bg-subtle text-fg-subtle',
                  )}
                >
                  {state === 'done' ? (
                    <CheckIcon size={15} />
                  ) : state === 'stopped' ? (
                    <XIcon size={15} />
                  ) : state === 'active' ? (
                    <Spinner size={14} />
                  ) : (
                    index + 1
                  )}
                </span>

                <div className="min-w-0 pt-1">
                  <div className="flex flex-wrap items-baseline justify-between gap-x-4">
                    <span className={cx('text-sm font-semibold', state === 'pending' ? 'text-fg-subtle' : 'text-fg')}>
                      {stage.label}
                    </span>
                    <span className={cx('text-xs', state === 'stopped' ? 'text-danger' : 'text-fg-muted')}>
                      {state === 'stopped'
                        ? job.status === 'cancelled'
                          ? 'cancelled here'
                          : 'failed here'
                        : state === 'active'
                          ? (progress?.message ?? stage.note)
                          : stage.note}
                    </span>
                  </div>
                  {state === 'active' && <ProgressBar value={fraction} className="mt-2.5" label={stage.label} />}
                </div>
              </li>
            )
          })}
        </ol>
      </Card>
    </div>
  )
}
