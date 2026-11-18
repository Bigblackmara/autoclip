import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import {
  autopilotApi,
  errorText,
  PLATFORM_NAME,
  when,
  type Autopilot,
  type AutopilotDetail as Detail,
} from '../publishing/api'
import { PlatformTag, PostStatusBadge } from '../publishing/ui'
import {
  Badge,
  Button,
  ButtonLink,
  CalendarIcon,
  Callout,
  Card,
  CheckIcon,
  Dialog,
  EmptyState,
  ExternalIcon,
  PageHeader,
  PageSkeleton,
  PlayIcon,
  PlusIcon,
  RocketIcon,
  SettingsIcon,
  Skeleton,
  TrashIcon,
  XIcon,
  useToast,
  type Tone,
} from '../ui'

const VIDEO_LOOK: Record<string, { label: string; tone: Tone }> = {
  waiting: { label: 'Waiting', tone: 'neutral' },
  ready: { label: 'Next in line', tone: 'accent' },
  clipping: { label: 'Clipping', tone: 'info' },
  done: { label: 'Clipped', tone: 'success' },
  skipped: { label: 'Skipped', tone: 'neutral' },
  failed: { label: 'Failed', tone: 'danger' },
}

const CLIP_LOOK: Record<string, { label: string; tone: Tone }> = {
  review: { label: 'Waiting for you', tone: 'warn' },
  ready: { label: 'In line', tone: 'accent' },
  booked: { label: 'Booked', tone: 'info' },
  posted: { label: 'Posted', tone: 'success' },
  rejected: { label: 'Removed', tone: 'neutral' },
  failed: { label: 'Failed', tone: 'danger' },
}

function usePoll<T>(load: () => Promise<T>, ms: number) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState('')
  const reload = useCallback(async () => {
    try {
      setData(await load())
      setError('')
    } catch (err) {
      setError(errorText(err))
    }
  }, [load])
  useEffect(() => {
    void reload()
    const timer = window.setInterval(() => void reload(), ms)
    return () => window.clearInterval(timer)
  }, [reload, ms])
  return { data, error, reload }
}

function OnBadge({ enabled }: { enabled: boolean }) {
  return enabled ? (
    <Badge tone="success" dot pulse>
      On
    </Badge>
  ) : (
    <Badge>Paused</Badge>
  )
}

/** Every autopilot, with what it is doing. */
export function AutopilotList() {
  const load = useCallback(() => autopilotApi.list(), [])
  const { data, error, reload } = usePoll(load, 15_000)
  const [busy, setBusy] = useState('')
  const toast = useToast()

  async function toggle(ap: Autopilot) {
    setBusy(ap.id)
    try {
      await (ap.enabled ? autopilotApi.pause(ap.id) : autopilotApi.resume(ap.id))
      toast.success(ap.enabled ? `${ap.name} paused` : `${ap.name} is on`)
      await reload()
    } catch (err) {
      toast.error('That didn’t work', errorText(err))
    } finally {
      setBusy('')
    }
  }

  return (
    <div>
      <PageHeader
        title="Autopilot"
        description="Watches channels and topics, waits for YouTube’s “Most replayed” graph, clips the best new videos and books the clips into your daily posting times."
        actions={
          <>
            <ButtonLink to="/autopilot/new" variant="primary" icon={PlusIcon}>
              New autopilot
            </ButtonLink>
            <ButtonLink to="/schedule" icon={CalendarIcon}>
              Schedule
            </ButtonLink>
          </>
        }
      />
      {error && (
        <Callout tone="danger" className="mb-6">
          {error}
        </Callout>
      )}
      {data && !data.running && (
        <Callout tone="warn" className="mb-6" title="The background engine isn't running">
          This AutoClip was started without its worker (AUTOCLIP_NO_WORKER), so autopilots don't do anything on
          their own.
        </Callout>
      )}
      {!data && !error && (
        <div className="grid gap-4">
          <Skeleton className="h-28 w-full rounded-[var(--radius-card)]" />
          <Skeleton className="h-28 w-full rounded-[var(--radius-card)]" />
        </div>
      )}
      {data && data.autopilots.length === 0 && (
        <EmptyState
          icon={RocketIcon}
          title="No autopilots yet"
          description="An autopilot finds new videos from channels or topics you pick, clips them and posts the clips at your daily times. You can approve every clip first."
          action={
            <ButtonLink to="/autopilot/new" variant="primary" icon={PlusIcon}>
              Create your first autopilot
            </ButtonLink>
          }
        />
      )}
      <div className="grid gap-4">
        {(data?.autopilots ?? []).map((ap) => (
          <Card key={ap.id} padding="md">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <Link to={`/autopilot/${ap.id}`} className="text-base font-semibold text-fg hover:text-accent-text">
                    {ap.name}
                  </Link>
                  <OnBadge enabled={ap.enabled} />
                  {ap.counts.clips_review > 0 && <Badge tone="warn">{ap.counts.clips_review} to review</Badge>}
                </div>
                {ap.state.message && <p className="mt-1 text-sm text-fg-muted">{ap.state.message}</p>}
                <p className="mt-1 text-sm text-fg-muted">
                  Posts to {ap.platforms.map((p) => PLATFORM_NAME[p]).join(' + ') || 'nowhere yet'} at{' '}
                  <span className="numeric font-medium text-fg">{ap.slots.join(', ') || '—'}</span>
                </p>
                <div className="numeric mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-fg-muted">
                  <span>
                    <b className="text-fg">{ap.counts.videos_seen}</b> videos seen
                  </span>
                  <span>
                    <b className="text-fg">{ap.counts.videos_clipping}</b> clipping
                  </span>
                  <span>
                    <b className="text-fg">{ap.counts.clips_booked}</b> booked
                  </span>
                  <span>
                    <b className="text-fg">{ap.counts.clips_posted}</b> posted
                  </span>
                </div>
                {ap.state.last_error && <p className="mt-2 text-xs text-danger">{ap.state.last_error}</p>}
              </div>
              <div className="flex shrink-0 gap-2">
                <Button size="sm" disabled={busy === ap.id} onClick={() => toggle(ap)}>
                  {ap.enabled ? 'Pause' : 'Resume'}
                </Button>
                <ButtonLink size="sm" to={`/autopilot/${ap.id}`} variant="primary">
                  Open
                </ButtonLink>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}

/** One autopilot: its plan, clips to review, recent runs, videos and posts. */
export function AutopilotDetail() {
  const { autopilotId = '' } = useParams()
  const navigate = useNavigate()
  const load = useCallback(() => autopilotApi.get(autopilotId), [autopilotId])
  const { data, error, reload } = usePoll<Detail>(load, 10_000)
  const [busy, setBusy] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const toast = useToast()

  async function act(fn: () => Promise<unknown>, done?: string) {
    setBusy(true)
    try {
      await fn()
      if (done) toast.success(done)
      await reload()
    } catch (err) {
      toast.error('That didn’t work', errorText(err))
    } finally {
      setBusy(false)
    }
  }

  if (!data) {
    return error ? (
      <div className="max-w-3xl">
        <PageHeader back={{ to: '/autopilot', label: 'All autopilots' }} title="Autopilot" />
        <Callout tone="danger">{error}</Callout>
      </div>
    ) : (
      <PageSkeleton cards={3} />
    )
  }
  const ap = data
  const review = ap.clips.filter((c) => c.status === 'review')
  const others = ap.clips.filter((c) => c.status !== 'review')

  return (
    <div>
      <PageHeader
        back={{ to: '/autopilot', label: 'All autopilots' }}
        eyebrow={<OnBadge enabled={ap.enabled} />}
        title={ap.name}
        description={
          <>
            {ap.enabled ? ap.state.message : 'Paused — posts already booked still go out.'}
            {ap.state.last_scan_at && ` Last scan ${when(ap.state.last_scan_at)}.`}
          </>
        }
        actions={
          <>
            <Button
              variant="primary"
              icon={PlayIcon}
              disabled={busy || !ap.enabled}
              title={!ap.enabled ? 'Resume it first' : 'Look for new videos now'}
              onClick={() => act(() => autopilotApi.run(ap.id), 'Scanning now — this page updates by itself.')}
            >
              Run now
            </Button>
            <Button
              disabled={busy}
              onClick={() =>
                act(() => (ap.enabled ? autopilotApi.pause(ap.id) : autopilotApi.resume(ap.id)), ap.enabled ? 'Paused' : 'Resumed')
              }
            >
              {ap.enabled ? 'Pause' : 'Resume'}
            </Button>
            <ButtonLink to={`/autopilot/${ap.id}/edit`} icon={SettingsIcon}>
              Edit
            </ButtonLink>
            <Button variant="ghost" icon={TrashIcon} disabled={busy} onClick={() => setConfirmDelete(true)}>
              Delete
            </Button>
          </>
        }
      />
      {ap.state.last_error && (
        <Callout tone="danger" className="mb-6" title="Last run had a problem">
          {ap.state.last_error}
        </Callout>
      )}

      <div className="grid items-start gap-6 xl:grid-cols-2">
        {review.length > 0 && (
          <Card
            padding="none"
            className="xl:col-span-2"
            title={`Waiting for your approval (${review.length})`}
            description="Approved clips are booked into the next free posting time."
          >
            <ul className="divide-y divide-line border-t border-line">
              {review.map((clip) => (
                <li key={clip.clip_id} className="flex flex-wrap items-center gap-3 px-5 py-3 sm:px-6">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-fg">{clip.title || 'Untitled clip'}</p>
                    <p className="numeric text-xs text-fg-muted">
                      Score {clip.score} ·{' '}
                      <Link to={`/jobs/${clip.job_id}/clips`} className="link">
                        Watch it
                      </Link>
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="primary"
                      icon={CheckIcon}
                      disabled={busy}
                      onClick={() => act(() => autopilotApi.clipAction(ap.id, clip.clip_id, 'approve'), 'Approved')}
                    >
                      Approve
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      icon={XIcon}
                      disabled={busy}
                      onClick={() => act(() => autopilotApi.clipAction(ap.id, clip.clip_id, 'reject'), 'Removed')}
                    >
                      Remove
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          </Card>
        )}

        <Card padding="none" title="Next posting times" icon={CalendarIcon}>
          <ul className="divide-y divide-line border-t border-line">
            {ap.plan.length === 0 && <li className="px-5 py-6 text-sm text-fg-muted sm:px-6">No posting times set.</li>}
            {ap.plan.map((slot) => (
              <li key={slot.slot_at} className="grid grid-cols-[8.5rem_minmax(0,1fr)] items-center gap-3 px-5 py-2.5 sm:px-6">
                <span className="numeric text-sm font-medium text-fg">{when(slot.slot_at)}</span>
                <span className="flex min-w-0 items-center justify-between gap-2">
                  <span className="truncate text-sm text-fg">
                    {slot.title || <span className="text-fg-subtle">Open slot</span>}
                  </span>
                  {slot.clip_id && (
                    <Badge tone={CLIP_LOOK[slot.status]?.tone ?? 'neutral'}>{CLIP_LOOK[slot.status]?.label ?? slot.status}</Badge>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card padding="none" title="Booked posts">
          <ul className="divide-y divide-line border-t border-line">
            {ap.posts.length === 0 && <li className="px-5 py-6 text-sm text-fg-muted sm:px-6">Nothing booked yet.</li>}
            {ap.posts.map((post) => (
              <li key={post.id} className="px-5 py-2.5 sm:px-6">
                <p className="truncate text-sm font-medium text-fg">{post.title}</p>
                <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-muted">
                  <span className="numeric">{when(post.scheduled_at)}</span>
                  <PlatformTag platform={post.platform} />
                  <PostStatusBadge status={post.status} />
                  {post.error && <span className="text-danger">{post.error}</span>}
                </div>
              </li>
            ))}
          </ul>
          <p className="border-t border-line px-5 py-3 text-xs text-fg-muted sm:px-6">
            Cancel or retry single posts on the{' '}
            <Link to="/schedule" className="link">
              Schedule
            </Link>{' '}
            page.
          </p>
        </Card>

        <Card padding="none" title="Clips">
          <ul className="divide-y divide-line border-t border-line">
            {others.length === 0 && <li className="px-5 py-6 text-sm text-fg-muted sm:px-6">No clips yet.</li>}
            {others.map((clip) => (
              <li key={clip.clip_id} className="flex flex-wrap items-center gap-3 px-5 py-2.5 sm:px-6">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm text-fg">{clip.title || 'Untitled clip'}</p>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-fg-muted">
                    <Badge tone={CLIP_LOOK[clip.status]?.tone ?? 'neutral'}>{CLIP_LOOK[clip.status]?.label ?? clip.status}</Badge>
                    {clip.slot_key && <span className="numeric">{when(clip.slot_key)}</span>}
                    {clip.reason && <span>{clip.reason}</span>}
                  </div>
                </div>
                {(clip.status === 'ready' || clip.status === 'booked') && (
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={busy}
                    onClick={() => act(() => autopilotApi.clipAction(ap.id, clip.clip_id, 'reject'), 'Removed')}
                  >
                    Remove
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </Card>

        <Card padding="none" title="Recent runs">
          <ul className="divide-y divide-line border-t border-line">
            {ap.runs.length === 0 && <li className="px-5 py-6 text-sm text-fg-muted sm:px-6">No runs yet.</li>}
            {ap.runs.map((r) => (
              <li key={r.at} className="grid grid-cols-[8.5rem_minmax(0,1fr)] gap-3 px-5 py-2.5 text-sm sm:px-6">
                <span className="numeric text-fg-muted">{when(r.at)}</span>
                <span className="min-w-0 text-fg">
                  {r.message}
                  {r.errors.length > 0 && (
                    <span className="block text-xs text-danger [overflow-wrap:anywhere]">{r.errors.join(' · ')}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </Card>

        <Card padding="none" className="xl:col-span-2" title="Videos found">
          <ul className="divide-y divide-line border-t border-line">
            {ap.videos.length === 0 && <li className="px-5 py-6 text-sm text-fg-muted sm:px-6">Nothing found yet.</li>}
            {ap.videos.map((v) => (
              <li key={v.id} className="flex flex-col gap-1 px-5 py-2.5 sm:px-6 md:flex-row md:items-center md:gap-4">
                <a href={v.url} target="_blank" rel="noreferrer" className="group min-w-0 flex-1">
                  <span className="flex items-center gap-1 truncate text-sm font-medium text-fg group-hover:text-accent-text">
                    <span className="truncate">{v.title || v.url}</span>
                    <ExternalIcon size={12} className="text-fg-subtle" />
                  </span>
                  <span className="block truncate text-xs text-fg-muted">
                    {v.channel || v.source}
                    {v.views != null && ` · ${v.views.toLocaleString()} views`}
                    {v.upload_date && ` · ${v.upload_date}`}
                  </span>
                </a>
                <div className="flex flex-wrap items-center gap-2 md:w-80 md:justify-end">
                  {v.reason && <span className="text-xs text-fg-muted">{v.reason}</span>}
                  <Badge tone={VIDEO_LOOK[v.status]?.tone ?? 'neutral'}>{VIDEO_LOOK[v.status]?.label ?? v.status}</Badge>
                </div>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <Dialog
        open={confirmDelete}
        onClose={() => setConfirmDelete(false)}
        size="sm"
        title={`Delete “${ap.name}”?`}
        description="Posts it booked that weren't sent yet are cancelled. Clips it already made stay in Projects."
        footer={
          <>
            <Button onClick={() => setConfirmDelete(false)}>Keep it</Button>
            <Button
              variant="danger"
              data-autofocus
              onClick={() => {
                setConfirmDelete(false)
                void act(async () => {
                  await autopilotApi.remove(ap.id)
                  toast.success('Autopilot deleted')
                  navigate('/autopilot')
                })
              }}
            >
              Delete
            </Button>
          </>
        }
      />
    </div>
  )
}
