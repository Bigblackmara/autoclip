import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import {
  autopilotApi,
  errorText,
  PLATFORM_NAME,
  STATUS_LABEL,
  statusTone,
  when,
  type Autopilot,
  type AutopilotDetail as Detail,
} from '../publishing/api'

const VIDEO_LABEL: Record<string, string> = {
  waiting: 'Waiting',
  ready: 'Next in line',
  clipping: 'Clipping',
  done: 'Clipped',
  skipped: 'Skipped',
  failed: 'Failed',
}

const CLIP_LABEL: Record<string, string> = {
  review: 'Waiting for you',
  ready: 'In line',
  booked: 'Booked',
  posted: 'Posted',
  rejected: 'Removed',
  failed: 'Failed',
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

/** Every autopilot, with what it is doing. */
export function AutopilotList() {
  const load = useCallback(() => autopilotApi.list(), [])
  const { data, error, reload } = usePoll(load, 15_000)
  const [busy, setBusy] = useState('')

  async function toggle(ap: Autopilot) {
    setBusy(ap.id)
    try {
      await (ap.enabled ? autopilotApi.pause(ap.id) : autopilotApi.resume(ap.id))
      await reload()
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="rise space-y-10 pt-10">
      <header className="flex flex-wrap items-end gap-6">
        <div>
          <p className="eyebrow">Posting</p>
          <h1 className="font-display text-5xl text-ink-100">Autopilot</h1>
          <p className="mt-2 max-w-2xl text-sm text-ink-400">
            Watches channels and topics, waits for YouTube's “Most replayed” graph, clips the best
            new videos and books the clips into your daily posting times.
          </p>
        </div>
        <div className="ml-auto flex gap-3">
          <Link to="/autopilot/new" className="btn btn-primary">
            New autopilot
          </Link>
          <Link to="/schedule" className="btn btn-ghost">
            Schedule
          </Link>
        </div>
      </header>
      {error && <p className="text-sm text-signal-bad">{error}</p>}
      {data && !data.running && (
        <p className="text-sm text-ink-500">
          The background engine isn't running in this process (AUTOCLIP_NO_WORKER), so nothing
          happens on its own.
        </p>
      )}
      {data && data.autopilots.length === 0 && (
        <p className="text-sm text-ink-500">No autopilots yet.</p>
      )}
      <ul className="divide-y divide-ink-800 border-y border-ink-800">
        {(data?.autopilots ?? []).map((ap) => (
          <li key={ap.id} className="grid gap-2 py-4 md:grid-cols-[1fr_auto] md:items-center">
            <div className="min-w-0">
              <Link to={`/autopilot/${ap.id}`} className="text-lg text-ink-100 hover:text-sodium-400">
                {ap.name}
              </Link>
              <span className={`ml-3 text-xs ${ap.enabled ? 'text-signal-good' : 'text-ink-500'}`}>
                {ap.enabled ? 'On' : 'Paused'}
              </span>
              <p className="text-sm text-ink-400">
                {ap.state.message} · {ap.platforms.map((p) => PLATFORM_NAME[p]).join(' + ')} at{' '}
                {ap.slots.join(', ')}
              </p>
              <p className="numeric text-xs text-ink-500">
                {ap.counts.videos_seen} videos seen · {ap.counts.videos_clipping} clipping ·{' '}
                {ap.counts.clips_review} to review · {ap.counts.clips_booked} booked ·{' '}
                {ap.counts.clips_posted} posted
              </p>
              {ap.state.last_error && (
                <p className="text-xs text-signal-bad">{ap.state.last_error}</p>
              )}
            </div>
            <div className="flex gap-2">
              <button type="button" className="btn btn-ghost" disabled={busy === ap.id} onClick={() => toggle(ap)}>
                {ap.enabled ? 'Pause' : 'Resume'}
              </button>
              <Link to={`/autopilot/${ap.id}`} className="btn btn-quiet">
                Open
              </Link>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** One autopilot: its plan, clips to review, recent runs, videos and posts. */
export function AutopilotDetail() {
  const { autopilotId = '' } = useParams()
  const navigate = useNavigate()
  const load = useCallback(() => autopilotApi.get(autopilotId), [autopilotId])
  const { data, error, reload } = usePoll<Detail>(load, 10_000)
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  async function act(fn: () => Promise<unknown>, done?: string) {
    setBusy(true)
    setNotice('')
    try {
      await fn()
      if (done) setNotice(done)
      await reload()
    } catch (err) {
      setNotice(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  if (!data) return <p className="pt-10 text-sm text-ink-400">{error || 'Loading…'}</p>
  const ap = data
  const review = ap.clips.filter((c) => c.status === 'review')

  return (
    <div className="rise space-y-10 pt-10">
      <header className="flex flex-wrap items-end gap-6">
        <div>
          <p className="eyebrow">
            <Link to="/autopilot" className="hover:text-ink-200">
              Autopilot
            </Link>
          </p>
          <h1 className="font-display text-5xl text-ink-100">{ap.name}</h1>
          <p className="mt-2 text-sm text-ink-400">
            <span className={ap.enabled ? 'text-signal-good' : 'text-ink-500'}>
              {ap.enabled ? 'On' : 'Paused - booked posts still go out'}
            </span>{' '}
            · {ap.state.message}
            {ap.state.last_scan_at && ` · last scan ${when(ap.state.last_scan_at)}`}
          </p>
          {ap.state.last_error && <p className="text-sm text-signal-bad">{ap.state.last_error}</p>}
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy || !ap.enabled}
            onClick={() => act(() => autopilotApi.run(ap.id), 'Scanning now; this page updates by itself.')}
          >
            Run now
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            disabled={busy}
            onClick={() => act(() => (ap.enabled ? autopilotApi.pause(ap.id) : autopilotApi.resume(ap.id)))}
          >
            {ap.enabled ? 'Pause' : 'Resume'}
          </button>
          <Link to={`/autopilot/${ap.id}/edit`} className="btn btn-ghost">
            Edit
          </Link>
          <button
            type="button"
            className="btn btn-quiet"
            disabled={busy}
            onClick={() => {
              if (!window.confirm('Delete this autopilot? Its posts that were not sent yet are cancelled.')) return
              void act(async () => {
                await autopilotApi.remove(ap.id)
                navigate('/autopilot')
              })
            }}
          >
            Delete
          </button>
        </div>
      </header>
      {notice && (
        <p className="text-sm text-ink-200" role="status">
          {notice}
        </p>
      )}

      <section>
        <h2 className="eyebrow border-b border-ink-800 pb-2">Next posting times</h2>
        <ul className="divide-y divide-ink-800">
          {ap.plan.map((slot) => (
            <li key={slot.slot_at} className="grid grid-cols-[9rem_1fr_auto] gap-4 py-2 text-sm">
              <span className="numeric text-ink-200">{when(slot.slot_at)}</span>
              <span className="truncate text-ink-100">{slot.title || <span className="text-ink-500">open</span>}</span>
              <span className="text-xs text-ink-400">{slot.clip_id ? CLIP_LABEL[slot.status] ?? slot.status : ''}</span>
            </li>
          ))}
        </ul>
      </section>

      {review.length > 0 && (
        <section>
          <h2 className="eyebrow border-b border-ink-800 pb-2">Waiting for your approval</h2>
          <ul className="divide-y divide-ink-800">
            {review.map((clip) => (
              <li key={clip.clip_id} className="flex flex-wrap items-center gap-4 py-3">
                <span className="text-sm text-ink-100">{clip.title || 'Untitled clip'}</span>
                <span className="numeric text-xs text-ink-500">score {clip.score}</span>
                <Link to={`/jobs/${clip.job_id}/clips`} className="text-xs text-ink-400 underline">
                  watch
                </Link>
                <div className="ml-auto flex gap-2">
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={busy}
                    onClick={() => act(() => autopilotApi.clipAction(ap.id, clip.clip_id, 'approve'))}
                  >
                    Approve
                  </button>
                  <button
                    type="button"
                    className="btn btn-quiet"
                    disabled={busy}
                    onClick={() => act(() => autopilotApi.clipAction(ap.id, clip.clip_id, 'reject'))}
                  >
                    Remove
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <h2 className="eyebrow border-b border-ink-800 pb-2">Clips</h2>
        {ap.clips.length === 0 && <p className="py-3 text-sm text-ink-500">No clips yet.</p>}
        <ul className="divide-y divide-ink-800">
          {ap.clips
            .filter((c) => c.status !== 'review')
            .map((clip) => (
              <li key={clip.clip_id} className="flex flex-wrap items-center gap-4 py-2 text-sm">
                <span className="text-ink-100">{clip.title || 'Untitled clip'}</span>
                <span className="text-xs text-ink-400">
                  {CLIP_LABEL[clip.status] ?? clip.status}
                  {clip.slot_key && ` · ${when(clip.slot_key)}`}
                  {clip.reason && ` · ${clip.reason}`}
                </span>
                {(clip.status === 'ready' || clip.status === 'booked') && (
                  <button
                    type="button"
                    className="btn btn-quiet ml-auto"
                    disabled={busy}
                    onClick={() => act(() => autopilotApi.clipAction(ap.id, clip.clip_id, 'reject'))}
                  >
                    Remove
                  </button>
                )}
              </li>
            ))}
        </ul>
      </section>

      <section>
        <h2 className="eyebrow border-b border-ink-800 pb-2">Booked posts</h2>
        {ap.posts.length === 0 && <p className="py-3 text-sm text-ink-500">Nothing booked yet.</p>}
        <ul className="divide-y divide-ink-800">
          {ap.posts.map((post) => (
            <li key={post.id} className="grid grid-cols-[9rem_6rem_1fr] gap-4 py-2 text-sm">
              <span className="numeric text-ink-200">{when(post.scheduled_at)}</span>
              <span className="text-ink-300">{PLATFORM_NAME[post.platform]}</span>
              <span className="min-w-0 truncate">
                <span className="text-ink-100">{post.title}</span>{' '}
                <span className={`text-xs ${statusTone(post.status)}`}>{STATUS_LABEL[post.status]}</span>
                {post.error && <span className="text-xs text-ink-500"> · {post.error}</span>}
              </span>
            </li>
          ))}
        </ul>
        <p className="mt-2 text-xs text-ink-500">
          Cancel or retry single posts on the <Link to="/schedule" className="underline">Schedule</Link> page.
        </p>
      </section>

      <section>
        <h2 className="eyebrow border-b border-ink-800 pb-2">Recent runs</h2>
        {ap.runs.length === 0 && <p className="py-3 text-sm text-ink-500">No runs yet.</p>}
        <ul className="divide-y divide-ink-800">
          {ap.runs.map((r) => (
            <li key={r.at} className="grid grid-cols-[9rem_1fr] gap-4 py-2 text-sm">
              <span className="numeric text-ink-400">{when(r.at)}</span>
              <span className="text-ink-200">
                {r.message}
                {r.errors.length > 0 && <span className="block text-xs text-signal-bad">{r.errors.join(' · ')}</span>}
              </span>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="eyebrow border-b border-ink-800 pb-2">Videos seen</h2>
        {ap.videos.length === 0 && <p className="py-3 text-sm text-ink-500">Nothing found yet.</p>}
        <ul className="divide-y divide-ink-800">
          {ap.videos.map((v) => (
            <li key={v.id} className="grid gap-1 py-2 text-sm md:grid-cols-[1fr_8rem_20rem] md:gap-4">
              <a href={v.url} target="_blank" rel="noreferrer" className="truncate text-ink-100 hover:text-sodium-400">
                {v.title || v.url}
                <span className="block text-xs text-ink-500">
                  {v.channel || v.source}
                  {v.views != null && ` · ${v.views.toLocaleString()} views`}
                  {v.upload_date && ` · ${v.upload_date}`}
                </span>
              </a>
              <span className="text-xs text-ink-300">{VIDEO_LABEL[v.status] ?? v.status}</span>
              <span className="text-xs text-ink-500">{v.reason}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
