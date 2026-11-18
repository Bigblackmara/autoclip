import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { PostClipPanel } from '../components/PostClipPanel'
import {
  errorText,
  PLATFORM_NAME,
  publishingApi,
  STATUS_LABEL,
  statusTone,
  type Platform,
  type PostableClip,
  type PublishingStatus,
  type ScheduleItem,
  type ScheduleOut,
} from '../publishing/api'

const SOURCE_LABEL: Record<ScheduleItem['source'], string> = {
  normal: 'You',
  autopilot: 'Autopilot',
  gta: 'GTA Mode',
}

function dayOf(iso: string | null): string {
  if (!iso) return 'As soon as possible'
  const d = new Date(iso)
  const today = new Date()
  if (d.toDateString() === today.toDateString()) return 'Today'
  if (d.toDateString() === new Date(Date.now() + 86_400_000).toDateString()) return 'Tomorrow'
  if (d.toDateString() === new Date(Date.now() - 86_400_000).toDateString()) return 'Yesterday'
  return d.toLocaleDateString(undefined, { weekday: 'long', day: 'numeric', month: 'long' })
}

/** Every post going out (or gone out): clips you post, Autopilot, and GTA Mode. */
export function Schedule() {
  const [data, setData] = useState<ScheduleOut | null>(null)
  const [status, setStatus] = useState<PublishingStatus | null>(null)
  const [tab, setTab] = useState<'upcoming' | 'past'>('upcoming')
  const [only, setOnly] = useState<'all' | Platform>('all')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busyKey, setBusyKey] = useState('')
  const [showPost, setShowPost] = useState(false)

  const reload = useCallback(async () => {
    try {
      const [sched, st] = await Promise.all([publishingApi.schedule(), publishingApi.status()])
      setData(sched)
      setStatus(st)
      setError('')
    } catch (err) {
      setError(errorText(err))
    }
  }, [])

  useEffect(() => {
    void reload()
    const timer = window.setInterval(() => void reload(), 10_000)
    return () => window.clearInterval(timer)
  }, [reload])

  const items = useMemo(() => {
    const list = (tab === 'upcoming' ? data?.upcoming : data?.past) ?? []
    return list.filter((i) => only === 'all' || i.platform === only)
  }, [data, tab, only])

  const days = useMemo(() => {
    const map = new Map<string, ScheduleItem[]>()
    for (const item of items) map.set(dayOf(item.when), [...(map.get(dayOf(item.when)) ?? []), item])
    return [...map.entries()]
  }, [items])

  async function act(item: ScheduleItem, what: 'cancel' | 'retry', option?: 'private' | 'delete') {
    if (what === 'retry' && item.retry_confirm) {
      const where = item.platform === 'youtube' ? 'YouTube Studio' : 'Zernio and Instagram'
      if (!window.confirm(`Check ${where} first: is this post really not there? Retry sends it again.`))
        return
    }
    if (what === 'cancel' && option === 'delete') {
      if (!window.confirm('Delete this video from your YouTube channel? This cannot be undone.')) return
    }
    setBusyKey(item.key)
    setNotice('')
    try {
      if (what === 'cancel') {
        const post = await publishingApi.cancel(item.id, option)
        setNotice(post.error || 'Cancelled.')
      } else if (item.source === 'gta') {
        await publishingApi.retryGta(item.id)
        setNotice('Back in GTA Mode’s line; its schedule picks it up.')
      } else {
        await publishingApi.retry(item.id)
        setNotice('Back in line; it goes out on the next pass.')
      }
      await reload()
    } catch (err) {
      setNotice(errorText(err))
    } finally {
      setBusyKey('')
    }
  }

  const problems = status
    ? (['youtube', 'instagram'] as Platform[])
        .filter((p) => status.ready[p])
        .map((p) => `${PLATFORM_NAME[p]}: ${status.ready[p]}`)
    : []

  return (
    <div className="rise space-y-10 pt-10">
      <header className="flex flex-wrap items-end gap-6">
        <div>
          <p className="eyebrow">Posting</p>
          <h1 className="font-display text-5xl text-ink-100">Schedule</h1>
          <p className="mt-2 max-w-2xl text-sm text-ink-400">
            Every post going out to YouTube and Instagram: clips you post, Autopilot and GTA Mode.
          </p>
        </div>
        <div className="ml-auto flex gap-3">
          <button type="button" className="btn btn-primary" onClick={() => setShowPost((v) => !v)}>
            {showPost ? 'Close' : 'Post a clip'}
          </button>
          <Link to="/autopilot" className="btn btn-ghost">
            Autopilot
          </Link>
          <Link to="/health" className="btn btn-ghost">
            Health
          </Link>
        </div>
      </header>

      {problems.length > 0 && (
        <div className="border-l-2 border-signal-bad pl-4 text-sm text-ink-200">
          {problems.map((p) => (
            <p key={p}>{p}</p>
          ))}
          <p className="mt-1 text-ink-500">
            Connections are shared with GTA Mode: set them up in its Posting tab.
          </p>
        </div>
      )}

      {showPost && <PostPicker onPosted={reload} />}

      <div className="grid grid-cols-3 gap-6 border-y border-ink-800 py-5">
        <Stat label="Upcoming" value={data?.counts.upcoming ?? 0} />
        <Stat label="Posted" value={data?.counts.posted ?? 0} />
        <Stat label="Failed" value={data?.counts.failed ?? 0} bad={!!data?.counts.failed} />
      </div>

      <div className="flex flex-wrap items-center gap-6">
        {(['upcoming', 'past'] as const).map((t) => (
          <button
            key={t}
            type="button"
            className={`text-sm font-medium ${tab === t ? 'text-sodium-500' : 'text-ink-400 hover:text-ink-200'}`}
            onClick={() => setTab(t)}
          >
            {t === 'upcoming' ? `Upcoming (${data?.upcoming.length ?? 0})` : `History (${data?.past.length ?? 0})`}
          </button>
        ))}
        <div className="ml-auto flex gap-3 text-xs">
          {(['all', 'youtube', 'instagram'] as const).map((p) => (
            <button
              key={p}
              type="button"
              className={only === p ? 'text-sodium-500' : 'text-ink-400 hover:text-ink-200'}
              onClick={() => setOnly(p)}
            >
              {p === 'all' ? 'All' : PLATFORM_NAME[p]}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="text-sm text-signal-bad">{error}</p>}
      {notice && (
        <p className="text-sm text-ink-200" role="status">
          {notice}
        </p>
      )}
      {data && items.length === 0 && (
        <p className="text-sm text-ink-500">
          {tab === 'upcoming' ? 'Nothing is scheduled.' : 'Nothing has been posted yet.'}
        </p>
      )}

      {days.map(([day, list]) => (
        <section key={day}>
          <h2 className="eyebrow border-b border-ink-800 pb-2">{day}</h2>
          <ul className="divide-y divide-ink-800">
            {list.map((item) => (
              <ScheduleRow key={item.key} item={item} busy={busyKey === item.key} onAct={act} />
            ))}
          </ul>
        </section>
      ))}
    </div>
  )
}

function Stat({ label, value, bad }: { label: string; value: number; bad?: boolean }) {
  return (
    <div>
      <p className={`numeric font-display text-4xl ${bad ? 'text-signal-bad' : 'text-ink-100'}`}>{value}</p>
      <p className="eyebrow mt-1">{label}</p>
    </div>
  )
}

function ScheduleRow({
  item,
  busy,
  onAct,
}: {
  item: ScheduleItem
  busy: boolean
  onAct: (item: ScheduleItem, what: 'cancel' | 'retry', option?: 'private' | 'delete') => void
}) {
  const time = item.when
    ? new Date(item.when).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : 'now'
  return (
    <li className="grid grid-cols-[4.5rem_1fr] gap-x-4 gap-y-1 py-3 md:grid-cols-[4.5rem_7rem_1fr_auto]">
      <span className="numeric text-sm text-ink-200">{time}</span>
      <span className="text-sm text-ink-300">
        {PLATFORM_NAME[item.platform]}
        <span className="block text-xs text-ink-500">{SOURCE_LABEL[item.source]}</span>
      </span>
      <div className="col-start-2 min-w-0 md:col-start-3">
        <p className="truncate text-sm text-ink-100">{item.title || 'Untitled clip'}</p>
        <p className="text-xs">
          <span className={statusTone(item.status)}>{STATUS_LABEL[item.status] ?? item.status}</span>
          {item.url && (
            <>
              {' · '}
              <a className="text-ink-300 underline hover:text-sodium-400" href={item.url} target="_blank" rel="noreferrer">
                open
              </a>
            </>
          )}
          {item.error && <span className="text-ink-400"> · {item.error}</span>}
        </p>
        {!item.can_cancel && item.cancel_note && item.upcoming && item.source !== 'gta' && (
          <p className="mt-1 text-xs text-ink-500">{item.cancel_note}</p>
        )}
      </div>
      <div className="col-start-2 flex flex-wrap gap-2 md:col-start-4 md:justify-end">
        {item.can_cancel && item.cancel_options.length === 0 && (
          <button type="button" className="btn btn-quiet" disabled={busy} onClick={() => onAct(item, 'cancel')}>
            Cancel
          </button>
        )}
        {item.can_cancel &&
          item.cancel_options.map((option) => (
            <button
              key={option}
              type="button"
              className="btn btn-quiet"
              disabled={busy}
              onClick={() => onAct(item, 'cancel', option)}
            >
              {option === 'private' ? 'Unschedule (keep private)' : 'Delete from YouTube'}
            </button>
          ))}
        {item.can_retry && (
          <button type="button" className="btn btn-quiet" disabled={busy} onClick={() => onAct(item, 'retry')}>
            Retry
          </button>
        )}
      </div>
    </li>
  )
}

/** Recently exported clips, each with its Post panel. */
function PostPicker({ onPosted }: { onPosted: () => void }) {
  const [clips, setClips] = useState<PostableClip[] | null>(null)
  const [open, setOpen] = useState('')
  const [error, setError] = useState('')

  const load = useCallback(() => {
    publishingApi
      .clips()
      .then(setClips)
      .catch((err) => setError(errorText(err)))
  }, [])

  useEffect(load, [load])

  return (
    <section className="space-y-3">
      <h2 className="eyebrow border-b border-ink-800 pb-2">Exported clips</h2>
      {error && <p className="text-sm text-signal-bad">{error}</p>}
      {clips && clips.length === 0 && (
        <p className="text-sm text-ink-500">No exported clips yet. Clip a video first.</p>
      )}
      <ul className="divide-y divide-ink-800">
        {(clips ?? []).map((clip) => (
          <li key={clip.clip_id} className="py-3">
            <button
              type="button"
              className="flex w-full items-baseline gap-4 text-left"
              onClick={() => setOpen(open === clip.clip_id ? '' : clip.clip_id)}
            >
              <span className="text-sm text-ink-100">{clip.title || 'Untitled clip'}</span>
              <span className="truncate text-xs text-ink-500">{clip.source_title}</span>
              <span className="ml-auto text-xs text-ink-400">
                {(['youtube', 'instagram'] as Platform[])
                  .filter((p) => clip.active[p])
                  .map((p) => `${PLATFORM_NAME[p]}: ${STATUS_LABEL[clip.active[p]!.status]}`)
                  .join(' · ')}
              </span>
            </button>
            {open === clip.clip_id && (
              <div className="mt-3">
                <PostClipPanel
                  clip={clip}
                  onPosted={() => {
                    load()
                    onPosted()
                  }}
                />
              </div>
            )}
          </li>
        ))}
      </ul>
    </section>
  )
}

