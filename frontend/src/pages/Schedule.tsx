import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import { PostClipPanel } from '../components/PostClipPanel'
import {
  errorText,
  PLATFORM_NAME,
  publishingApi,
  STATUS_LABEL,
  type Platform,
  type PostableClip,
  type PublishingStatus,
  type ScheduleItem,
  type ScheduleOut,
} from '../publishing/api'
import { PlatformTag, PostStatusBadge } from '../publishing/ui'
import {
  ActivityIcon,
  Button,
  ButtonLink,
  CalendarIcon,
  Callout,
  Card,
  ChevronDownIcon,
  Dialog,
  EmptyState,
  ExternalIcon,
  PageHeader,
  RocketIcon,
  Segmented,
  SendIcon,
  Skeleton,
  Tabs,
  cx,
  tabPanelProps,
  useToast,
} from '../ui'

const SOURCE_LABEL: Record<ScheduleItem['source'], string> = {
  normal: 'Posted by you',
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

type Pending = { item: ScheduleItem; what: 'cancel' | 'retry'; option?: 'private' | 'delete' }

/** Every post going out (or gone out): clips you post, Autopilot, and GTA Mode. */
export function Schedule() {
  const [data, setData] = useState<ScheduleOut | null>(null)
  const [status, setStatus] = useState<PublishingStatus | null>(null)
  const [tab, setTab] = useState<'upcoming' | 'past'>('upcoming')
  const [only, setOnly] = useState<'all' | Platform>('all')
  const [error, setError] = useState('')
  const [busyKey, setBusyKey] = useState('')
  const [showPost, setShowPost] = useState(false)
  const [confirm, setConfirm] = useState<Pending | null>(null)
  const toast = useToast()
  const tabsId = 'schedule-tabs'

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

  /** Retries that might double-post, and YouTube deletes, ask first. */
  function request(item: ScheduleItem, what: 'cancel' | 'retry', option?: 'private' | 'delete') {
    if ((what === 'retry' && item.retry_confirm) || (what === 'cancel' && option === 'delete')) {
      setConfirm({ item, what, option })
      return
    }
    void act({ item, what, option })
  }

  async function act({ item, what, option }: Pending) {
    setConfirm(null)
    setBusyKey(item.key)
    try {
      if (what === 'cancel') {
        const post = await publishingApi.cancel(item.id, option)
        if (post.error) toast.error('Cancel did not go through', post.error)
        else toast.success('Post cancelled')
      } else if (item.source === 'gta') {
        await publishingApi.retryGta(item.id)
        toast.success('Back in line', 'GTA Mode’s schedule picks it up.')
      } else {
        await publishingApi.retry(item.id)
        toast.success('Back in line', 'It goes out on the next pass.')
      }
      await reload()
    } catch (err) {
      toast.error('That didn’t work', errorText(err))
    } finally {
      setBusyKey('')
    }
  }

  const problems = status
    ? (['youtube', 'instagram'] as Platform[]).filter((p) => status.ready[p]).map((p) => ({ p, text: status.ready[p] }))
    : []

  return (
    <div>
      <PageHeader
        title="Schedule"
        description="Every post going out to YouTube and Instagram — clips you post, Autopilot and GTA Mode — in one list."
        actions={
          <>
            <Button variant="primary" icon={SendIcon} onClick={() => setShowPost((v) => !v)} aria-expanded={showPost}>
              {showPost ? 'Close' : 'Post a clip'}
            </Button>
            <ButtonLink to="/autopilot" icon={RocketIcon}>
              Autopilot
            </ButtonLink>
            <ButtonLink to="/health" icon={ActivityIcon}>
              Health check
            </ButtonLink>
          </>
        }
      />

      {problems.length > 0 && (
        <Callout
          tone="warn"
          title="Posting isn't fully set up"
          className="mb-6"
          action={
            <ButtonLink to="/posting" size="sm">
              Open Posting & accounts
            </ButtonLink>
          }
        >
          {problems.map(({ p, text }) => (
            <span key={p} className="block">
              {PLATFORM_NAME[p]}: {text}
            </span>
          ))}
        </Callout>
      )}

      {showPost && <PostPicker onPosted={reload} />}

      <div className="mb-6 grid grid-cols-3 gap-3 sm:gap-4">
        <Stat label="Upcoming" value={data?.counts.upcoming} />
        <Stat label="Posted" value={data?.counts.posted} />
        <Stat label="Failed" value={data?.counts.failed} bad={!!data?.counts.failed} />
      </div>

      <Card padding="none">
        <div className="flex flex-col gap-3 px-5 pt-4 sm:flex-row sm:items-end sm:justify-between sm:px-6">
          <Tabs
            label="Posts"
            idBase={tabsId}
            className="border-b-0"
            items={[
              { key: 'upcoming', label: `Upcoming (${data?.upcoming.length ?? 0})` },
              { key: 'past', label: `History (${data?.past.length ?? 0})` },
            ]}
            value={tab}
            onChange={setTab}
          />
          <Segmented
            className="mb-2"
            label="Platform"
            size="sm"
            options={[
              { value: 'all', label: 'All' },
              { value: 'youtube', label: 'YouTube' },
              { value: 'instagram', label: 'Instagram' },
            ]}
            value={only}
            onChange={setOnly}
          />
        </div>
        <div className="border-t border-line" {...tabPanelProps(tabsId, tab)}>
          {error && (
            <div className="p-5">
              <Callout tone="danger">{error}</Callout>
            </div>
          )}
          {!data && !error && (
            <div className="space-y-3 p-5 sm:px-6" aria-label="Loading">
              <Skeleton className="h-4 w-1/3" />
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          )}
          {data && items.length === 0 && (
            <div className="p-5 sm:p-6">
              <EmptyState
                compact
                icon={CalendarIcon}
                title={tab === 'upcoming' ? 'Nothing is scheduled' : 'Nothing has been posted yet'}
                description={
                  tab === 'upcoming'
                    ? 'Post an exported clip, or let an Autopilot fill your posting times.'
                    : 'Posts show up here once they go out.'
                }
                action={
                  tab === 'upcoming' && (
                    <Button variant="primary" icon={SendIcon} onClick={() => setShowPost(true)}>
                      Post a clip
                    </Button>
                  )
                }
              />
            </div>
          )}
          {days.map(([day, list]) => (
            <section key={day}>
              <h2 className="bg-subtle px-5 py-2 text-xs font-semibold text-fg-muted sm:px-6">{day}</h2>
              <ul className="divide-y divide-line">
                {list.map((item) => (
                  <ScheduleRow key={item.key} item={item} busy={busyKey === item.key} onAct={request} />
                ))}
              </ul>
            </section>
          ))}
        </div>
      </Card>

      <Dialog
        open={confirm !== null}
        onClose={() => setConfirm(null)}
        size="sm"
        title={confirm?.what === 'retry' ? 'Send it again?' : 'Delete from YouTube?'}
        description={
          confirm?.what === 'retry'
            ? `First check ${confirm.item.platform === 'youtube' ? 'YouTube Studio' : 'Zernio and Instagram'}: is this post really not there? Retrying sends it again.`
            : 'This deletes the video from your YouTube channel. It cannot be undone.'
        }
        footer={
          <>
            <Button onClick={() => setConfirm(null)}>Keep it</Button>
            <Button
              variant={confirm?.what === 'retry' ? 'primary' : 'danger'}
              data-autofocus
              onClick={() => confirm && void act(confirm)}
            >
              {confirm?.what === 'retry' ? 'Retry' : 'Delete'}
            </Button>
          </>
        }
      />
    </div>
  )
}

function Stat({ label, value, bad }: { label: string; value: number | undefined; bad?: boolean }) {
  return (
    <div className="card p-4">
      <p className="text-xs font-medium text-fg-muted">{label}</p>
      {value === undefined ? (
        <Skeleton className="mt-2 h-6 w-10" />
      ) : (
        <p className={cx('numeric mt-1 text-2xl font-bold', bad ? 'text-danger' : 'text-fg')}>{value}</p>
      )}
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
    : 'Now'
  return (
    <li className="grid grid-cols-[3.5rem_minmax(0,1fr)] gap-x-4 gap-y-2 px-5 py-3 sm:px-6 md:grid-cols-[4rem_minmax(0,1fr)_auto] md:items-center">
      <span className="numeric pt-0.5 text-sm font-semibold text-fg">{time}</span>
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-fg">{item.title || 'Untitled clip'}</p>
        <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-muted">
          <PostStatusBadge status={item.status} />
          <PlatformTag platform={item.platform} />
          <span>{SOURCE_LABEL[item.source]}</span>
          {item.url && (
            <a className="inline-flex items-center gap-1 font-medium text-accent-text hover:underline" href={item.url} target="_blank" rel="noreferrer">
              Open <ExternalIcon size={12} />
            </a>
          )}
        </div>
        {item.error && <p className="mt-1 text-xs text-danger [overflow-wrap:anywhere]">{item.error}</p>}
        {!item.can_cancel && item.cancel_note && item.upcoming && item.source !== 'gta' && (
          <p className="mt-1 text-xs text-fg-muted">{item.cancel_note}</p>
        )}
      </div>
      <div className="col-start-2 flex flex-wrap gap-2 md:col-start-3 md:justify-end">
        {item.can_cancel && item.cancel_options.length === 0 && (
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => onAct(item, 'cancel')}>
            Cancel
          </Button>
        )}
        {item.can_cancel &&
          item.cancel_options.map((option) => (
            <Button key={option} size="sm" variant="ghost" disabled={busy} onClick={() => onAct(item, 'cancel', option)}>
              {option === 'private' ? 'Unschedule (keep private)' : 'Delete from YouTube'}
            </Button>
          ))}
        {item.can_retry && (
          <Button size="sm" disabled={busy} onClick={() => onAct(item, 'retry')}>
            Retry
          </Button>
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
    <Card
      padding="none"
      className="mb-6 overflow-hidden"
      title="Post a clip"
      description="Your recently exported clips. Open one to post it now or at a set time."
    >
      <div className="border-t border-line">
        {error && (
          <div className="p-5">
            <Callout tone="danger">{error}</Callout>
          </div>
        )}
        {!clips && !error && (
          <div className="space-y-3 p-5 sm:px-6">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        )}
        {clips && clips.length === 0 && (
          <p className="px-5 py-8 text-center text-sm text-fg-muted sm:px-6">
            No exported clips yet. Export a clip on its project's Review page, then post it here.{' '}
            <Link to="/projects" className="link">
              Open Projects
            </Link>
          </p>
        )}
        <ul className="divide-y divide-line">
          {(clips ?? []).map((clip) => {
            const isOpen = open === clip.clip_id
            const booked = (['youtube', 'instagram'] as Platform[]).filter((p) => clip.active[p])
            return (
              <li key={clip.clip_id}>
                <button
                  type="button"
                  aria-expanded={isOpen}
                  className="flex w-full items-center gap-3 px-5 py-3 text-left transition-colors hover:bg-subtle sm:px-6"
                  onClick={() => setOpen(isOpen ? '' : clip.clip_id)}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm font-medium text-fg">{clip.title || 'Untitled clip'}</span>
                    <span className="block truncate text-xs text-fg-muted">
                      {clip.source_title}
                      {booked.length > 0 &&
                        ` · ${booked.map((p) => `${PLATFORM_NAME[p]}: ${STATUS_LABEL[clip.active[p]!.status]}`).join(' · ')}`}
                    </span>
                  </span>
                  <ChevronDownIcon size={16} className={cx('text-fg-subtle transition-transform', isOpen && 'rotate-180')} />
                </button>
                {isOpen && (
                  <div className="border-t border-line bg-subtle/50 px-5 py-5 sm:px-6">
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
            )
          })}
        </ul>
      </div>
    </Card>
  )
}
