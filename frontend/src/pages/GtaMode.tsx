/**
 * GTA Mode — strict clipping of famous face-cam streamers.
 *
 * Four views under one header: the stream queue with every finished clip
 * (copy-ready caption + hashtags), GTA Live ("Clip this live stream"), the
 * feedback loop ("Look at what's working") and GTA Mode's own settings. The
 * view lives in the URL (?tab=...). Lists poll every few seconds while open.
 */

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { ApiError } from '../api'
import {
  ACTIVE,
  LIVE_RUNNING,
  MOMENT_LABEL,
  STATUS_LABEL,
  clock,
  GTA_CAPTION_STYLES,
  GTA_MOTION_LEVELS,
  gtaApi,
  parseClock,
  type FeedbackReport,
  type GroupStat,
  type GtaClip,
  type GtaSettings,
  type GtaStream,
  MAX_UPLOAD_ATTEMPTS,
  PLATFORMS,
  PLATFORM_NAME,
  PLATFORM_SHORT,
  slotLabel,
  type UploadStatus,
} from '../gta/api'
import { YouTubeTab } from '../gta/YouTubeTab'
import {
  Badge,
  Button,
  Callout,
  CaptionsIcon,
  Card,
  CheckIcon,
  ChevronDownIcon,
  CopyIcon,
  EmptyState,
  Field,
  FilmIcon,
  FolderIcon,
  Input,
  KeyIcon,
  PageHeader,
  PlayIcon,
  PlusIcon,
  ProgressBar,
  RadioIcon,
  RefreshIcon,
  SectionTitle,
  Select,
  SendIcon,
  SettingsIcon,
  Skeleton,
  SkeletonCard,
  StampIcon,
  Tabs,
  Textarea,
  Toggle,
  TrashIcon,
  TrendingIcon,
  cx,
  tabPanelProps,
  useToast,
  type TabItem,
  type Tone,
} from '../ui'

type Tab = 'queue' | 'live' | 'youtube' | 'working' | 'settings'

const TABS: readonly TabItem<Tab>[] = [
  { key: 'queue', label: 'Streams & clips', icon: FilmIcon },
  { key: 'live', label: 'GTA Live', icon: RadioIcon },
  { key: 'youtube', label: 'Posting', icon: SendIcon },
  { key: 'working', label: "What's working", icon: TrendingIcon },
  { key: 'settings', label: 'Settings', icon: SettingsIcon },
]


const POLL_MS = 2500

function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.hint ? `${err.message}\n${err.hint}` : err.message
  return err instanceof Error ? err.message : String(err)
}

/** Poll ``load`` while mounted; returns [data, error, refresh]. */
function usePoll<T>(load: () => Promise<T>, ms = POLL_MS): [T | null, string, () => void] {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState('')
  const loadRef = useRef(load)
  loadRef.current = load
  const tick = useCallback(() => {
    loadRef
      .current()
      .then((value) => {
        setData(value)
        setError('')
      })
      .catch((err) => setError(errorText(err)))
  }, [])
  useEffect(() => {
    tick()
    const timer = window.setInterval(tick, ms)
    return () => window.clearInterval(timer)
  }, [tick, ms])
  return [data, error, tick]
}

async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    const area = document.createElement('textarea')
    area.value = text
    area.style.position = 'fixed'
    area.style.opacity = '0'
    document.body.appendChild(area)
    area.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(area)
    return ok
  }
}

function groupByStream(clips: GtaClip[]): Map<string, GtaClip[]> {
  const map = new Map<string, GtaClip[]>()
  for (const clip of clips) {
    const list = map.get(clip.stream_id)
    if (list) list.push(clip)
    else map.set(clip.stream_id, [clip])
  }
  return map
}

/** Run an action with busy/error state, then refresh. */
function useAction(onDone: () => void) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    setError('')
    try {
      await fn()
      onDone()
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }
  return { busy, error, run }
}

export function GtaMode() {
  const [params, setParams] = useSearchParams()
  // Back from Google's consent page (/gta?connected=1) -> the YouTube tab.
  const tab: Tab =
    params.get('connected') === '1'
      ? 'youtube'
      : (TABS.find((t) => t.key === params.get('tab'))?.key ?? 'queue')
  const [streams, streamsError, refreshStreams] = usePoll(gtaApi.streams)
  const [clips, clipsError, refreshClips] = usePoll(gtaApi.clips)
  const refresh = useCallback(() => {
    refreshStreams()
    refreshClips()
  }, [refreshStreams, refreshClips])
  const tabsId = useId()

  const show = (key: Tab) =>
    setParams(
      (p) => {
        const next = new URLSearchParams(p)
        if (key === 'queue') next.delete('tab')
        else next.set('tab', key)
        return next
      },
      { replace: true },
    )

  const vodStreams = (streams ?? []).filter((s) => !s.is_live)
  const liveStreams = (streams ?? []).filter((s) => s.is_live)

  return (
    <div>
      <PageHeader
        eyebrow={<Badge tone="accent">Separate mode</Badge>}
        title="GTA Mode"
        description="Strict clipping of face-cam streamers: finds the hype moments first, judges every pick twice, keeps the whole stream frame over a blurred fill, and checks each clip before it can be posted."
      >
        <Tabs label="GTA Mode sections" idBase={tabsId} items={TABS} value={tab} onChange={show} />
      </PageHeader>

      {(streamsError || clipsError) && (
        <Callout tone="danger" className="mb-6">
          <span className="whitespace-pre-line">{streamsError || clipsError}</span>
        </Callout>
      )}

      <div {...tabPanelProps(tabsId, tab)}>
        {tab === 'queue' && (
          <QueueTab streams={vodStreams} clips={clips ?? []} onChange={refresh} loaded={streams !== null} />
        )}
        {tab === 'live' && <LiveTab streams={liveStreams} clips={clips ?? []} onChange={refresh} />}
        {tab === 'youtube' && <YouTubeTab />}
        {tab === 'working' && <WorkingTab />}
        {tab === 'settings' && <SettingsTab />}
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Queue
// --------------------------------------------------------------------------

function QueueTab({
  streams,
  clips,
  onChange,
  loaded,
}: {
  streams: GtaStream[]
  clips: GtaClip[]
  onChange: () => void
  loaded: boolean
}) {
  const ids = useMemo(() => new Set(streams.map((s) => s.id)), [streams])
  const ready = clips.filter((c) => ids.has(c.stream_id) && c.status === 'ready')
  const byStream = useMemo(() => groupByStream(clips), [clips])

  return (
    <div className="grid gap-6">
      <AddStreams onAdded={onChange} />
      <Card padding="none" title="Queue" description="Streams being analysed, newest first.">
        <div className="border-t border-line">
          {!loaded ? (
            <div className="space-y-3 p-5" aria-label="Loading">
              <Skeleton className="h-4 w-1/2" />
              <Skeleton className="h-3 w-1/3" />
            </div>
          ) : streams.length === 0 ? (
            <p className="px-5 py-8 text-center text-sm text-fg-muted sm:px-6">
              No streams yet. Paste links above to start.
            </p>
          ) : (
            <ul className="divide-y divide-line">
              {streams.map((s) => (
                <StreamRow key={s.id} stream={s} clips={byStream.get(s.id) ?? []} onChange={onChange} />
              ))}
            </ul>
          )}
        </div>
      </Card>
      <ClipWall title="Ready to post" clips={ready} streams={streams} />
    </div>
  )
}

function AddStreams({ onAdded }: { onAdded: () => void }) {
  const [text, setText] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)

  const submit = async () => {
    const urls = text
      .split(/[\s,]+/)
      .map((u) => u.trim())
      .filter(Boolean)
    if (!urls.length) return
    const startS = parseClock(start)
    const endS = parseClock(end)
    if (Number.isNaN(startS) || Number.isNaN(endS)) {
      setMessage({ tone: 'bad', text: 'Use h:mm:ss for the range, e.g. 1:30:00.' })
      return
    }
    setBusy(true)
    setMessage(null)
    try {
      const result = await gtaApi.addStreams(urls, { start: startS, end: endS })
      const skipped = result.skipped.map((s) => `${s.url} — ${s.reason}`).join('\n')
      setMessage({
        tone: result.added.length ? 'good' : 'bad',
        text: `${result.added.length} stream(s) queued.${skipped ? `\nSkipped:\n${skipped}` : ''}`,
      })
      if (result.added.length) setText('')
      onAdded()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card title="Add streams" description="Paste stream or VOD links — one per line. YouTube, Twitch and Kick work.">
      <Textarea
        aria-label="Stream links"
        className="min-h-24 font-mono"
        placeholder={'https://www.twitch.tv/videos/…\nhttps://www.youtube.com/watch?v=…'}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="mt-4 flex flex-wrap items-end gap-4">
        <Field label="Only from" optional className="w-36">
          <Input placeholder="0:00:00" value={start} onChange={(e) => setStart(e.target.value)} />
        </Field>
        <Field label="Until" optional className="w-36">
          <Input placeholder="end" value={end} onChange={(e) => setEnd(e.target.value)} />
        </Field>
        <Button variant="primary" icon={PlusIcon} loading={busy} disabled={!text.trim()} onClick={submit}>
          {busy ? 'Queuing…' : 'Queue streams'}
        </Button>
      </div>
      {message && (
        <Callout tone={message.tone === 'good' ? 'success' : 'danger'} className="mt-4" onDismiss={() => setMessage(null)}>
          <span className="whitespace-pre-line">{message.text}</span>
        </Callout>
      )}
    </Card>
  )
}

function StreamRow({
  stream,
  clips,
  onChange,
}: {
  stream: GtaStream
  clips: GtaClip[]
  onChange: () => void
}) {
  const [open, setOpen] = useState(false)
  const { busy, error, run } = useAction(onChange)
  const active = ACTIVE.has(stream.status)
  const counts = stream.clip_counts
  const signals = stream.signals
  const notes = Array.isArray(signals.notes) ? (signals.notes as string[]) : []
  const working = (counts.approved ?? 0) + (counts.downloading ?? 0) + (counts.rendering ?? 0)
  const tone: Tone =
    stream.status === 'failed' ? 'danger' : stream.status === 'done' ? 'success' : stream.status === 'cancelled' ? 'neutral' : 'accent'

  return (
    <li className="px-5 py-4 sm:px-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-fg">
            {stream.channel || stream.platform || 'Stream'}
            <span className="ml-2 font-normal text-fg-muted">{stream.title || stream.url}</span>
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-fg-muted">
            <Badge tone={tone} dot pulse={active && stream.status !== 'analyzed'}>
              {STATUS_LABEL[stream.status] ?? stream.status}
            </Badge>
            {stream.detail && <span>{stream.detail}</span>}
            {stream.duration_s > 0 && <span>· {clock(stream.duration_s)} long</span>}
            {(stream.range_start_s != null || stream.range_end_s != null) && (
              <span>
                · range {clock(stream.range_start_s ?? 0)}–
                {stream.range_end_s != null ? clock(stream.range_end_s) : 'end'}
              </span>
            )}
          </div>
          {active && stream.status !== 'analyzed' && (
            <ProgressBar value={stream.progress} className="mt-2.5 max-w-md" label="Stream progress" />
          )}
          <p className="mt-2 text-xs text-fg-muted">
            {counts.ready ?? 0} ready · {working} in progress · {counts.rejected ?? 0} rejected
            {counts.qa_failed ? ` · ${counts.qa_failed} failed QA` : ''}
            {typeof signals.hype_markers === 'number' && ` · ${signals.hype_markers} hype markers`}
            {typeof signals.chat_messages === 'number' && ` · ${signals.chat_messages} chat msgs`}
            {typeof signals.comments === 'number' && signals.comments > 0 && ` · ${signals.comments} comments read`}
            {typeof signals.caption_parts === 'string' && ` · YouTube captions for ${signals.caption_parts} parts`}
          </p>
          {notes.length > 0 && <p className="mt-1 text-xs text-fg-muted">Signals: {notes.join('; ')}</p>}
          {stream.error && <p className="mt-2 whitespace-pre-line text-xs text-danger">{stream.error}</p>}
          {error && <p className="mt-2 whitespace-pre-line text-xs text-danger">{error}</p>}
        </div>
        <div className="flex shrink-0 flex-wrap gap-2">
          {clips.length > 0 && (
            <Button size="sm" onClick={() => setOpen((v) => !v)} iconRight={ChevronDownIcon} aria-expanded={open}>
              {open ? 'Hide moments' : `Moments (${clips.length})`}
            </Button>
          )}
          {active && (
            <Button size="sm" disabled={busy} onClick={() => run(() => gtaApi.cancel(stream.id))}>
              Cancel
            </Button>
          )}
          {(stream.status === 'failed' || stream.status === 'cancelled') && (
            <Button size="sm" variant="primary" icon={RefreshIcon} disabled={busy} onClick={() => run(() => gtaApi.retry(stream.id))}>
              Retry
            </Button>
          )}
          {!active && (
            <Button size="sm" variant="ghost" icon={TrashIcon} disabled={busy} onClick={() => run(() => gtaApi.remove(stream.id))}>
              Remove
            </Button>
          )}
        </div>
      </div>
      {open && <Moments clips={clips} />}
    </li>
  )
}

function Moments({ clips }: { clips: GtaClip[] }) {
  const sorted = [...clips].sort((a, b) => a.start_s - b.start_s)
  const tone = (c: GtaClip) =>
    c.status === 'ready' ? 'text-success' : c.status === 'rejected' ? 'text-fg-subtle' : 'text-accent-text'
  return (
    <ul className="mt-4 grid gap-1.5 rounded-xl bg-subtle p-3">
      {sorted.map((c) => (
        <li key={c.id} className="text-xs leading-relaxed">
          <span className="font-mono text-fg-muted">
            {clock(c.start_s)}–{clock(c.end_s)}
          </span>{' '}
          <span className={cx('font-medium', tone(c))}>{c.status}</span>{' '}
          <span className="text-fg">{c.hook_title || c.hook_quote}</span>
          {c.hype_score > 0 && <span className="text-fg-muted"> · hype {Math.round(c.hype_score)}</span>}
          {c.reject_reason && <span className="text-fg-muted"> — {c.reject_reason}</span>}
          {c.error && c.status !== 'ready' && <span className="text-danger"> — {c.error}</span>}
        </li>
      ))}
    </ul>
  )
}

// --------------------------------------------------------------------------
// Clip wall
// --------------------------------------------------------------------------

function ClipWall({ title, clips, streams }: { title: string; clips: GtaClip[]; streams: GtaStream[] }) {
  const names = useMemo(() => new Map(streams.map((s) => [s.id, s.channel || s.platform])), [streams])
  const sorted = [...clips].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99))
  return (
    <section>
      <SectionTitle
        title={`${title} (${clips.length})`}
        description="Each clip comes with a copy-ready caption and hashtags."
      />
      {clips.length === 0 ? (
        <EmptyState compact icon={FilmIcon} title="No finished clips yet" description="Finished clips appear here with their caption and hashtags." />
      ) : (
        <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-3">
          {sorted.map((c) => (
            <ClipCard key={c.id} clip={c} streamer={names.get(c.stream_id) ?? ''} />
          ))}
        </div>
      )}
    </section>
  )
}

const POSTING_WORD: Record<UploadStatus, string> = {
  uploading: 'uploading…',
  scheduled: 'scheduled',
  checking: 'checking',
  posted: 'live',
  failed: 'failed',
  interrupted: 'interrupted',
  skipped: 'excluded',
}

const POSTING_TONE: Record<UploadStatus, string> = {
  uploading: 'text-accent-text',
  scheduled: 'text-success',
  checking: 'text-accent-text',
  posted: 'text-success',
  failed: 'text-danger',
  interrupted: 'text-danger',
  skipped: 'text-fg-subtle',
}

/** Where this clip stands in the posting schedule (per platform), with Don't post / Allow. */
function PostingRow({ clip }: { clip: GtaClip }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const rows = PLATFORMS.flatMap((p) => {
    const upload = clip.uploads?.[p]
    return upload ? [{ platform: p, upload }] : []
  })
  const toggle = async (skip: boolean) => {
    setBusy(true)
    setError('')
    try {
      await gtaApi.setClipPosting(clip.id, skip)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }
  const skipped = rows.length > 0 && rows.every((r) => r.upload.status === 'skipped')
  const anyUploading = rows.some((r) => r.upload.status === 'uploading')
  // Nothing left that "Don't post" could still stop (every platform already sent or excluded).
  const allSettled =
    rows.length === PLATFORMS.length &&
    rows.every((r) => ['scheduled', 'posted', 'skipped', 'checking'].includes(r.upload.status))

  let text: React.ReactNode
  if (rows.length === 0) {
    text = clip.qa.passed ? 'Waiting for a posting slot' : 'Not posted (QA failed)'
  } else if (skipped) {
    text = 'Excluded from posting'
  } else {
    text = rows.map(({ platform, upload }, i) => (
      <span key={platform}>
        {i > 0 && ' · '}
        <span title={upload.error ?? undefined}>
          {PLATFORM_SHORT[platform]}:{' '}
          <span className={cx('font-medium', POSTING_TONE[upload.status])}>
            {POSTING_WORD[upload.status]}
            {upload.status === 'failed' && ` (${upload.attempts}/${MAX_UPLOAD_ATTEMPTS})`}
          </span>
          {(upload.status === 'scheduled' || upload.status === 'posted') && upload.slot_at && (
            <> {slotLabel(upload.slot_at)}</>
          )}
          {upload.url && (
            <>
              {' '}
              <a href={upload.url} target="_blank" rel="noreferrer" className="link">
                open ↗
              </a>
            </>
          )}
        </span>
      </span>
    ))
  }

  let action: React.ReactNode = null
  if (skipped) {
    action = (
      <Button size="sm" variant="ghost" disabled={busy} onClick={() => toggle(false)}>
        Allow posting
      </Button>
    )
  } else if (clip.qa.passed && !allSettled) {
    action = (
      <Button
        size="sm"
        variant="ghost"
        disabled={busy || anyUploading}
        title={anyUploading ? 'Uploading right now — wait for it to finish' : undefined}
        onClick={() => toggle(true)}
      >
        Don't post
      </Button>
    )
  }
  const failure = rows.find((r) => r.upload.status === 'failed' && r.upload.error)
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3 text-xs text-fg-muted">
      <span className="min-w-0">{text}</span>
      {action}
      {failure && (
        <span className="w-full text-danger">
          {PLATFORM_NAME[failure.platform]}: {failure.upload.error}
        </span>
      )}
      {error && <span className="w-full text-danger">{error}</span>}
    </div>
  )
}

function QaChip({ label, ok }: { label: string; ok: boolean | undefined }) {
  if (ok === undefined) return null
  return (
    <Badge tone={ok ? 'success' : 'danger'} title={ok ? 'Passed' : 'Failed'}>
      {ok ? '✓' : '✕'} {label}
    </Badge>
  )
}

function ClipCard({ clip, streamer }: { clip: GtaClip; streamer: string }) {
  const [copied, setCopied] = useState(false)
  const toast = useToast()
  const copy = async () => {
    if (await copyText(clip.post_text)) {
      setCopied(true)
      toast.success('Caption copied', 'Paste it with the clip when you post.')
      window.setTimeout(() => setCopied(false), 1800)
    }
  }
  const qa = clip.qa
  return (
    <article className="card flex flex-col gap-3 p-3">
      {clip.has_video ? (
        <video
          className="aspect-[9/16] w-full rounded-lg bg-media"
          src={gtaApi.videoUrl(clip.id)}
          controls
          preload="metadata"
        />
      ) : (
        <div className="flex aspect-[9/16] w-full items-center justify-center rounded-lg bg-subtle text-xs text-fg-muted">
          Video file missing
        </div>
      )}
      <div className="px-1">
        <p className="text-sm font-semibold text-fg">{clip.hook_title}</p>
        <p className="mt-0.5 text-xs text-fg-muted">
          {streamer} · {MOMENT_LABEL[clip.moment_type] ?? clip.moment_type} · score {clip.score}
          {qa.duration_s ? ` · ${qa.duration_s.toFixed(0)} s` : ''}
          {qa.turnaround_s !== undefined && ` · live turnaround ${Math.round(qa.turnaround_s)} s`}
        </p>
      </div>
      <div className="flex flex-wrap gap-1.5 px-1">
        <QaChip label="< 60 s" ok={qa.length?.ok} />
        <QaChip label="loudness" ok={qa.loudness?.ok} />
        <QaChip label="no black" ok={qa.black_frames?.ok} />
        <QaChip label="face-cam" ok={qa.facecam?.ok} />
        <QaChip label="1080x1920 30fps" ok={qa.format?.ok} />
        <QaChip label="own edit" ok={qa.own_value?.ok} />
        <QaChip label="logo" ok={qa.watermark?.ok} />
      </div>
      {qa.notes && qa.notes.length > 0 && <p className="px-1 text-xs text-fg-muted">{qa.notes.join('; ')}</p>}
      <Textarea
        readOnly
        className="min-h-24 text-xs"
        value={clip.post_text}
        onFocus={(e) => e.currentTarget.select()}
        aria-label="Caption and hashtags"
      />
      <Button variant="primary" icon={copied ? CheckIcon : CopyIcon} onClick={copy}>
        {copied ? 'Copied' : 'Copy caption + hashtags'}
      </Button>
      {clip.status === 'ready' && <PostingRow clip={clip} />}
      <p className="truncate px-1 font-mono text-[10px] text-fg-subtle" title={clip.path}>
        {clip.path}
      </p>
    </article>
  )
}

// --------------------------------------------------------------------------
// GTA Live
// --------------------------------------------------------------------------

function LiveTab({
  streams,
  clips,
  onChange,
}: {
  streams: GtaStream[]
  clips: GtaClip[]
  onChange: () => void
}) {
  const [url, setUrl] = useState('')
  const { busy, error, run } = useAction(onChange)
  const ids = useMemo(() => new Set(streams.map((s) => s.id)), [streams])
  const ready = clips.filter((c) => ids.has(c.stream_id) && c.status === 'ready')
  const byStream = useMemo(() => groupByStream(clips), [clips])

  const start = () =>
    run(async () => {
      await gtaApi.startLive(url.trim())
      setUrl('')
    })

  return (
    <div className="grid gap-6">
      <Card
        title="Clip a live stream"
        icon={RadioIcon}
        description="Listens every 20 seconds. When the chat explodes or the streamer loses it, the last 60 seconds are judged with the same strict rules, and a “yes” is rendered within about 3 minutes. Twitch also uses live chat; YouTube and Kick lives trigger on the streamer’s audio."
      >
        <div className="flex flex-col gap-3 sm:flex-row">
          <Input
            aria-label="Live stream link"
            className="sm:max-w-xl"
            placeholder="https://www.twitch.tv/streamer"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <Button variant="primary" icon={PlayIcon} disabled={busy || !url.trim()} onClick={start}>
            Start watching
          </Button>
        </div>
        {error && (
          <Callout tone="danger" className="mt-3">
            <span className="whitespace-pre-line">{error}</span>
          </Callout>
        )}
      </Card>
      <Card padding="none" title="Live watches">
        <div className="border-t border-line">
          {streams.length === 0 ? (
            <p className="px-5 py-8 text-center text-sm text-fg-muted sm:px-6">Nothing is being watched.</p>
          ) : (
            <ul className="divide-y divide-line">
              {streams.map((s) => {
                const running = LIVE_RUNNING.has(s.status)
                const mine = byStream.get(s.id) ?? []
                const readyCount = mine.filter((c) => c.status === 'ready').length
                const rejected = mine.filter((c) => c.status === 'rejected').length
                return (
                  <li key={s.id} className="flex flex-wrap items-start justify-between gap-4 px-5 py-4 sm:px-6">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold text-fg">{s.channel || s.url}</p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-fg-muted">
                        <Badge tone={running ? 'success' : 'neutral'} dot pulse={running}>
                          {STATUS_LABEL[s.status] ?? s.status}
                        </Badge>
                        {s.detail && <span>{s.detail}</span>}
                        <span>
                          · {readyCount} ready · {rejected} rejected
                        </span>
                      </div>
                      {s.error && <p className="mt-1.5 whitespace-pre-line text-xs text-danger">{s.error}</p>}
                    </div>
                    <div className="flex gap-2">
                      {running ? (
                        <Button size="sm" disabled={busy} onClick={() => run(() => gtaApi.stopLive(s.id))}>
                          Stop
                        </Button>
                      ) : (
                        <>
                          <Button size="sm" variant="primary" disabled={busy} onClick={() => run(() => gtaApi.retry(s.id))}>
                            Watch again
                          </Button>
                          <Button size="sm" variant="ghost" icon={TrashIcon} disabled={busy} onClick={() => run(() => gtaApi.remove(s.id))}>
                            Remove
                          </Button>
                        </>
                      )}
                    </div>
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </Card>
      <ClipWall title="Live clips ready" clips={ready} streams={streams} />
    </div>
  )
}

// --------------------------------------------------------------------------
// What's working
// --------------------------------------------------------------------------

function StatTable({ title, rows }: { title: string; rows: Record<string, GroupStat> }) {
  const entries = Object.entries(rows)
  if (!entries.length) return null
  return (
    <Card padding="none" title={title}>
      <div className="overflow-x-auto border-t border-line">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-fg-muted">
              <th className="px-5 py-2 font-medium">Group</th>
              <th className="px-3 py-2 text-right font-medium">Shorts</th>
              <th className="px-3 py-2 text-right font-medium">Avg views</th>
              <th className="px-5 py-2 text-right font-medium">vs channel</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {entries.map(([name, stat]) => (
              <tr key={name}>
                <td className="px-5 py-2 text-fg">{MOMENT_LABEL[name] ?? name}</td>
                <td className="numeric px-3 py-2 text-right text-fg-muted">{stat.count}</td>
                <td className="numeric px-3 py-2 text-right text-fg-muted">{stat.average_views.toLocaleString()}</td>
                <td
                  className={cx(
                    'numeric px-5 py-2 text-right font-semibold',
                    stat.vs_channel >= 1 ? 'text-success' : 'text-danger',
                  )}
                >
                  {stat.vs_channel.toFixed(2)}×
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function WorkingTab() {
  const [url, setUrl] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [report, setReport] = useState<FeedbackReport | null>(null)

  useEffect(() => {
    gtaApi
      .feedback()
      .then((latest) => {
        if (latest) {
          setReport(latest.report)
          setUrl(latest.channel_url)
        }
      })
      .catch(() => undefined)
  }, [])

  const runReport = async () => {
    setBusy(true)
    setError('')
    try {
      setReport(await gtaApi.runFeedback(url.trim()))
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid gap-6">
      <Card
        title="See what's working"
        icon={TrendingIcon}
        description="Reads the view counts of your channel's posted Shorts, finds which moment types and streamers beat the channel average, and leans future picks toward them (only groups with 3+ Shorts count)."
      >
        <div className="flex flex-col gap-3 sm:flex-row">
          <Input
            aria-label="Your YouTube channel link"
            className="sm:max-w-xl"
            placeholder="https://www.youtube.com/@yourchannel"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <Button variant="primary" loading={busy} disabled={!url.trim()} onClick={runReport}>
            {busy ? 'Reading views…' : 'Check my channel'}
          </Button>
        </div>
        {error && (
          <Callout tone="danger" className="mt-3">
            <span className="whitespace-pre-line">{error}</span>
          </Callout>
        )}
      </Card>
      {report && (
        <>
          <div className="grid gap-4 sm:grid-cols-3">
            <Stat label="Shorts read" value={report.count.toLocaleString()} />
            <Stat label="Channel average" value={`${report.channel_average.toLocaleString()} views`} />
            <Stat label="Matched to GTA clips" value={report.matched.toLocaleString()} />
          </div>
          {report.bias.summary && (
            <Callout tone="info" title="What AutoClip will lean toward">
              <span className="whitespace-pre-line">{report.bias.summary}</span>
            </Callout>
          )}
          <div className="grid gap-6 lg:grid-cols-2">
            <StatTable title="Moment types" rows={report.moment_types} />
            <StatTable title="Streamers" rows={report.streamers} />
          </div>
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card p-4">
      <p className="text-xs font-medium text-fg-muted">{label}</p>
      <p className="numeric mt-1 text-xl font-bold text-fg">{value}</p>
    </div>
  )
}

// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------

type NumberKey =
  | 'chunk_minutes'
  | 'min_clip_s'
  | 'target_clip_s'
  | 'max_clip_s'
  | 'max_clips_per_stream'
  | 'hook_title_s'
  | 'dead_air_min_s'
  | 'loudness_lufs'
  | 'face_min_ratio'
  | 'network_workers'
  | 'punch_in_zoom'
  | 'watermark_width_px'
  | 'watermark_opacity'

const NUMBER_FIELDS: { key: NumberKey; label: string; step: number }[] = [
  { key: 'chunk_minutes', label: 'Transcript chunk (minutes)', step: 1 },
  { key: 'min_clip_s', label: 'Shortest clip (s)', step: 1 },
  { key: 'target_clip_s', label: 'Target length (s)', step: 1 },
  { key: 'max_clip_s', label: 'Longest clip (s, under 60)', step: 1 },
  { key: 'max_clips_per_stream', label: 'Most clips per stream', step: 1 },
  { key: 'hook_title_s', label: 'Hook title on screen (s)', step: 0.5 },
  { key: 'dead_air_min_s', label: 'Cut silences longer than (s)', step: 0.1 },
  { key: 'loudness_lufs', label: 'Loudness target (LUFS)', step: 0.5 },
  { key: 'face_min_ratio', label: 'Face visible in at least (0–1)', step: 0.05 },
  { key: 'network_workers', label: 'Parallel downloads / judges', step: 1 },
  { key: 'punch_in_zoom', label: 'Punch-in zoom (×)', step: 0.05 },
  { key: 'watermark_width_px', label: 'Logo width (px)', step: 10 },
  { key: 'watermark_opacity', label: 'Logo opacity (0.1–1)', step: 0.05 },
]

/** Which number fields go in which settings card. */
const NUMBER_GROUPS: { title: string; description: string; keys: NumberKey[] }[] = [
  {
    title: 'Clip length & picking',
    description: 'How long clips are and how many each stream can give.',
    keys: ['min_clip_s', 'target_clip_s', 'max_clip_s', 'max_clips_per_stream', 'chunk_minutes', 'face_min_ratio'],
  },
  {
    title: 'Editing',
    description: 'Hook title, silence cuts, zoom and loudness.',
    keys: ['hook_title_s', 'dead_air_min_s', 'punch_in_zoom', 'loudness_lufs'],
  },
  { title: 'Speed', description: 'How much runs at once.', keys: ['network_workers'] },
]

function SettingsTab() {
  const [settings, setSettings] = useState<GtaSettings | null>(null)
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [message, setMessage] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)
  const [clientId, setClientId] = useState('')
  const [secret, setSecret] = useState('')
  const [logoVersion, setLogoVersion] = useState('0')
  const toast = useToast()

  const load = useCallback(() => {
    gtaApi
      .settings()
      .then((s) => {
        setSettings(s)
        setLogoVersion(String(Date.now()))
        const next: Record<string, string> = {
          output_dir: s.output_dir,
          caption_style: s.caption_style,
          watermark_path: s.watermark_path,
          watermark_text: s.watermark_text,
          watermark_font: s.watermark_font,
          watermark_corner: s.watermark_corner,
          punch_in: String(s.punch_in),
          reaction_text: String(s.reaction_text),
          keyword_highlight: String(s.keyword_highlight),
          motion: s.motion,
          comment_signal: String(s.comment_signal),
          youtube_captions: String(s.youtube_captions),
        }
        for (const f of NUMBER_FIELDS) next[f.key] = String(s[f.key])
        next.listen_interval_s = String(s.live.listen_interval_s)
        next.context_s = String(s.live.context_s)
        setDraft(next)
      })
      .catch((err) => setMessage({ tone: 'bad', text: errorText(err) }))
  }, [])
  useEffect(load, [load])

  if (!settings) {
    return message ? (
      <Callout tone="danger">
        <span className="whitespace-pre-line">{message.text}</span>
      </Callout>
    ) : (
      <div className="grid gap-6">
        <SkeletonCard lines={4} />
        <SkeletonCard lines={3} />
      </div>
    )
  }

  const save = async () => {
    const live = {
      ...settings.live,
      listen_interval_s: Number(draft.listen_interval_s),
      context_s: Number(draft.context_s),
    }
    if (!Number.isFinite(live.listen_interval_s) || !Number.isFinite(live.context_s)) {
      setMessage({ tone: 'bad', text: 'GTA Live timings must be numbers.' })
      return
    }
    const patch: Record<string, unknown> = {
      output_dir: draft.output_dir ?? '',
      watermark_path: (draft.watermark_path ?? '').trim().replace(/^"|"$/g, ''),
      watermark_text: (draft.watermark_text ?? '').trim(),
      watermark_font: (draft.watermark_font ?? '').trim().replace(/^"|"$/g, ''),
      watermark_corner: draft.watermark_corner || 'top_right',
      punch_in: draft.punch_in === 'true',
      reaction_text: draft.reaction_text === 'true',
      keyword_highlight: draft.keyword_highlight === 'true',
      motion: draft.motion || settings.motion,
      comment_signal: draft.comment_signal === 'true',
      youtube_captions: draft.youtube_captions === 'true',
      caption_style: draft.caption_style || settings.caption_style,
      live,
    }
    for (const f of NUMBER_FIELDS) {
      const value = Number(draft[f.key])
      if (draft[f.key] === '' || !Number.isFinite(value)) {
        setMessage({ tone: 'bad', text: `${f.label}: not a number.` })
        return
      }
      patch[f.key] = value
    }
    try {
      await gtaApi.saveSettings(patch as Partial<GtaSettings>)
      setMessage(null)
      toast.success('GTA settings saved')
      load()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    }
  }

  const saveTwitch = async (clear = false) => {
    try {
      const result = await gtaApi.saveTwitch(clear ? '' : clientId.trim(), clear ? '' : secret.trim())
      toast.success(result.twitch_credentials_set ? 'Twitch credentials saved' : 'Twitch credentials removed')
      setClientId('')
      setSecret('')
      load()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    }
  }

  const set = (key: string, value: string) => setDraft((d) => ({ ...d, [key]: value }))
  const field = (key: string, label: string, props: Record<string, unknown> = {}, hint?: string) => (
    <Field key={key} label={label} hint={hint}>
      <Input value={draft[key] ?? ''} onChange={(e) => set(key, e.target.value)} {...props} />
    </Field>
  )
  const numberField = (f: (typeof NUMBER_FIELDS)[number]) => field(f.key, f.label, { type: 'number', step: f.step })
  const check = (key: string, label: string, hint?: string) => (
    <Toggle key={key} label={label} hint={hint} checked={draft[key] === 'true'} onChange={(on) => set(key, String(on))} />
  )

  return (
    <div className="grid max-w-4xl gap-6">
      {NUMBER_GROUPS.map((group) => (
        <Card key={group.title} title={group.title} description={group.description}>
          <div className="grid gap-5 sm:grid-cols-2">
            {NUMBER_FIELDS.filter((f) => group.keys.includes(f.key)).map(numberField)}
          </div>
        </Card>
      ))}

      <Card title="Captions & motion" icon={CaptionsIcon}>
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Caption style" hint={GTA_CAPTION_STYLES.find((option) => option.value === draft.caption_style)?.hint}>
            <Select
              value={draft.caption_style ?? ''}
              onChange={(e) => set('caption_style', e.target.value)}
              options={GTA_CAPTION_STYLES.map((o) => ({ value: o.value, label: o.label }))}
            />
          </Field>
          <Field label="Motion" hint="Zooms on the stream frame only.">
            <Select
              value={draft.motion ?? 'subtle'}
              onChange={(e) => set('motion', e.target.value)}
              options={GTA_MOTION_LEVELS.map((o) => ({ value: o.value, label: o.label }))}
            />
          </Field>
          <div className="grid gap-4 sm:col-span-2">
            {check('punch_in', 'Punch-in zoom on the key moment')}
            {check('reaction_text', 'Reaction text at the key moment')}
            {check('keyword_highlight', 'Highlight 3–8 key words in the captions')}
          </div>
        </div>
      </Card>

      <Card title="Logo" icon={StampIcon} description="Every GTA clip carries your logo. Use a PNG, or text drawn in a font.">
        <div className="grid gap-5 sm:grid-cols-2">
          {field('watermark_text', 'Logo text', { maxLength: 40 }, 'Drawn when no PNG is set.')}
          {field('watermark_font', 'Logo font file', { placeholder: 'C:\\Windows\\Fonts\\your-font.otf' }, 'Empty = Lemon Milk Bold.')}
          <div className="sm:col-span-2">
            {field(
              'watermark_path',
              'Logo image (PNG)',
              { placeholder: 'C:\\Users\\you\\Pictures\\logo.png' },
              'Full path to a PNG. Wins over the text.',
            )}
            {!settings.watermark_path && !settings.watermark_text ? (
              <Callout tone="warn" className="mt-3">
                No logo set yet — clips are flagged until you add one.
              </Callout>
            ) : (
              <div className="mt-3 inline-flex items-center gap-3 rounded-xl border border-line bg-subtle p-3">
                <span className="text-xs text-fg-muted">Current logo</span>
                <img key={logoVersion} src={gtaApi.logoUrl(logoVersion)} alt="Current logo" className="h-8 w-auto" />
              </div>
            )}
          </div>
          <Field label="Logo corner">
            <Select
              value={draft.watermark_corner ?? 'top_right'}
              onChange={(e) => set('watermark_corner', e.target.value)}
              options={[
                { value: 'top_right', label: 'Top right' },
                { value: 'top_left', label: 'Top left' },
              ]}
            />
          </Field>
          <div className="grid gap-5 sm:col-span-2 sm:grid-cols-2">
            {NUMBER_FIELDS.filter((f) => f.key === 'watermark_width_px' || f.key === 'watermark_opacity').map(numberField)}
          </div>
        </div>
      </Card>

      <Card title="GTA Live" icon={RadioIcon} description="How often a live stream is checked, and how much of it is judged.">
        <div className="grid gap-5 sm:grid-cols-2">
          {field('listen_interval_s', 'Listen every (seconds)', { type: 'number', step: 1 })}
          {field('context_s', 'Judge the last (seconds)', { type: 'number', step: 5 })}
        </div>
      </Card>

      <Card title="YouTube VODs" description="Extra signals for YouTube videos.">
        <div className="grid gap-4">
          {check('comment_signal', 'Use timestamps viewers wrote in the comments in the Hype Curve')}
          {check(
            'youtube_captions',
            'Use YouTube’s own captions as the transcript when they look right',
            'Skips Whisper; automatic captions have no punctuation.',
          )}
        </div>
      </Card>

      <Card title="Where clips are saved" icon={FolderIcon}>
        {field('output_dir', 'Save finished clips to', {}, 'Empty = AutoClip exports/gta.')}
      </Card>

      <div className="sticky bottom-3 z-10 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-line bg-surface/95 px-4 py-3 shadow-[var(--shadow-pop)] backdrop-blur">
        <p className="min-w-0 text-sm text-fg-muted" aria-live="polite">
          {message?.tone === 'bad' ? (
            <span className="whitespace-pre-line text-danger">{message.text}</span>
          ) : (
            'Changes apply after you save.'
          )}
        </p>
        <Button variant="primary" icon={CheckIcon} onClick={save}>
          Save settings
        </Button>
      </div>

      <Card
        title="Twitch viewer clips"
        icon={KeyIcon}
        description="Optional. With a free Twitch developer app (dev.twitch.tv → Your Console → Register an application), the Hype Curve also uses the clips viewers made of a VOD."
        actions={
          <Badge tone={settings.twitch_credentials_set ? 'success' : 'neutral'} dot>
            {settings.twitch_credentials_set ? 'Connected' : 'Not set'}
          </Badge>
        }
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Client ID">
            <Input value={clientId} onChange={(e) => setClientId(e.target.value)} autoComplete="off" />
          </Field>
          <Field label="Client secret">
            <Input type="password" value={secret} onChange={(e) => setSecret(e.target.value)} autoComplete="off" />
          </Field>
        </div>
        <div className="mt-4 flex gap-2">
          <Button disabled={!clientId.trim() || !secret.trim()} onClick={() => saveTwitch()}>
            Save credentials
          </Button>
          {settings.twitch_credentials_set && (
            <Button variant="ghost" icon={TrashIcon} onClick={() => saveTwitch(true)}>
              Remove
            </Button>
          )}
        </div>
      </Card>
    </div>
  )
}
