/**
 * GTA Mode — strict clipping of famous face-cam streamers.
 *
 * Four views under one header: the stream queue with every finished clip
 * (copy-ready caption + hashtags), GTA Live ("Clip this live stream"), the
 * feedback loop ("Look at what's working") and GTA Mode's own settings. The
 * view lives in the URL (?tab=...). Lists poll every few seconds while open.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { ApiError } from '../api'
import {
  ACTIVE,
  LIVE_RUNNING,
  MOMENT_LABEL,
  STATUS_LABEL,
  clock,
  gtaApi,
  parseClock,
  type FeedbackReport,
  type GroupStat,
  type GtaClip,
  type GtaSettings,
  type GtaStream,
} from '../gta/api'

type Tab = 'queue' | 'live' | 'working' | 'settings'

const TABS: readonly { key: Tab; label: string }[] = [
  { key: 'queue', label: 'Streams & clips' },
  { key: 'live', label: 'GTA Live' },
  { key: 'working', label: "What's working" },
  { key: 'settings', label: 'Settings' },
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
  const tab: Tab = TABS.find((t) => t.key === params.get('tab'))?.key ?? 'queue'
  const [streams, streamsError, refreshStreams] = usePoll(gtaApi.streams)
  const [clips, clipsError, refreshClips] = usePoll(gtaApi.clips)
  const refresh = useCallback(() => {
    refreshStreams()
    refreshClips()
  }, [refreshStreams, refreshClips])

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
    <div className="pt-10">
      <header className="flex flex-wrap items-end justify-between gap-6 border-b border-ink-800 pb-5">
        <div>
          <p className="eyebrow">Separate mode</p>
          <h1 className="font-display mt-1 text-4xl leading-none text-ink-100">
            GTA <span className="italic text-sodium-500">Mode</span>
          </h1>
          <p className="mt-2 max-w-2xl text-sm text-ink-400">
            Face-cam streamers only. Hype Curve first, 12-minute chunks, two harsh judging rounds, the
            whole stream frame uncropped over a blurred fill, QA on every clip.
          </p>
        </div>
        <nav className="flex gap-5" role="tablist" aria-label="GTA Mode views">
          {TABS.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              onClick={() => show(t.key)}
              className={`border-b-2 pb-1 text-sm font-medium transition-colors ${
                tab === t.key
                  ? 'border-sodium-500 text-sodium-500'
                  : 'border-transparent text-ink-400 hover:text-ink-200'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {(streamsError || clipsError) && (
        <p className="mt-6 whitespace-pre-line text-sm text-signal-bad">{streamsError || clipsError}</p>
      )}

      {tab === 'queue' && (
        <QueueTab
          streams={vodStreams}
          clips={clips ?? []}
          onChange={refresh}
          loaded={streams !== null}
        />
      )}
      {tab === 'live' && <LiveTab streams={liveStreams} clips={clips ?? []} onChange={refresh} />}
      {tab === 'working' && <WorkingTab />}
      {tab === 'settings' && <SettingsTab />}
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
    <div className="mt-8 grid gap-12">
      <AddStreams onAdded={onChange} />
      <section>
        <h2 className="eyebrow mb-3">Queue</h2>
        {!loaded ? (
          <p className="text-sm text-ink-500">Loading…</p>
        ) : streams.length === 0 ? (
          <p className="text-sm text-ink-500">No streams yet. Paste links above to start.</p>
        ) : (
          <ul className="divide-y divide-ink-800 border-y border-ink-800">
            {streams.map((s) => (
              <StreamRow key={s.id} stream={s} clips={byStream.get(s.id) ?? []} onChange={onChange} />
            ))}
          </ul>
        )}
      </section>
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
    <section>
      <h2 className="eyebrow mb-3">Add streams</h2>
      <textarea
        className="field min-h-24 font-mono text-sm"
        placeholder="Paste stream / VOD links — one per line (YouTube, Twitch, Kick)"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <div className="mt-4 flex flex-wrap items-end gap-6">
        <label className="w-40 text-xs text-ink-400">
          Only from (optional)
          <input
            className="field"
            placeholder="0:00:00"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
        </label>
        <label className="w-40 text-xs text-ink-400">
          Until (optional)
          <input className="field" placeholder="end" value={end} onChange={(e) => setEnd(e.target.value)} />
        </label>
        <button className="btn btn-primary" disabled={busy || !text.trim()} onClick={submit}>
          {busy ? 'Queuing…' : 'Queue streams'}
        </button>
      </div>
      {message && (
        <p
          className={`mt-3 whitespace-pre-line text-sm ${
            message.tone === 'good' ? 'text-signal-good' : 'text-signal-bad'
          }`}
        >
          {message.text}
        </p>
      )}
    </section>
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

  return (
    <li className="py-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-ink-100">
            {stream.channel || stream.platform || 'Stream'}
            <span className="ml-2 font-normal text-ink-400">{stream.title || stream.url}</span>
          </p>
          <p className="mt-1 text-xs text-ink-400">
            <span className={stream.status === 'failed' ? 'text-signal-bad' : 'text-sodium-400'}>
              {STATUS_LABEL[stream.status] ?? stream.status}
            </span>
            {stream.detail && <span> · {stream.detail}</span>}
            {stream.duration_s > 0 && <span> · {clock(stream.duration_s)} long</span>}
            {(stream.range_start_s != null || stream.range_end_s != null) && (
              <span>
                {' '}
                · range {clock(stream.range_start_s ?? 0)}–
                {stream.range_end_s != null ? clock(stream.range_end_s) : 'end'}
              </span>
            )}
          </p>
          {active && stream.status !== 'analyzed' && (
            <div className="mt-2 h-1 w-full max-w-md bg-ink-800">
              <div
                className="h-1 bg-sodium-500 transition-all"
                style={{ width: `${Math.round(stream.progress * 100)}%` }}
              />
            </div>
          )}
          <p className="mt-2 text-xs text-ink-500">
            {counts.ready ?? 0} ready · {working} in progress · {counts.rejected ?? 0} rejected
            {counts.qa_failed ? ` · ${counts.qa_failed} failed QA` : ''}
            {typeof signals.hype_markers === 'number' && ` · ${signals.hype_markers} hype markers`}
            {typeof signals.chat_messages === 'number' && ` · ${signals.chat_messages} chat msgs`}
          </p>
          {notes.length > 0 && <p className="mt-1 text-xs text-ink-500">Signals: {notes.join('; ')}</p>}
          {stream.error && <p className="mt-2 whitespace-pre-line text-xs text-signal-bad">{stream.error}</p>}
          {error && <p className="mt-2 whitespace-pre-line text-xs text-signal-bad">{error}</p>}
        </div>
        <div className="flex shrink-0 gap-2">
          {clips.length > 0 && (
            <button className="btn btn-ghost" onClick={() => setOpen((v) => !v)}>
              {open ? 'Hide moments' : `Moments (${clips.length})`}
            </button>
          )}
          {active && (
            <button className="btn btn-ghost" disabled={busy} onClick={() => run(() => gtaApi.cancel(stream.id))}>
              Cancel
            </button>
          )}
          {(stream.status === 'failed' || stream.status === 'cancelled') && (
            <button className="btn btn-primary" disabled={busy} onClick={() => run(() => gtaApi.retry(stream.id))}>
              Retry
            </button>
          )}
          {!active && (
            <button className="btn btn-quiet" disabled={busy} onClick={() => run(() => gtaApi.remove(stream.id))}>
              Remove
            </button>
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
    c.status === 'ready' ? 'text-signal-good' : c.status === 'rejected' ? 'text-ink-500' : 'text-sodium-400'
  return (
    <ul className="mt-4 grid gap-2 border-l border-ink-800 pl-4">
      {sorted.map((c) => (
        <li key={c.id} className="text-xs">
          <span className="font-mono text-ink-400">
            {clock(c.start_s)}–{clock(c.end_s)}
          </span>{' '}
          <span className={tone(c)}>{c.status}</span>{' '}
          <span className="text-ink-200">{c.hook_title || c.hook_quote}</span>
          {c.hype_score > 0 && <span className="text-ink-500"> · hype {Math.round(c.hype_score)}</span>}
          {c.reject_reason && <span className="text-ink-500"> — {c.reject_reason}</span>}
          {c.error && c.status !== 'ready' && <span className="text-signal-bad"> — {c.error}</span>}
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
      <h2 className="eyebrow mb-3">
        {title} ({clips.length})
      </h2>
      {clips.length === 0 ? (
        <p className="text-sm text-ink-500">Finished clips appear here with their caption and hashtags.</p>
      ) : (
        <div className="grid gap-6 sm:grid-cols-2 xl:grid-cols-3">
          {sorted.map((c) => (
            <ClipCard key={c.id} clip={c} streamer={names.get(c.stream_id) ?? ''} />
          ))}
        </div>
      )}
    </section>
  )
}

function QaChip({ label, ok }: { label: string; ok: boolean | undefined }) {
  if (ok === undefined) return null
  return (
    <span
      className={`border px-1.5 py-0.5 text-[10px] tracking-wide uppercase ${
        ok ? 'border-signal-good text-signal-good' : 'border-signal-bad text-signal-bad'
      }`}
    >
      {label}
    </span>
  )
}

function ClipCard({ clip, streamer }: { clip: GtaClip; streamer: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    if (await copyText(clip.post_text)) {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    }
  }
  const qa = clip.qa
  return (
    <article className="flex flex-col gap-3 bg-ink-850 p-3">
      {clip.has_video ? (
        <video
          className="aspect-[9/16] w-full bg-black"
          src={gtaApi.videoUrl(clip.id)}
          controls
          preload="metadata"
        />
      ) : (
        <div className="flex aspect-[9/16] w-full items-center justify-center bg-ink-800 text-xs text-ink-500">
          Video file missing
        </div>
      )}
      <div>
        <p className="text-sm font-semibold text-ink-100">{clip.hook_title}</p>
        <p className="text-xs text-ink-400">
          {streamer} · {MOMENT_LABEL[clip.moment_type] ?? clip.moment_type} · score {clip.score}
          {qa.duration_s ? ` · ${qa.duration_s.toFixed(0)} s` : ''}
          {qa.turnaround_s !== undefined && ` · live turnaround ${Math.round(qa.turnaround_s)} s`}
        </p>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <QaChip label="< 60 s" ok={qa.length?.ok} />
        <QaChip label="loudness" ok={qa.loudness?.ok} />
        <QaChip label="no black" ok={qa.black_frames?.ok} />
        <QaChip label="face-cam" ok={qa.facecam?.ok} />
      </div>
      <textarea
        readOnly
        className="field min-h-24 text-xs"
        value={clip.post_text}
        onFocus={(e) => e.currentTarget.select()}
        aria-label="Caption and hashtags"
      />
      <button className="btn btn-primary" onClick={copy}>
        {copied ? 'Copied ✓' : 'Copy caption + hashtags'}
      </button>
      <p className="truncate font-mono text-[10px] text-ink-500" title={clip.path}>
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
    <div className="mt-8 grid gap-12">
      <section>
        <h2 className="eyebrow mb-3">Clip this live stream</h2>
        <p className="mb-3 max-w-2xl text-sm text-ink-400">
          Listens every 20 seconds. When the chat explodes or the streamer loses it, the last 60 seconds
          are judged with the same strict rules and a "yes" is rendered within about 3 minutes. Twitch
          also uses live chat; YouTube and Kick lives trigger on the streamer's audio.
        </p>
        <div className="flex flex-wrap items-end gap-4">
          <input
            className="field max-w-xl flex-1"
            placeholder="https://www.twitch.tv/streamer"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button className="btn btn-primary" disabled={busy || !url.trim()} onClick={start}>
            Start watching
          </button>
        </div>
        {error && <p className="mt-3 whitespace-pre-line text-sm text-signal-bad">{error}</p>}
      </section>
      <section>
        <h2 className="eyebrow mb-3">Live watches</h2>
        {streams.length === 0 ? (
          <p className="text-sm text-ink-500">Nothing is being watched.</p>
        ) : (
          <ul className="divide-y divide-ink-800 border-y border-ink-800">
            {streams.map((s) => {
              const running = LIVE_RUNNING.has(s.status)
              const mine = byStream.get(s.id) ?? []
              const readyCount = mine.filter((c) => c.status === 'ready').length
              const rejected = mine.filter((c) => c.status === 'rejected').length
              return (
                <li key={s.id} className="flex flex-wrap items-start justify-between gap-4 py-4">
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-semibold text-ink-100">{s.channel || s.url}</p>
                    <p className="mt-1 text-xs text-ink-400">
                      <span className={running ? 'text-signal-good' : 'text-ink-400'}>
                        {STATUS_LABEL[s.status] ?? s.status}
                      </span>
                      {s.detail && ` · ${s.detail}`} · {readyCount} ready · {rejected} rejected
                    </p>
                    {s.error && <p className="mt-1 whitespace-pre-line text-xs text-signal-bad">{s.error}</p>}
                  </div>
                  <div className="flex gap-2">
                    {running ? (
                      <button className="btn btn-ghost" disabled={busy} onClick={() => run(() => gtaApi.stopLive(s.id))}>
                        Stop
                      </button>
                    ) : (
                      <>
                        <button className="btn btn-primary" disabled={busy} onClick={() => run(() => gtaApi.retry(s.id))}>
                          Watch again
                        </button>
                        <button className="btn btn-quiet" disabled={busy} onClick={() => run(() => gtaApi.remove(s.id))}>
                          Remove
                        </button>
                      </>
                    )}
                  </div>
                </li>
              )
            })}
          </ul>
        )}
      </section>
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
    <div>
      <h3 className="eyebrow mb-2">{title}</h3>
      <table className="w-full text-sm">
        <tbody>
          {entries.map(([name, stat]) => (
            <tr key={name} className="border-b border-ink-800">
              <td className="py-1.5 text-ink-200">{MOMENT_LABEL[name] ?? name}</td>
              <td className="py-1.5 text-right text-ink-400">{stat.count} Shorts</td>
              <td className="py-1.5 text-right text-ink-400">{stat.average_views.toLocaleString()} avg</td>
              <td
                className={`py-1.5 text-right font-semibold ${
                  stat.vs_channel >= 1 ? 'text-signal-good' : 'text-signal-bad'
                }`}
              >
                {stat.vs_channel.toFixed(2)}×
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
    <div className="mt-8 grid gap-10">
      <section>
        <h2 className="eyebrow mb-3">Look at what's working</h2>
        <p className="mb-3 max-w-2xl text-sm text-ink-400">
          Reads the view counts of the channel's posted Shorts, finds which moment types and streamers beat
          the channel average, and leans future picks toward them (only groups with 3+ Shorts count).
        </p>
        <div className="flex flex-wrap items-end gap-4">
          <input
            className="field max-w-xl flex-1"
            placeholder="https://www.youtube.com/@yourchannel"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
          <button className="btn btn-primary" disabled={busy || !url.trim()} onClick={runReport}>
            {busy ? 'Reading views…' : 'Look at what’s working'}
          </button>
        </div>
        {error && <p className="mt-3 whitespace-pre-line text-sm text-signal-bad">{error}</p>}
      </section>
      {report && (
        <section className="grid gap-8">
          <p className="text-sm text-ink-300">
            {report.count} Shorts · channel average {report.channel_average.toLocaleString()} views ·{' '}
            {report.matched} matched to GTA clips
          </p>
          {report.bias.summary && (
            <p className="whitespace-pre-line border-l-2 border-sodium-500 pl-3 text-sm text-ink-200">
              {report.bias.summary}
            </p>
          )}
          <div className="grid gap-8 lg:grid-cols-2">
            <StatTable title="Moment types" rows={report.moment_types} />
            <StatTable title="Streamers" rows={report.streamers} />
          </div>
        </section>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// Settings
// --------------------------------------------------------------------------

type NumberKey =
  | 'chunk_minutes'
  | 'min_clip_s'
  | 'max_clip_s'
  | 'max_clips_per_stream'
  | 'hook_title_s'
  | 'dead_air_min_s'
  | 'loudness_lufs'
  | 'face_min_ratio'
  | 'network_workers'

const NUMBER_FIELDS: { key: NumberKey; label: string; step: number }[] = [
  { key: 'chunk_minutes', label: 'Transcript chunk (minutes)', step: 1 },
  { key: 'min_clip_s', label: 'Shortest clip (s)', step: 1 },
  { key: 'max_clip_s', label: 'Longest clip (s, under 60)', step: 1 },
  { key: 'max_clips_per_stream', label: 'Most clips per stream', step: 1 },
  { key: 'hook_title_s', label: 'Hook title on screen (s)', step: 0.5 },
  { key: 'dead_air_min_s', label: 'Cut silences longer than (s)', step: 0.1 },
  { key: 'loudness_lufs', label: 'Loudness target (LUFS)', step: 0.5 },
  { key: 'face_min_ratio', label: 'Face visible in at least (0–1)', step: 0.05 },
  { key: 'network_workers', label: 'Parallel downloads / judges', step: 1 },
]

function SettingsTab() {
  const [settings, setSettings] = useState<GtaSettings | null>(null)
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [message, setMessage] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)
  const [clientId, setClientId] = useState('')
  const [secret, setSecret] = useState('')

  const load = useCallback(() => {
    gtaApi
      .settings()
      .then((s) => {
        setSettings(s)
        const next: Record<string, string> = { output_dir: s.output_dir, caption_style: s.caption_style }
        for (const f of NUMBER_FIELDS) next[f.key] = String(s[f.key])
        next.listen_interval_s = String(s.live.listen_interval_s)
        next.context_s = String(s.live.context_s)
        setDraft(next)
      })
      .catch((err) => setMessage({ tone: 'bad', text: errorText(err) }))
  }, [])
  useEffect(load, [load])

  if (!settings) return <p className="mt-8 text-sm text-ink-500">{message?.text ?? 'Loading…'}</p>

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
      setMessage({ tone: 'good', text: 'Saved.' })
      load()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    }
  }

  const saveTwitch = async (clear = false) => {
    try {
      const result = await gtaApi.saveTwitch(clear ? '' : clientId.trim(), clear ? '' : secret.trim())
      setMessage({
        tone: 'good',
        text: result.twitch_credentials_set ? 'Twitch credentials saved.' : 'Twitch credentials removed.',
      })
      setClientId('')
      setSecret('')
      load()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    }
  }

  const field = (key: string, label: string, props: Record<string, unknown> = {}) => (
    <label key={key} className="text-xs text-ink-400">
      {label}
      <input
        className="field"
        value={draft[key] ?? ''}
        onChange={(e) => setDraft((d) => ({ ...d, [key]: e.target.value }))}
        {...props}
      />
    </label>
  )

  return (
    <div className="mt-8 grid max-w-3xl gap-10">
      <section className="grid gap-5 sm:grid-cols-2">
        {NUMBER_FIELDS.map((f) => field(f.key, f.label, { type: 'number', step: f.step }))}
        {field('listen_interval_s', 'GTA Live: listen every (s)', { type: 'number', step: 1 })}
        {field('context_s', 'GTA Live: judge the last (s)', { type: 'number', step: 5 })}
        <label className="text-xs text-ink-400">
          Caption style
          <select
            className="field"
            value={draft.caption_style ?? ''}
            onChange={(e) => setDraft((d) => ({ ...d, caption_style: e.target.value }))}
          >
            <option value="bold_pop">Bold Pop</option>
            <option value="karaoke_fill">Karaoke Fill</option>
          </select>
        </label>
        <div className="sm:col-span-2">
          {field('output_dir', 'Save finished clips to (empty = AutoClip exports/gta)')}
        </div>
      </section>
      <div className="flex items-center gap-4">
        <button className="btn btn-primary" onClick={save}>
          Save settings
        </button>
        {message && (
          <p
            className={`whitespace-pre-line text-sm ${
              message.tone === 'good' ? 'text-signal-good' : 'text-signal-bad'
            }`}
          >
            {message.text}
          </p>
        )}
      </div>
      <section>
        <h2 className="eyebrow mb-2">Twitch viewer clips (optional)</h2>
        <p className="mb-3 text-sm text-ink-400">
          With a free Twitch developer app (dev.twitch.tv → Your Console → Register an application), the
          Hype Curve also uses the clips viewers made of a VOD. Status:{' '}
          <span className={settings.twitch_credentials_set ? 'text-signal-good' : 'text-ink-300'}>
            {settings.twitch_credentials_set ? 'connected' : 'not set'}
          </span>
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <input
            className="field"
            placeholder="Client ID"
            value={clientId}
            onChange={(e) => setClientId(e.target.value)}
          />
          <input
            className="field"
            type="password"
            placeholder="Client Secret"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
          />
        </div>
        <div className="mt-4 flex gap-3">
          <button
            className="btn btn-ghost"
            disabled={!clientId.trim() || !secret.trim()}
            onClick={() => saveTwitch()}
          >
            Save credentials
          </button>
          {settings.twitch_credentials_set && (
            <button className="btn btn-quiet" onClick={() => saveTwitch(true)}>
              Remove
            </button>
          )}
        </div>
      </section>
    </div>
  )
}
