/**
 * GTA Live rows: the channel's name (not the link) with a near-live picture on
 * hover, and a status line that surfaces background problems (AI judge down,
 * capture stalled, render failed) instead of leaving them in the server log.
 */

import { useEffect, useRef, useState } from 'react'

import { type AiUsage, type AiUsageKey, gtaApi, type GtaStream } from './api'

interface Problem {
  where: string
  text: string
  at: string
}

interface Moment {
  why: string
  at: string
  result: string
}

function ago(iso: string | undefined): string {
  if (!iso) return ''
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return ''
  const s = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (s < 10) return 'just now'
  if (s < 60) return `${s} s ago`
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  return `${Math.floor(s / 3600)} h ago`
}

/** The channel name; falls back to something readable while it's looked up. */
export function streamName(s: GtaStream): string {
  if (s.channel) return s.channel
  try {
    const u = new URL(s.url)
    const host = u.hostname.replace(/^(www|m)\./, '')
    if (u.searchParams.get('v')) return `${host} live (looking up channel…)`
    return host + u.pathname.replace(/\/$/, '')
  } catch {
    return s.url
  }
}

const PREVIEW_W = 320
const PREVIEW_H = 212

export function LiveStreamName({ stream, running }: { stream: GtaStream; running: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  const [tick, setTick] = useState(0)
  const [failed, setFailed] = useState(false)
  const thumb = typeof stream.signals.thumbnail === 'string' ? stream.signals.thumbnail : ''

  useEffect(() => {
    if (!pos || !running) return
    const id = window.setInterval(() => setTick((t) => t + 1), 2500)
    return () => window.clearInterval(id)
  }, [pos, running])

  const open = () => {
    const r = ref.current?.getBoundingClientRect()
    if (!r) return
    setFailed(false)
    setTick((t) => t + 1)
    const below = r.bottom + 8 + PREVIEW_H < window.innerHeight
    setPos({
      left: Math.min(r.left, window.innerWidth - PREVIEW_W - 12),
      top: below ? r.bottom + 8 : Math.max(8, r.top - 8 - PREVIEW_H),
    })
  }
  const close = () => setPos(null)
  const src = failed ? thumb : `/api/gta/live/${encodeURIComponent(stream.id)}/frame?t=${tick}`

  return (
    <div
      ref={ref}
      className="min-w-0 max-w-full cursor-default outline-none"
      onMouseEnter={open}
      onMouseLeave={close}
      onFocus={open}
      onBlur={close}
      tabIndex={0}
    >
      <p className="truncate text-sm font-semibold text-fg">{streamName(stream)}</p>
      {stream.title && (
        <p className="truncate text-xs text-fg-muted" title={stream.title}>
          {stream.title}
        </p>
      )}
      {pos && (
        <div
          className="pointer-events-none fixed z-50 overflow-hidden rounded-xl border border-line bg-surface shadow-pop"
          style={{ left: pos.left, top: pos.top, width: PREVIEW_W }}
        >
          {src ? (
            <img
              src={src}
              alt={`Preview of ${streamName(stream)}`}
              className="aspect-video w-full bg-black object-cover"
              onError={() => setFailed(true)}
            />
          ) : (
            <div className="grid aspect-video w-full place-items-center bg-subtle text-xs text-fg-muted">
              No picture yet
            </div>
          )}
          <div className="flex items-center gap-1.5 px-3 py-2 text-xs text-fg-muted">
            {running && !failed && <span className="h-1.5 w-1.5 rounded-full bg-danger" />}
            {failed ? 'Stream thumbnail' : running ? 'Live · refreshes every few seconds' : 'Last picture captured'}
          </div>
        </div>
      )}
    </div>
  )
}

const WHERE: Record<string, string> = {
  ai: 'AI',
  capture: 'Capture',
  listen: 'Listening',
  judge: 'Judging',
  render: 'Render',
}

interface Pulse {
  at?: string
  listens?: number
  stream_s?: number
  need_z?: number
  audio_z?: number
  loudest_db?: number
  typical_db?: number
  baseline_s?: number
  warming_up?: boolean
  chat?: boolean
  chat_msgs?: number
  chat_x?: number | null
  cooldown?: boolean
}

interface ActivityRow {
  at: string
  text: string
}

function mmss(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = String(s % 60).padStart(2, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${r}` : `${m}:${r}`
}

/** "How close was the last pass to a trigger" as a small meter + words. */
function PulseLine({ pulse }: { pulse: Pulse }) {
  if (!pulse.at) return <p className="text-fg-muted">Connecting… the first listen pass runs within seconds.</p>
  const need = pulse.need_z ?? 3
  const z = pulse.audio_z
  const pct = z == null ? 0 : Math.max(0, Math.min(100, (z / need) * 100))
  let words: string
  if (pulse.warming_up) {
    words = `Learning the stream's normal loudness (${pulse.baseline_s ?? 0}/30 s heard)`
  } else if (z == null) {
    words = 'Measured, no sound in the last pass'
  } else {
    words = `Loudest moment ${z >= 0 ? '+' : ''}${z.toFixed(1)}σ vs normal (a clip needs +${need.toFixed(1)}σ)`
  }
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-fg-muted">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-success" />
          Listened {ago(pulse.at)} · pass #{pulse.listens ?? 0}
          {pulse.stream_s != null && ` · ${mmss(pulse.stream_s)} of stream heard`}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <div className="h-1.5 w-28 overflow-hidden rounded-full bg-subtle" title="How close the last pass came to triggering the AI judge">
          <div
            className={`h-full rounded-full ${pct >= 100 ? 'bg-danger' : pct >= 66 ? 'bg-warn' : 'bg-accent'}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-fg-muted">{words}</span>
      </div>
      <p className="text-fg-muted">
        {pulse.chat
          ? `Chat: ${pulse.chat_msgs ?? 0} messages in the last pass${pulse.chat_x != null ? ` (${pulse.chat_x}× normal; needs 2.5×)` : ''}`
          : 'Chat isn’t read on YouTube/Kick: loud spikes trigger the judge, and it also checks what was said every minute'}
      </p>
    </div>
  )
}

/** Moments flagged, the AI's last good answer, the last background problem, activity log. */
export function LiveHealth({ stream }: { stream: GtaStream }) {
  const sig = stream.signals
  const [showLog, setShowLog] = useState(false)
  const moments = typeof sig.moments === 'number' ? sig.moments : 0
  const last = sig.last_moment as Moment | undefined
  const problem = sig.last_problem as Problem | undefined
  const pulse = (sig.pulse as Pulse | undefined) ?? {}
  const activity = Array.isArray(sig.activity) ? (sig.activity as ActivityRow[]) : []
  const aiOk = typeof sig.ai_ok_at === 'string' ? sig.ai_ok_at : ''
  // A problem older than the AI's last good answer is no longer current for the AI.
  const stale =
    problem?.where === 'ai' && aiOk && new Date(aiOk).getTime() > new Date(problem.at).getTime()
  return (
    <div className="mt-2 space-y-1.5 text-xs">
      <PulseLine pulse={pulse} />
      <p className="text-fg-muted">
        {moments
          ? `${moments} moment${moments === 1 ? '' : 's'} sent to the AI judge`
          : 'Nothing sent to the AI judge yet (no spike big enough)'}
        {last && ` · last ${ago(last.at)}: ${last.result || last.why}`}
        {aiOk && ` · AI answered ${ago(aiOk)}`}
        {sig.stalled === true && ' · capture stalled'}
      </p>
      {problem && !stale && (
        <p className="text-warn" title={problem.text}>
          ⚠ {WHERE[problem.where] ?? problem.where} problem {ago(problem.at)}: {problem.text}
        </p>
      )}
      {activity.length > 0 && (
        <div>
          <button
            type="button"
            className="font-medium text-accent hover:underline"
            onClick={() => setShowLog((v) => !v)}
          >
            {showLog ? 'Hide activity' : `Show activity (${activity.length})`}
          </button>
          {showLog && (
            <ul className="mt-1 space-y-0.5 rounded-lg bg-subtle px-3 py-2 text-fg-muted">
              {activity.map((row, i) => (
                <li key={`${row.at}-${i}`} className="flex gap-2">
                  <span className="w-20 shrink-0 tabular-nums">{ago(row.at)}</span>
                  <span className="min-w-0">{row.text}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// AI usage: every key's calls today and the headroom its provider reports
// --------------------------------------------------------------------------

function num(v: string | number | undefined): number | null {
  if (v === undefined || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

function compact(n: number): string {
  return n >= 10_000 ? `${Math.round(n / 1000)}k` : n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}

function Headroom({ label, left, limit }: { label: string; left: number | null; limit: number | null }) {
  if (left === null) return null
  const pct = limit ? Math.max(0, Math.min(100, (left / limit) * 100)) : null
  const tone = pct === null ? 'bg-accent' : pct < 10 ? 'bg-danger' : pct < 30 ? 'bg-warn' : 'bg-success'
  return (
    <div className="min-w-[9rem]">
      <div className="flex justify-between gap-2 text-[11px] text-fg-muted">
        <span>{label}</span>
        <span className="tabular-nums">
          {compact(left)}
          {limit ? ` / ${compact(limit)}` : ''} left
        </span>
      </div>
      {pct !== null && (
        <div className="mt-0.5 h-1.5 overflow-hidden rounded-full bg-subtle">
          <div className={`h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  )
}

function KeyRow({ k, now }: { k: AiUsageKey; now: number }) {
  const lim = Object.values(k.limits).sort((a, b) => Number(b.at ?? 0) - Number(a.at ?? 0))[0] ?? {}
  const isWhisper = k.label.includes('(whisper)')
  const unlimited = k.host.includes('utim')
  const t = k.today
  const usedLine = isWhisper
    ? `${t.calls} transcriptions · ${Math.round(t.audio_s / 60)} min of audio`
    : `${t.calls} calls · ${compact(t.tokens_in + t.tokens_out)} tokens`
  const resting = k.resting_s > 0
  const ago = k.last_at ? Math.max(0, Math.round(now - k.last_at)) : null
  return (
    <li className="flex flex-wrap items-center gap-x-6 gap-y-2 py-3 text-xs">
      <div className="w-56 min-w-0">
        <p className="font-medium text-fg">{k.label}</p>
        <p className="text-fg-muted">
          {usedLine}
          {t.errors ? ` · ${t.errors} errors` : ''}
          {t.rate_limited ? ` · ${t.rate_limited}× rate-limited` : ''}
        </p>
      </div>
      {unlimited ? (
        <span className="text-fg-muted">No token limits · max 2 requests at once (extra calls go to Groq)</span>
      ) : (
        <>
          <Headroom
            label={isWhisper ? 'Requests (day)' : 'Requests (day)'}
            left={num(lim['x-ratelimit-remaining-requests'])}
            limit={num(lim['x-ratelimit-limit-requests'])}
          />
          <Headroom
            label={isWhisper ? 'Audio s (hour)' : 'Tokens (minute)'}
            left={num(lim['x-ratelimit-remaining-tokens'] ?? lim['x-ratelimit-remaining-audio-seconds'])}
            limit={num(lim['x-ratelimit-limit-tokens'] ?? lim['x-ratelimit-limit-audio-seconds'])}
          />
        </>
      )}
      <span className={`ml-auto ${resting ? 'font-semibold text-warn' : 'text-fg-muted'}`}>
        {resting
          ? `Resting ${Math.ceil(k.resting_s / 60)} min (rate limit)`
          : ago === null
            ? 'Not used yet'
            : `Used ${ago < 60 ? `${ago} s` : `${Math.round(ago / 60)} min`} ago`}
      </span>
      {k.last_error && now - k.last_error.at < 3600 && (
        <p className="w-full text-fg-muted">
          Last refused call (handled: it went to another key): {k.last_error.text}
        </p>
      )}
    </li>
  )
}

/** Calls and headroom per AI key, refreshed every 10 s. */
export function AiUsageCard() {
  const [data, setData] = useState<AiUsage | null>(null)
  const [open, setOpen] = useState(true)
  useEffect(() => {
    const load = () => {
      gtaApi
        .aiUsage()
        .then(setData)
        .catch(() => undefined)
    }
    load()
    const id = window.setInterval(load, 10_000)
    return () => window.clearInterval(id)
  }, [])
  if (!data) return null
  const keys = [...data.keys].sort((a, b) => a.label.localeCompare(b.label))
  const today = keys.reduce(
    (acc, k) => ({ calls: acc.calls + k.today.calls, audio: acc.audio + k.today.audio_s }),
    { calls: 0, audio: 0 },
  )
  const resting = keys.filter((k) => k.resting_s > 0).length
  return (
    <section className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-fg">AI usage today</h2>
        <button type="button" className="text-xs font-medium text-accent" onClick={() => setOpen((v) => !v)}>
          {open ? 'Hide keys' : 'Show keys'}
        </button>
      </div>
      <p className="mt-1 text-sm text-fg-muted">
        {today.calls} calls · {Math.round(today.audio / 60)} min transcribed
        {resting ? ` · ${resting} key${resting === 1 ? '' : 's'} resting after a rate limit` : ' · all keys available'}.
        Groq free-tier keys rest automatically when they run low; utim and local Whisper take over meanwhile.
      </p>
      {open && keys.length > 0 && (
        <ul className="mt-2 divide-y divide-line">
          {keys.map((k) => (
            <KeyRow key={k.label} k={k} now={data.now} />
          ))}
        </ul>
      )}
    </section>
  )
}
