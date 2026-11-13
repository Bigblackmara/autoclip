/**
 * StreamsPanel — the streams half of the Live Watch page.
 *
 * An operations desk for watched livestreams: a readout strip (watching /
 * clipping / queue depth / disk), compact table rows with monospace state
 * telemetry, and three rails of truth that can only disagree for seconds —
 * row actions merge the server's response immediately, SSE `state` events
 * patch rows the instant the backend changes them, and a 5s poll re-syncs
 * everything so a dropped event feed or a partial payload costs five
 * seconds of staleness at most, never correctness.
 */

import { Fragment, useEffect, useRef, useState } from 'react'

import type { LiveEvent, LiveStatus, LiveStream, LiveStreamState } from './api'
import { liveApi, subscribeLiveEvents } from './api'
import { ErrorNote } from '../components/ErrorNote'

const POLL_MS = 5_000
const FEED_STALE_MS = 15_000
const CONFIRM_RESET_MS = 4_500

/** Rows in these states offer Resume; every other state offers Pause. */
const RESUME_STATES: readonly LiveStreamState[] = ['off', 'ended', 'error']
const LIVE_STATES: readonly string[] = [
  'off',
  'waiting_for_live',
  'connecting',
  'watching',
  'clipping',
  'reconnecting',
  'ended',
  'error',
]

function isLiveState(value: unknown): value is LiveStreamState {
  return typeof value === 'string' && LIVE_STATES.includes(value)
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

/**
 * Merge whatever fields a `state` SSE payload happens to carry. The shape is
 * A10's to define, so every field is type-checked and optional; the 5s poll
 * backfills anything the event omits or garbles.
 */
function streamPatch(src: Record<string, unknown>): Partial<LiveStream> {
  const patch: Partial<LiveStream> = {}
  if (isLiveState(src.state)) patch.state = src.state
  if (typeof src.state_detail === 'string') patch.state_detail = src.state_detail
  if (typeof src.last_error === 'string' || src.last_error === null)
    patch.last_error = src.last_error
  if (typeof src.enabled === 'boolean') patch.enabled = src.enabled
  if (typeof src.label === 'string') patch.label = src.label
  if (typeof src.url === 'string') patch.url = src.url
  return patch
}

/** Label fallback: a scheme-less hand-typed URL still reads as its host… or itself. */
function hostOf(url: string): string {
  if (!url) return '(no url)'
  try {
    return new URL(url).host || url
  } catch {
    return url
  }
}

function fmtMB(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

const PILL: Record<LiveStreamState, { label: string; cls: string }> = {
  watching: { label: 'watching', cls: 'border-signal-good/40 bg-signal-good/10 text-signal-good' },
  clipping: { label: 'clipping', cls: 'border-sodium-500/40 bg-sodium-500/10 text-sodium-400' },
  reconnecting: {
    label: 'reconnecting',
    cls: 'border-violet-400/40 bg-violet-400/10 text-violet-300',
  },
  connecting: { label: 'connecting', cls: 'border-sky-400/40 bg-sky-400/10 text-sky-300' },
  error: { label: 'error', cls: 'border-signal-bad/40 bg-signal-bad/10 text-signal-bad' },
  waiting_for_live: {
    label: 'waiting',
    cls: 'border-slate-400/30 bg-slate-400/10 text-slate-300',
  },
  off: { label: 'off', cls: 'border-ink-600 bg-ink-850 text-ink-400' },
  ended: { label: 'ended', cls: 'border-ink-700 bg-ink-900 text-ink-400' },
}

function StatePill({ state }: { state: LiveStreamState }) {
  const pill = PILL[state]
  return (
    <span
      title={state}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 font-mono text-[10px] font-medium uppercase tracking-wider ${pill.cls}`}
    >
      <span className="size-1.5 rounded-full bg-current" />
      {pill.label}
    </span>
  )
}

function Stat({
  label,
  value,
  tone = 'default',
  title,
}: {
  label: string
  value: string
  tone?: 'default' | 'good' | 'bad' | 'sodium'
  title?: string
}) {
  const toneCls =
    tone === 'good'
      ? 'text-signal-good'
      : tone === 'bad'
        ? 'text-signal-bad'
        : tone === 'sodium'
          ? 'text-sodium-400'
          : 'text-ink-100'
  return (
    <span className="flex items-baseline gap-2 whitespace-nowrap" title={title}>
      <span className="text-ink-400">{label}</span>
      <span className={`numeric ${toneCls}`}>{value}</span>
    </span>
  )
}

export function StreamsPanel() {
  const [streams, setStreams] = useState<LiveStream[] | null>(null)
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<Error | null>(null)
  const [syncError, setSyncError] = useState<Error | null>(null)
  const [lastTick, setLastTick] = useState<number | null>(null)
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<Error | null>(null)
  const [url, setUrl] = useState('')
  const [label, setLabel] = useState('')
  const [urlError, setUrlError] = useState<string | null>(null)
  const [busy, setBusy] = useState<ReadonlySet<string>>(new Set<string>())
  const [rowErrors, setRowErrors] = useState<Record<string, Error | null>>({})
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const streamsRef = useRef<LiveStream[] | null>(null)
  const refreshRef = useRef<() => void>(() => {})
  const urlRef = useRef<HTMLInputElement | null>(null)

  /** All writes funnel through here so the ref mirror of the list never lies. */
  const updateStreams = (next: LiveStream[]) => {
    streamsRef.current = next
    setStreams(next)
  }

  const patchRow = (id: string, patch: Partial<LiveStream>): boolean => {
    const prev = streamsRef.current
    if (!prev || !prev.some((row) => row.id === id)) return false
    updateStreams(prev.map((row) => (row.id === id ? { ...row, ...patch } : row)))
    return true
  }

  const setRowBusy = (id: string, on: boolean) =>
    setBusy((prev) => {
      const next = new Set(prev)
      if (on) next.add(id)
      else next.delete(id)
      return next
    })

  const clearRowError = (id: string) => setRowErrors((prev) => ({ ...prev, [id]: null }))

  const putRowError = (id: string, err: unknown) =>
    setRowErrors((prev) => ({
      ...prev,
      [id]: err instanceof Error ? err : new Error(String(err)),
    }))

  useEffect(() => {
    let cancelled = false
    let inFlight = false

    const refresh = async () => {
      if (inFlight) return
      inFlight = true
      const [streamsRes, statusRes] = await Promise.allSettled([
        liveApi.listStreams(),
        liveApi.status(),
      ])
      inFlight = false
      if (cancelled) return
      let syncErr: Error | null = null
      if (streamsRes.status === 'fulfilled') {
        updateStreams(streamsRes.value)
        setLoadError(null)
      } else {
        syncErr = streamsRes.reason as Error
        if (!streamsRef.current) setLoadError(syncErr)
      }
      if (statusRes.status === 'fulfilled') setStatus(statusRes.value)
      else syncErr = syncErr ?? (statusRes.reason as Error)
      setSyncError(syncErr)
      setLoading(false)
    }

    refreshRef.current = () => void refresh()

    const onEvent = (ev: LiveEvent) => {
      setLastTick(Date.now())
      if (ev.type === 'snapshot') {
        refreshRef.current()
        return
      }
      if (ev.type !== 'state') return
      const data = ev.data
      const id =
        ev.stream_id ?? (data && typeof data.stream_id === 'string' ? data.stream_id : undefined)
      const src = asRecord(data?.stream) ?? data
      // Unknown stream or an unusable payload: fall back to a full resync.
      if (!id || !src) {
        refreshRef.current()
        return
      }
      if (!patchRow(id, streamPatch(src))) refreshRef.current()
    }

    const unsubscribe = subscribeLiveEvents(onEvent)
    refreshRef.current()
    const timer = window.setInterval(refreshRef.current, POLL_MS)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      unsubscribe()
    }
  }, [])

  useEffect(() => {
    if (!confirmId) return
    const timer = window.setTimeout(() => setConfirmId(null), CONFIRM_RESET_MS)
    return () => window.clearTimeout(timer)
  }, [confirmId])

  const addStream = async () => {
    const trimmedUrl = url.trim()
    const trimmedLabel = label.trim()
    if (!trimmedUrl) {
      setUrlError('A stream URL is required.')
      urlRef.current?.focus()
      return
    }
    setUrlError(null)
    setAddError(null)
    setAdding(true)
    try {
      const added = await liveApi.addStream(trimmedUrl, trimmedLabel)
      updateStreams([...(streamsRef.current ?? []), added])
      setUrl('')
      setLabel('')
    } catch (err) {
      setAddError(err instanceof Error ? err : new Error(String(err)))
    } finally {
      setAdding(false)
    }
  }

  const watchToggle = async (row: LiveStream) => {
    if (busy.has(row.id)) return
    setConfirmId(null)
    setRowBusy(row.id, true)
    clearRowError(row.id)
    try {
      const updated = RESUME_STATES.includes(row.state)
        ? await liveApi.resumeStream(row.id)
        : await liveApi.pauseStream(row.id)
      // The response is the whole row; merge it so the pill flips at once.
      patchRow(updated.id, updated)
    } catch (err) {
      putRowError(row.id, err)
    } finally {
      setRowBusy(row.id, false)
    }
  }

  const removeStream = async (id: string) => {
    if (busy.has(id)) return
    setConfirmId(null)
    setRowBusy(id, true)
    try {
      await liveApi.removeStream(id)
      if (streamsRef.current) updateStreams(streamsRef.current.filter((row) => row.id !== id))
    } catch (err) {
      putRowError(id, err)
    } finally {
      setRowBusy(id, false)
    }
  }

  if (loading) {
    return (
      <section className="w-full">
        <div className="h-5 w-56 animate-pulse bg-ink-800/60" />
        <div className="mt-4 h-10 animate-pulse border border-ink-800 bg-ink-850/40" />
        <div className="mt-6 space-y-5">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="flex animate-pulse items-center gap-6 border-b border-ink-850 pb-4"
            >
              <div className="h-9 flex-1 bg-ink-800/60" />
              <div className="h-5 w-24 bg-ink-800/60" />
              <div className="h-5 w-40 bg-ink-800/60" />
            </div>
          ))}
        </div>
      </section>
    )
  }

  if (!streams) {
    return (
      <section className="w-full">
        <div className="border-b border-ink-800 pb-2">
          <h2 className="eyebrow">Streams</h2>
        </div>
        <div className="mt-6 space-y-4">
          {loadError && <ErrorNote error={loadError} />}
          <button className="btn btn-ghost" onClick={() => refreshRef.current()}>
            Retry
          </button>
        </div>
      </section>
    )
  }

  const list = streams
  const caps = status?.caps
  const feedFresh = lastTick !== null && Date.now() - lastTick < FEED_STALE_MS
  const watchingText = status
    ? caps
      ? `${status.watching}/${caps.max_concurrent_streams}`
      : String(status.watching)
    : '—'
  const clippingText = status ? String(status.clipping) : '—'
  const diskText = caps ? fmtMB(caps.disk_used_mb) : '—'

  return (
    <section className="w-full">
      <div className="flex items-baseline justify-between gap-4 border-b border-ink-800 pb-2">
        <div className="flex items-baseline gap-3">
          <h2 className="eyebrow">Streams</h2>
          <span className="numeric font-mono text-xs text-ink-400">{list.length} on watch</span>
        </div>
        <span
          className="flex items-center gap-1.5 font-mono text-[11px] text-ink-400"
          title="Health of the live event feed"
        >
          <span
            className={`size-1.5 rounded-full ${
              feedFresh ? 'bg-signal-good' : lastTick ? 'bg-ink-600' : 'bg-ink-700'
            }`}
          />
          {feedFresh ? 'feed live' : lastTick ? 'polling only' : 'feed connecting'}
        </span>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-7 gap-y-2 border border-ink-800 bg-ink-850/30 px-4 py-2.5 font-mono text-xs">
        <Stat
          label="watching"
          value={watchingText}
          tone={status && status.watching > 0 ? 'good' : 'default'}
        />
        <Stat
          label="clipping"
          value={clippingText}
          tone={status && status.clipping > 0 ? 'sodium' : 'default'}
        />
        {caps && caps.clip_queue_max > 0 && (
          <span
            className="flex items-baseline gap-2 whitespace-nowrap"
            title={`Clip queue depth — capacity ${caps.clip_queue_max}`}
          >
            <span className="text-ink-400">queue</span>
            <span
              className={`numeric ${
                caps.clip_queue_len >= caps.clip_queue_max ? 'text-signal-bad' : 'text-ink-100'
              }`}
            >
              {caps.clip_queue_len}/{caps.clip_queue_max}
            </span>
            <span className="h-1 w-14 overflow-hidden bg-ink-700">
              <span
                className={`block h-full ${
                  caps.clip_queue_len >= caps.clip_queue_max
                    ? 'bg-signal-bad'
                    : 'bg-sodium-500'
                }`}
                style={{
                  width: `${Math.min(100, (caps.clip_queue_len / caps.clip_queue_max) * 100)}%`,
                }}
              />
            </span>
          </span>
        )}
        <Stat label="disk" value={diskText} title="Rolling buffer + clips on disk" />
        {syncError && (
          <span
            className="flex items-center gap-2 whitespace-nowrap text-signal-bad"
            title={syncError.message}
          >
            <span className="size-1.5 rounded-full bg-signal-bad" />
            sync failed
          </span>
        )}
      </div>

      {status && !status.enabled && (
        <p className="mt-3 border-l-2 border-sodium-600 pl-3 font-mono text-xs leading-relaxed text-ink-400">
          engine disabled — streams are saved but nothing is watched until Live Watch is
          enabled in settings
        </p>
      )}

      <form
        className="mt-6 grid gap-4 sm:grid-cols-[minmax(0,1fr)_14rem_auto] sm:items-end"
        noValidate
        onSubmit={(e) => {
          e.preventDefault()
          void addStream()
        }}
      >
        <label className="block">
          <span className="eyebrow">Stream URL</span>
          <input
            ref={urlRef}
            className="field mt-1 font-mono text-sm"
            value={url}
            disabled={adding}
            spellCheck={false}
            placeholder="https://www.twitch.tv/channel"
            onChange={(e) => {
              setUrl(e.target.value)
              if (urlError) setUrlError(null)
            }}
          />
          {urlError ? (
            <span className="mt-1.5 block text-xs text-signal-bad">{urlError}</span>
          ) : (
            <span className="mt-1.5 block text-xs leading-snug text-ink-600">
              Twitch, YouTube, Kick — any live link streamlink can play.
            </span>
          )}
        </label>
        <label className="block">
          <span className="eyebrow">Label</span>
          <input
            className="field mt-1 text-sm"
            value={label}
            disabled={adding}
            maxLength={80}
            placeholder="optional"
            onChange={(e) => setLabel(e.target.value)}
          />
        </label>
        <button type="submit" className="btn btn-primary" disabled={adding}>
          {adding ? 'Adding…' : 'Watch'}
        </button>
      </form>

      {addError && (
        <div className="mt-4">
          <ErrorNote error={addError} onDismiss={() => setAddError(null)} />
        </div>
      )}

      {list.length === 0 ? (
        <div className="mt-6 border border-dashed border-ink-700 px-6 py-10 text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-ink-600">
            no streams on watch
          </p>
          <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-ink-300">
            Add a live URL above. AutoClip tails the stream into a rolling buffer,
            listens for the spike, and cuts the clip while the moment is still live.
          </p>
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[760px] table-fixed border-collapse">
            <caption className="sr-only">
              Watched streams with live state, detail, and per-row controls
            </caption>
            <colgroup>
              <col />
              <col className="w-32" />
              <col className="w-72" />
              <col className="w-36" />
            </colgroup>
            <thead>
              <tr className="border-b border-ink-800">
                <th scope="col" className="pb-2 text-left">
                  <span className="eyebrow">Stream</span>
                </th>
                <th scope="col" className="pb-2 text-left">
                  <span className="eyebrow">State</span>
                </th>
                <th scope="col" className="pb-2 text-left">
                  <span className="eyebrow">Detail</span>
                </th>
                <th scope="col" className="pb-2 text-left">
                  <span className="eyebrow">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {list.map((row) => {
                const rowBusy = busy.has(row.id)
                const confirming = confirmId === row.id
                const rowErr = rowErrors[row.id] ?? null
                const offerResume = RESUME_STATES.includes(row.state)
                return (
                  <Fragment key={row.id}>
                    <tr className="border-b border-ink-850 transition-colors hover:bg-ink-850/30">
                      <td className="max-w-0 py-2.5 pr-4 align-middle">
                        <div className="truncate text-sm text-ink-100" title={row.label || row.url}>
                          {row.label || hostOf(row.url)}
                        </div>
                        <div className="mt-0.5 truncate font-mono text-xs text-ink-400" title={row.url}>
                          {row.url}
                        </div>
                      </td>
                      <td className="py-2.5 pr-4 align-middle">
                        <StatePill state={row.state} />
                      </td>
                      <td className="max-w-0 py-2.5 pr-4 align-middle">
                        <div className="truncate font-mono text-xs text-ink-400" title={row.state_detail}>
                          {row.state_detail || '—'}
                        </div>
                        {row.last_error && (
                          <div
                            className="mt-0.5 truncate font-mono text-xs text-signal-bad"
                            title={row.last_error}
                          >
                            {row.last_error}
                          </div>
                        )}
                      </td>
                      <td className="py-2.5 align-middle">
                        <div className="flex items-center gap-2 whitespace-nowrap">
                          <button
                            onClick={() => void watchToggle(row)}
                            disabled={rowBusy}
                            className={
                              offerResume
                                ? 'btn border border-sodium-700 px-2.5 py-1 text-xs text-sodium-400 transition-colors hover:bg-sodium-500/10'
                                : 'btn btn-ghost px-2.5 py-1 text-xs'
                            }
                          >
                            {offerResume ? 'Resume' : 'Pause'}
                          </button>
                          {confirming ? (
                            <button
                              onClick={() => void removeStream(row.id)}
                              disabled={rowBusy}
                              className="btn border border-signal-bad/50 bg-signal-bad/10 px-2.5 py-1 text-xs text-signal-bad transition-colors hover:bg-signal-bad/20"
                            >
                              Remove?
                            </button>
                          ) : (
                            <button
                              onClick={() => setConfirmId(row.id)}
                              disabled={rowBusy}
                              className="btn btn-quiet px-2.5 py-1 text-xs"
                            >
                              Remove
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                    {rowErr && (
                      <tr>
                        <td colSpan={4} className="pb-3">
                          <ErrorNote error={rowErr} onDismiss={() => clearRowError(row.id)} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
