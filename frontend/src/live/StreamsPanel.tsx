/**
 * StreamsPanel — the streams half of the Live Watch page.
 *
 * Three rails of truth that can only disagree for seconds: row actions merge
 * the server's answer at once, `stream_state` frames from the shared live feed
 * patch rows the instant the backend changes them, and a 5 s poll re-syncs
 * everything. A poll that was in flight while a change landed is dropped and
 * asked again, so a stream you just removed can never come back from a stale
 * list.
 */

import { useCallback, useEffect, useId, useRef, useState } from 'react'

import { ErrorNote } from '../components/ErrorNote'
import {
  STREAM_STATES,
  asError,
  errorStatus,
  errorText,
  liveApi,
  subscribeLiveEvents,
  subscribeLiveFeedState,
  type LiveEvent,
  type LiveFeedState,
  type LiveStatus,
  type LiveStream,
  type LiveStreamState,
} from './api'
import { formatMB, plural, streamName } from './format'
import { ConfirmInline, announce } from './ui'

const POLL_MS = 5_000

/** Desktop columns: stream · state · detail · [clips] · actions. Mobile stacks. */
const ROW_COLUMNS = 'sm:grid-cols-[minmax(0,1.3fr)_8.5rem_minmax(0,1fr)_5rem_13rem]'
const ROW_COLUMNS_NO_COUNTS = 'sm:grid-cols-[minmax(0,1.3fr)_8.5rem_minmax(0,1fr)_13rem]'

function isStreamState(value: unknown): value is LiveStreamState {
  return typeof value === 'string' && (STREAM_STATES as readonly string[]).includes(value)
}

/** Merge whatever a `stream_state` frame carries; the poll backfills the rest. */
function streamPatch(src: Record<string, unknown>): Partial<LiveStream> {
  const patch: Partial<LiveStream> = {}
  if (isStreamState(src.state)) patch.state = src.state
  const detail = src.state_detail ?? src.detail
  if (typeof detail === 'string') patch.state_detail = detail
  const error = 'last_error' in src ? src.last_error : src.error
  if (typeof error === 'string' || error === null) patch.last_error = error
  if (typeof src.enabled === 'boolean') patch.enabled = src.enabled
  if (typeof src.label === 'string') patch.label = src.label
  if (typeof src.url === 'string') patch.url = src.url
  if (typeof src.clip_count === 'number') patch.clip_count = src.clip_count
  return patch
}

const STATE_LOOK: Record<LiveStreamState, { label: string; tone: string; pulse?: boolean; help: string }> = {
  watching: { label: 'watching', tone: 'text-signal-good', help: 'Capturing and listening for moments.' },
  clipping: {
    label: 'clipping',
    tone: 'text-sodium-400',
    pulse: true,
    help: 'Judging a candidate moment; capture keeps running.',
  },
  connecting: { label: 'connecting', tone: 'text-ink-200', pulse: true, help: 'Resolving and opening the stream.' },
  reconnecting: {
    label: 'reconnecting',
    tone: 'text-sodium-600',
    pulse: true,
    help: 'The stream dropped; reconnecting with backoff.',
  },
  waiting_for_live: { label: 'waiting', tone: 'text-ink-300', help: 'The channel is offline; checked every 30 s.' },
  off: { label: 'off', tone: 'text-ink-500', help: 'Paused — not watching.' },
  ended: { label: 'ended', tone: 'text-ink-400', help: 'The stream ended.' },
  error: { label: 'error', tone: 'text-signal-bad', help: 'Stopped after repeated failures.' },
}

function StatePill({ stream, forceOff = false }: { stream: LiveStream; forceOff?: boolean }) {
  // A paused row reads "off" whatever its last runtime state was.
  const state: LiveStreamState = forceOff || stream.enabled === false ? 'off' : stream.state
  const look = STATE_LOOK[state] ?? { label: String(state), tone: 'text-ink-400', help: '' }
  return (
    <span
      title={look.help}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap text-[0.6875rem] font-semibold uppercase tracking-[0.14em] ${look.tone}`}
    >
      <span aria-hidden className={`size-1.5 rounded-full bg-current ${look.pulse ? 'animate-pulse' : ''}`} />
      {look.label}
    </span>
  )
}

function FeedIndicator({ state }: { state: LiveFeedState }) {
  const look =
    state === 'open'
      ? { dot: 'bg-signal-good', text: 'live updates' }
      : state === 'retrying'
        ? { dot: 'animate-pulse bg-sodium-600', text: 'reconnecting…' }
        : state === 'connecting'
          ? { dot: 'animate-pulse bg-ink-500', text: 'connecting…' }
          : { dot: 'bg-ink-600', text: 'updates paused' }
  return (
    <span
      className="flex items-center gap-1.5 text-xs text-ink-500"
      title="The live event feed. A 5-second refresh backs it up, so nothing is missed while it reconnects."
    >
      <span aria-hidden className={`size-1.5 rounded-full ${look.dot}`} />
      {look.text}
    </span>
  )
}

function Stat({
  label,
  value,
  tone = 'text-ink-100',
  title,
  children,
}: {
  label: string
  value: string
  tone?: string
  title?: string
  children?: React.ReactNode
}) {
  return (
    <div className="flex items-baseline gap-2 whitespace-nowrap" title={title}>
      <dt className="text-ink-500">{label}</dt>
      <dd className={`numeric ${tone}`}>{value}</dd>
      {children}
    </div>
  )
}

export function StreamsPanel({
  onStreamsChange,
  onStreamRemoved,
  onStatusChange,
  onRequestEnable,
  refreshSignal,
}: {
  /**
   * Every accepted stream list — the clip wall uses it for labels and filters.
   * `complete` is false while Live Watch is off, when a server may list fewer
   * streams than it holds; merge rather than replace then.
   */
  onStreamsChange: (streams: LiveStream[], complete: boolean) => void
  /** A stream was removed here (its clips went with it). */
  onStreamRemoved: (id: string) => void
  /** Every fresh status — the page uses `enabled`. */
  onStatusChange: (status: LiveStatus) => void
  /** "Turn on in Settings": the page switches to Settings and focuses the toggle. */
  onRequestEnable: () => void
  /** Bumped by the page when something outside changed (settings saved). */
  refreshSignal: number
}) {
  const [streams, setStreams] = useState<LiveStream[] | null>(null)
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [listError, setListError] = useState<Error | null>(null)
  const [statusError, setStatusError] = useState<Error | null>(null)
  const [feedState, setFeedState] = useState<LiveFeedState>('paused')
  const [url, setUrl] = useState('')
  const [label, setLabel] = useState('')
  const [urlError, setUrlError] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)
  const [addError, setAddError] = useState<Error | null>(null)
  const [busy, setBusy] = useState<Record<string, 'pause' | 'resume' | 'remove'>>({})
  const [rowErrors, setRowErrors] = useState<Record<string, Error>>({})
  const [confirmId, setConfirmId] = useState<string | null>(null)

  const streamsRef = useRef<LiveStream[] | null>(null)
  const enabledRef = useRef<boolean | null>(null)
  const epochRef = useRef(0)
  const inFlightRef = useRef(false)
  const queuedRef = useRef(false)
  const mountedRef = useRef(true)
  const refreshRef = useRef<() => void>(() => undefined)
  const urlRef = useRef<HTMLInputElement>(null)
  const headingRef = useRef<HTMLHeadingElement>(null)
  const removeButtons = useRef(new Map<string, HTMLButtonElement>())
  const reportStreams = useRef(onStreamsChange)
  const reportStatus = useRef(onStatusChange)
  const reportRemoved = useRef(onStreamRemoved)
  reportStreams.current = onStreamsChange
  reportStatus.current = onStatusChange
  reportRemoved.current = onStreamRemoved

  const ids = { heading: useId(), url: useId(), urlNote: useId(), label: useId() }

  /** Every write funnels through here so the ref mirror and the page never lie. */
  const commitStreams = useCallback((next: LiveStream[]) => {
    streamsRef.current = next
    setStreams(next)
    reportStreams.current(next, enabledRef.current !== false)
  }, [])

  const patchRow = useCallback(
    (id: string, patch: Partial<LiveStream>): boolean => {
      const prev = streamsRef.current
      if (!prev || !prev.some((row) => row.id === id)) return false
      commitStreams(prev.map((row) => (row.id === id ? { ...row, ...patch } : row)))
      return true
    },
    [commitStreams],
  )

  const setRowBusy = (id: string, action: 'pause' | 'resume' | 'remove' | null) =>
    setBusy((prev) => {
      const next = { ...prev }
      if (action) next[id] = action
      else delete next[id]
      return next
    })

  const setRowError = (id: string, error: Error | null) =>
    setRowErrors((prev) => {
      const next = { ...prev }
      if (error) next[id] = error
      else delete next[id]
      return next
    })

  useEffect(() => {
    mountedRef.current = true

    const refresh = async () => {
      if (inFlightRef.current) {
        queuedRef.current = true
        return
      }
      inFlightRef.current = true
      const epoch = epochRef.current
      try {
        const [streamsRes, statusRes] = await Promise.allSettled([
          liveApi.listStreams(),
          liveApi.status(),
        ])
        if (!mountedRef.current) return
        if (statusRes.status === 'fulfilled') {
          enabledRef.current = statusRes.value.enabled
          setStatus(statusRes.value)
          setStatusError(null)
          reportStatus.current(statusRes.value)
        } else {
          setStatusError(asError(statusRes.reason))
        }
        if (epoch !== epochRef.current) {
          // A change landed while this list was in flight, so it may predate
          // it (a removed stream would reappear). Drop it and ask again.
          queuedRef.current = true
        } else if (streamsRes.status === 'fulfilled') {
          commitStreams(streamsRes.value)
          setListError(null)
        } else {
          setListError(asError(streamsRes.reason))
        }
        setLoading(false)
      } finally {
        inFlightRef.current = false
        if (queuedRef.current && mountedRef.current) {
          queuedRef.current = false
          void refresh()
        }
      }
    }
    refreshRef.current = () => void refresh()

    const onEvent = (event: LiveEvent) => {
      if (event.type === 'snapshot') {
        refreshRef.current()
        return
      }
      if (event.type !== 'stream_state') return
      const data = event.data
      const id = typeof data.stream_id === 'string' ? data.stream_id : null
      const patch = streamPatch(data)
      if (id && Object.keys(patch).length > 0 && patchRow(id, patch)) {
        epochRef.current += 1
      } else {
        // An unknown stream (added elsewhere) or a bare notice: resync.
        refreshRef.current()
      }
    }

    const unsubscribe = subscribeLiveEvents(onEvent)
    const unfollow = subscribeLiveFeedState(setFeedState)
    refreshRef.current()
    const timer = window.setInterval(() => {
      if (!document.hidden) refreshRef.current()
    }, POLL_MS)

    return () => {
      mountedRef.current = false
      window.clearInterval(timer)
      unsubscribe()
      unfollow()
    }
  }, [commitStreams, patchRow])

  useEffect(() => {
    if (refreshSignal > 0) refreshRef.current()
  }, [refreshSignal])

  const addStream = async () => {
    const trimmedUrl = url.trim()
    if (!trimmedUrl) {
      setUrlError('Paste the live URL of a channel or stream.')
      urlRef.current?.focus()
      return
    }
    if (/\s/.test(trimmedUrl)) {
      setUrlError('A URL can’t contain spaces.')
      urlRef.current?.focus()
      return
    }
    setUrlError(null)
    setAddError(null)
    setAdding(true)
    try {
      const added = await liveApi.addStream(trimmedUrl, label.trim())
      epochRef.current += 1
      commitStreams([...(streamsRef.current ?? []).filter((row) => row.id !== added.id), added])
      setUrl('')
      setLabel('')
      announce(`Now watching ${streamName(added)}.`)
      urlRef.current?.focus()
    } catch (err) {
      setAddError(asError(err))
      announce(`Couldn’t add the stream: ${errorText(err)}`)
      if (errorStatus(err) === 409) refreshRef.current()
    } finally {
      setAdding(false)
    }
  }

  const toggleWatch = async (row: LiveStream) => {
    if (busy[row.id]) return
    const resume = row.enabled === false
    const name = streamName(row)
    setConfirmId(null)
    setRowBusy(row.id, resume ? 'resume' : 'pause')
    setRowError(row.id, null)
    try {
      const updated = resume ? await liveApi.resumeStream(row.id) : await liveApi.pauseStream(row.id)
      epochRef.current += 1
      patchRow(row.id, {
        ...updated,
        enabled: typeof updated.enabled === 'boolean' ? updated.enabled : resume,
        clip_count: typeof updated.clip_count === 'number' ? updated.clip_count : row.clip_count,
      })
      announce(resume ? `Resumed ${name}.` : `Paused ${name}.`)
    } catch (err) {
      setRowError(row.id, asError(err))
      announce(`Couldn’t ${resume ? 'resume' : 'pause'} ${name}: ${errorText(err)}`)
      if (errorStatus(err) === 409) refreshRef.current()
    } finally {
      setRowBusy(row.id, null)
    }
  }

  const cancelRemove = (id: string) => {
    setConfirmId(null)
    requestAnimationFrame(() => removeButtons.current.get(id)?.focus())
  }

  const removeStream = async (row: LiveStream) => {
    if (busy[row.id]) return
    const name = streamName(row)
    setRowBusy(row.id, 'remove')
    setRowError(row.id, null)
    try {
      await liveApi.removeStream(row.id)
      epochRef.current += 1
      setConfirmId(null)
      commitStreams((streamsRef.current ?? []).filter((r) => r.id !== row.id))
      reportRemoved.current(row.id)
      const count = row.clip_count ?? 0
      announce(count > 0 ? `Removed ${name} and its ${plural(count, 'clip')}.` : `Removed ${name}.`)
      // The row is gone; land somewhere stable instead of on <body>.
      headingRef.current?.focus()
    } catch (err) {
      setConfirmId(null)
      setRowError(row.id, asError(err))
      announce(`Couldn’t remove ${name}: ${errorText(err)}`)
      requestAnimationFrame(() => removeButtons.current.get(row.id)?.focus())
      if (errorStatus(err) === 409) refreshRef.current()
    } finally {
      setRowBusy(row.id, null)
    }
  }

  const off = status?.enabled === false
  // Switched on, but this server process runs without workers: nothing can
  // be watched or changed here, and turning it "on" again wouldn't help.
  const idle = status?.enabled === true && status.running === false
  const frozen = off || idle
  const frozenNote = off ? 'not watched while Live Watch is off' : 'not watched — this server runs without workers'
  const list = streams ?? []

  const header = (
    <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 border-b border-ink-800 pb-2">
      <div className="flex items-baseline gap-3">
        <h2 id={ids.heading} ref={headingRef} tabIndex={-1} className="eyebrow">
          Streams
        </h2>
        {streams && !(frozen && list.length === 0) && (
          <span className="numeric text-xs text-ink-500">{list.length} on the list</span>
        )}
      </div>
      {!off && !idle && <FeedIndicator state={feedState} />}
    </div>
  )

  if (loading) {
    return (
      <section aria-labelledby={ids.heading} aria-busy="true">
        {header}
        <div className="mt-5 h-4 w-80 max-w-full animate-pulse bg-ink-850" />
        <div className="mt-8 space-y-5">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex animate-pulse items-center gap-6 border-b border-ink-850 pb-4">
              <div className="h-9 flex-1 bg-ink-850" />
              <div className="hidden h-4 w-24 bg-ink-850 sm:block" />
              <div className="h-4 w-32 bg-ink-850" />
            </div>
          ))}
        </div>
      </section>
    )
  }

  // Nothing usable came back at all: say why, offer the obvious next step.
  if (!status && !streams) {
    const error = statusError ?? listError
    return (
      <section aria-labelledby={ids.heading}>
        {header}
        <div className="mt-6 max-w-3xl space-y-4">
          {error && <ErrorNote error={error} />}
          <button className="btn btn-ghost" onClick={() => refreshRef.current()}>
            Try again
          </button>
        </div>
      </section>
    )
  }

  const caps = status?.caps
  const queueFull = !!caps && caps.clip_queue_max > 0 && caps.clip_queue_len >= caps.clip_queue_max
  const hasCounts = list.some((row) => typeof row.clip_count === 'number')

  return (
    <section aria-labelledby={ids.heading}>
      {header}

      {off ? (
        <div className="rise mt-6 max-w-2xl border-l-2 border-ink-700 pl-5">
          <p className="font-display text-2xl leading-tight text-ink-200">Live Watch is off.</p>
          <p className="mt-2 text-sm leading-relaxed text-ink-400">
            Nothing is being watched or clipped. Streams on your list stay saved and pick up
            where they left off once it’s back on.
          </p>
          <button className="btn btn-ghost mt-4" onClick={onRequestEnable}>
            Turn on in Settings
          </button>
        </div>
      ) : idle ? (
        <div className="rise mt-6 max-w-2xl border-l-2 border-ink-700 pl-5">
          <p className="font-display text-2xl leading-tight text-ink-200">Live Watch is on, but not running here.</p>
          <p className="mt-2 text-sm leading-relaxed text-ink-400">
            This AutoClip server was started without background workers, so nothing is watched
            or clipped and the list can’t be changed. Restart AutoClip normally to start watching.
          </p>
        </div>
      ) : (
        <>
          <dl className="mt-4 flex flex-wrap items-baseline gap-x-8 gap-y-2 text-xs">
            <Stat
              label="Watchers"
              value={status ? (caps ? `${status.watching}/${caps.max_concurrent_streams}` : String(status.watching)) : '—'}
              tone={status && status.watching > 0 ? 'text-signal-good' : 'text-ink-100'}
              title="Streams with a running watcher — live or waiting to go live — out of the concurrency cap"
            />
            <Stat
              label="Clipping"
              value={status ? String(status.clipping) : '—'}
              tone={status && status.clipping > 0 ? 'text-sodium-400' : 'text-ink-100'}
              title="Streams with a candidate moment in front of the judge"
            />
            {caps && caps.clip_queue_max > 0 && (
              <Stat
                label="Render queue"
                value={`${caps.clip_queue_len}/${caps.clip_queue_max}`}
                tone={queueFull ? 'text-signal-bad' : 'text-ink-100'}
                title="Accepted clips waiting for a render worker. When full, new candidates are dropped with a reason."
              >
                <span aria-hidden className="h-1 w-14 self-center overflow-hidden bg-ink-700">
                  <span
                    className={`block h-full ${queueFull ? 'bg-signal-bad' : 'bg-sodium-500'}`}
                    style={{ width: `${Math.min(100, (caps.clip_queue_len / caps.clip_queue_max) * 100)}%` }}
                  />
                </span>
              </Stat>
            )}
            {caps && (
              <Stat
                label="Waiting to render"
                value={formatMB(caps.disk_used_mb)}
                title="Disk held by clip windows that haven’t rendered yet. New candidates are refused past the pending-work cap."
              />
            )}
            {(statusError || listError) && (
              <div className="flex items-center gap-2 whitespace-nowrap text-signal-bad" title={(statusError ?? listError)?.message}>
                <span aria-hidden className="size-1.5 rounded-full bg-signal-bad" />
                <dt className="sr-only">Sync</dt>
                <dd>sync failed — retrying</dd>
              </div>
            )}
          </dl>

          <form
            className="mt-8"
            noValidate
            onSubmit={(event) => {
              event.preventDefault()
              void addStream()
            }}
          >
            <div className="grid gap-x-6 gap-y-4 sm:grid-cols-[minmax(0,1fr)_14rem_auto] sm:items-end">
              <div>
                <label htmlFor={ids.url} className="eyebrow">
                  Stream URL
                </label>
                <input
                  id={ids.url}
                  ref={urlRef}
                  className="field mt-1 font-mono text-sm"
                  value={url}
                  disabled={adding}
                  spellCheck={false}
                  autoComplete="off"
                  inputMode="url"
                  placeholder="https://www.twitch.tv/channel"
                  aria-invalid={urlError ? true : undefined}
                  aria-describedby={ids.urlNote}
                  onChange={(event) => {
                    setUrl(event.target.value)
                    if (urlError) setUrlError(null)
                  }}
                />
              </div>
              <div>
                <label htmlFor={ids.label} className="eyebrow">
                  Label <span className="tracking-normal normal-case text-ink-600">— optional</span>
                </label>
                <input
                  id={ids.label}
                  className="field mt-1 text-sm"
                  value={label}
                  disabled={adding}
                  maxLength={80}
                  autoComplete="off"
                  placeholder="shown on cards"
                  onChange={(event) => setLabel(event.target.value)}
                />
              </div>
              <button type="submit" className="btn btn-primary" disabled={adding}>
                {adding ? 'Adding…' : 'Watch'}
              </button>
            </div>
            <p
              id={ids.urlNote}
              className={`mt-2 text-xs leading-snug ${urlError ? 'text-signal-bad' : 'text-ink-500'}`}
            >
              {urlError ?? 'Twitch, YouTube, Kick — any live link streamlink can play. Offline channels wait until they go live.'}
            </p>
          </form>

          {addError && (
            <div className="mt-4 max-w-3xl">
              <ErrorNote error={addError} onDismiss={() => setAddError(null)} />
            </div>
          )}
        </>
      )}

      {list.length === 0 ? (
        !frozen && (
          <div className="mt-8 border border-dashed border-ink-700 px-6 py-10 text-center">
            <p className="font-display text-2xl text-ink-200">No streams on the list.</p>
            <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-ink-400">
              Add a live URL above. AutoClip records it into a rolling buffer, listens for the
              spike, and cuts the clip while the moment is still live.
            </p>
          </div>
        )
      ) : (
        <div className="mt-8">
          {frozen && <p className="eyebrow mb-2">On the list</p>}
          <div
            aria-hidden
            className={`hidden gap-x-5 border-b border-ink-800 pb-2 sm:grid ${hasCounts ? ROW_COLUMNS : ROW_COLUMNS_NO_COUNTS}`}
          >
            <span className="eyebrow">Stream</span>
            <span className="eyebrow">State</span>
            <span className="eyebrow">Detail</span>
            {hasCounts && <span className="eyebrow text-right">Clips</span>}
            <span className="eyebrow text-right">{frozen ? '' : 'Actions'}</span>
          </div>
          <ul aria-label="Streams on the list">
            {list.map((row) => (
              <StreamRow
                key={row.id}
                row={row}
                readOnly={frozen}
                readOnlyNote={frozenNote}
                hasCounts={hasCounts}
                busy={busy[row.id] ?? null}
                error={rowErrors[row.id] ?? null}
                confirming={confirmId === row.id}
                removeRef={(el) => {
                  if (el) removeButtons.current.set(row.id, el)
                  else removeButtons.current.delete(row.id)
                }}
                onToggle={() => void toggleWatch(row)}
                onAskRemove={() => setConfirmId(row.id)}
                onCancelRemove={() => cancelRemove(row.id)}
                onRemove={() => void removeStream(row)}
                onDismissError={() => setRowError(row.id, null)}
              />
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}

function StreamRow({
  row,
  readOnly,
  readOnlyNote,
  hasCounts,
  busy,
  error,
  confirming,
  removeRef,
  onToggle,
  onAskRemove,
  onCancelRemove,
  onRemove,
  onDismissError,
}: {
  row: LiveStream
  /** Nothing runs (Live Watch off, or no workers): no actions, no stale telemetry. */
  readOnly: boolean
  readOnlyNote: string
  hasCounts: boolean
  busy: 'pause' | 'resume' | 'remove' | null
  error: Error | null
  confirming: boolean
  removeRef: (el: HTMLButtonElement | null) => void
  onToggle: () => void
  onAskRemove: () => void
  onCancelRemove: () => void
  onRemove: () => void
  onDismissError: () => void
}) {
  const confirmId = useId()
  const name = streamName(row)
  const paused = row.enabled === false
  const count = typeof row.clip_count === 'number' ? row.clip_count : null
  // While nothing runs, the last runtime detail would mislead.
  const detail = readOnly ? readOnlyNote : row.state_detail || (paused ? 'paused' : '')

  return (
    <li className="border-b border-ink-850">
      <div
        className={`grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-5 gap-y-2 py-3.5 ${hasCounts ? ROW_COLUMNS : ROW_COLUMNS_NO_COUNTS}`}
      >
        <div className="min-w-0">
          <p className="truncate text-[0.9375rem] text-ink-100" title={name}>
            {name}
          </p>
          <p className="mt-0.5 truncate font-mono text-xs text-ink-500" title={row.url}>
            {row.url}
          </p>
        </div>

        <div className="justify-self-end sm:justify-self-start">
          <StatePill stream={row} forceOff={readOnly} />
        </div>

        <div className="col-span-2 min-w-0 sm:col-span-1">
          <p className="truncate text-xs text-ink-400" title={detail || undefined}>
            {detail || '—'}
          </p>
          {row.last_error && !readOnly && (
            <p className="mt-0.5 truncate text-xs text-signal-bad" title={row.last_error}>
              {row.last_error}
            </p>
          )}
        </div>

        {hasCounts ? (
          <p className="numeric text-xs text-ink-400 sm:text-right">
            {count !== null ? plural(count, 'clip') : '—'}
          </p>
        ) : (
          <span aria-hidden className="sm:hidden" />
        )}

        <div className="flex items-center justify-end gap-1.5">
          {!readOnly && (
            <>
              <button
                onClick={onToggle}
                disabled={busy !== null}
                aria-label={`${paused ? 'Resume' : 'Pause'} ${name}`}
                className={
                  paused
                    ? 'btn border border-sodium-700 px-3 py-1 text-xs text-sodium-400 hover:bg-sodium-500/10'
                    : 'btn btn-ghost px-3 py-1 text-xs'
                }
              >
                {busy === 'pause' ? 'Pausing…' : busy === 'resume' ? 'Resuming…' : paused ? 'Resume' : 'Pause'}
              </button>
              <button
                ref={removeRef}
                onClick={confirming ? onCancelRemove : onAskRemove}
                disabled={busy !== null}
                aria-label={`Remove ${name}`}
                aria-expanded={confirming}
                aria-controls={confirming ? confirmId : undefined}
                className="btn btn-quiet px-2 py-1 text-xs hover:text-signal-bad"
              >
                {busy === 'remove' ? 'Removing…' : 'Remove'}
              </button>
            </>
          )}
        </div>
      </div>

      {confirming && (
        <div id={confirmId} className="pb-4 sm:flex sm:justify-end">
          <ConfirmInline
            question={
              count !== null
                ? count > 0
                  ? `Remove ${name} and its ${plural(count, 'clip')}?`
                  : `Remove ${name}? It has no clips yet.`
                : `Remove ${name} and all of its clips?`
            }
            detail="Its buffer and clip files are deleted too. This can’t be undone."
            confirmLabel="Remove"
            confirmName={
              count === null
                ? `Yes, remove ${name} and all of its clips`
                : count > 0
                  ? `Yes, remove ${name} and its ${plural(count, 'clip')}`
                  : `Yes, remove ${name}`
            }
            busy={busy === 'remove'}
            onConfirm={onRemove}
            onCancel={onCancelRemove}
            className="sm:max-w-md"
          />
        </div>
      )}

      {error && (
        <div className="pb-4">
          <ErrorNote error={error} onDismiss={onDismissError} />
        </div>
      )}
    </li>
  )
}
