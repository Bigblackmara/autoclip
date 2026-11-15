/**
 * Live Watch — the clip wall.
 *
 * Detection answers continuously while streams are live, so this is a feed,
 * not an index: frames from the shared live feed upsert cards in place, and a
 * 10 s poll backfills whatever a dropped connection missed. A poll that was in
 * flight while something changed is dropped and asked again, so it can never
 * roll a card back. Rejected windows are fetched separately and listed below
 * the wall on request; they never push ready clips off it.
 */

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'

import { formatDuration } from '../api'
import { ErrorNote } from '../components/ErrorNote'
import {
  CLIP_STATUSES,
  PLATFORMS,
  UPLOAD_STATUSES,
  WALL_STATUSES,
  asError,
  describeUploadError,
  errorStatus,
  errorText,
  liveApi,
  platformLabel,
  subscribeLiveEvents,
  type LiveClip,
  type LiveClipStatus,
  type LiveEvent,
  type LivePlatform,
  type LiveStream,
  type LiveUpload,
} from './api'
import {
  absoluteTime,
  describeFailure,
  describeRejection,
  plural,
  relativeTime,
  streamName,
  wasInterrupted,
  wasJudged,
} from './format'
import { ConfirmInline, SpeakerIcon, Spinner, announce, useNow } from './ui'

const CLIP_LIMIT = 100
const REJECTED_LIMIT = 100
const POLL_MS = 10_000
const SOUND_KEY = 'autoclip.live.previewSound'
const IN_PROGRESS: readonly LiveClipStatus[] = ['detecting', 'queued', 'rendering']

// --------------------------------------------------------------------------
// Payload coercion — several publishers, so trust nothing about the shape
// --------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isClipStatus(value: unknown): value is LiveClipStatus {
  return typeof value === 'string' && (CLIP_STATUSES as readonly string[]).includes(value)
}

function isUploadStatus(value: unknown): value is LiveUpload['status'] {
  return typeof value === 'string' && (UPLOAD_STATUSES as readonly string[]).includes(value)
}

const str = (value: unknown, fallback = ''): string => (typeof value === 'string' ? value : fallback)
const num = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0)
const nullableStr = (value: unknown): string | null => (typeof value === 'string' ? value : null)

function clipIdOf(value: Record<string, unknown>): string | null {
  const candidate = value.clip_id ?? value.id
  return typeof candidate === 'string' && candidate ? candidate : null
}

/** A whole clip row, or null when the payload is only a patch. */
function coerceClip(value: unknown): LiveClip | null {
  if (!isRecord(value)) return null
  const id = clipIdOf(value)
  if (!id || !isClipStatus(value.status) || typeof value.created_at !== 'string') return null
  return {
    id,
    stream_id: str(value.stream_id),
    status: value.status,
    title: str(value.title),
    score: num(value.score),
    reason: str(value.reason),
    hook: str(value.hook),
    trigger_start_s: num(value.trigger_start_s),
    trigger_end_s: num(value.trigger_end_s),
    clip_start_s: num(value.clip_start_s),
    clip_end_s: num(value.clip_end_s),
    caption_style: str(value.caption_style),
    ratio: str(value.ratio),
    error: nullableStr(value.error),
    created_at: value.created_at,
    updated_at: str(value.updated_at, value.created_at),
  }
}

function coerceUpload(value: unknown): LiveUpload | null {
  if (!isRecord(value)) return null
  if (typeof value.id !== 'string' || !value.id) return null
  if (typeof value.clip_id !== 'string' || !value.clip_id) return null
  if (value.platform !== 'youtube' && value.platform !== 'instagram') return null
  if (!isUploadStatus(value.status)) return null
  return {
    id: value.id,
    clip_id: value.clip_id,
    platform: value.platform,
    status: value.status,
    upload_url: str(value.upload_url),
    platform_media_id: str(value.platform_media_id),
    error: nullableStr(value.error),
    created_at: str(value.created_at),
    updated_at: str(value.updated_at),
  }
}

/** Patch the well-typed fields a thin `clip_status` frame may carry. */
function mergeClipFields(clip: LiveClip, patch: Record<string, unknown>): LiveClip {
  const number = (key: keyof LiveClip, fallback: number) =>
    typeof patch[key] === 'number' && Number.isFinite(patch[key]) ? (patch[key] as number) : fallback
  return {
    ...clip,
    status: isClipStatus(patch.status) ? patch.status : clip.status,
    title: typeof patch.title === 'string' && patch.title ? patch.title : clip.title,
    score: number('score', clip.score),
    hook: typeof patch.hook === 'string' ? patch.hook : clip.hook,
    reason: typeof patch.reason === 'string' ? patch.reason : clip.reason,
    clip_start_s: number('clip_start_s', clip.clip_start_s),
    clip_end_s: number('clip_end_s', clip.clip_end_s),
    error: typeof patch.error === 'string' || patch.error === null ? (patch.error as string | null) : clip.error,
    updated_at: typeof patch.updated_at === 'string' ? patch.updated_at : clip.updated_at,
  }
}

function newestFirst(a: LiveClip, b: LiveClip): number {
  const ta = Date.parse(a.created_at)
  const tb = Date.parse(b.created_at)
  if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return tb - ta
  return b.id.localeCompare(a.id)
}

function upsertById<T extends { id: string }>(rows: T[], row: T): T[] {
  const index = rows.findIndex((item) => item.id === row.id)
  if (index === -1) return [row, ...rows]
  const next = [...rows]
  next[index] = row
  return next
}

/** The record a card shows for one platform: in flight, else done, else the latest attempt. */
function pickUpload(uploads: LiveUpload[], platform: LivePlatform): LiveUpload | null {
  const mine = uploads
    .filter((upload) => upload.platform === platform)
    .sort((a, b) => (b.updated_at || b.created_at).localeCompare(a.updated_at || a.created_at))
  return (
    mine.find((upload) => upload.status === 'uploading' || upload.status === 'pending') ??
    mine.find((upload) => upload.status === 'done') ??
    mine[0] ??
    null
  )
}

function readSoundPref(): boolean {
  try {
    return window.localStorage.getItem(SOUND_KEY) === '1'
  } catch {
    return false
  }
}

function writeSoundPref(on: boolean) {
  try {
    window.localStorage.setItem(SOUND_KEY, on ? '1' : '0')
  } catch {
    /* Private mode or blocked storage: the preference just won't stick. */
  }
}

const quoted = (clip: LiveClip) => `“${clip.title || 'Untitled clip'}”`

// --------------------------------------------------------------------------
// The panel
// --------------------------------------------------------------------------

export function ClipsPanel({
  streams,
  removedStream,
  minScore,
  liveEnabled,
  onOpenUploadSettings,
}: {
  /** Every stream the page knows of, for labels and the filter chips. */
  streams: LiveStream[]
  /** Set when a stream was removed here; its clips went with it. */
  removedStream: { id: string; seq: number } | null
  /** The judge's bar (settings `judge_min_score`); colours the score badges. */
  minScore: number
  /** From the live status; null until known. Only shapes the empty state. */
  liveEnabled: boolean | null
  /** Jump to Settings → Uploads, for upload errors whose fix lives there. */
  onOpenUploadSettings: () => void
}) {
  const [clips, setClips] = useState<LiveClip[]>([])
  const [uploads, setUploads] = useState<LiveUpload[]>([])
  const [loaded, setLoaded] = useState(false)
  const [loadError, setLoadError] = useState<Error | null>(null)
  const [stale, setStale] = useState(false)
  const [rejectedCapped, setRejectedCapped] = useState(false)
  const [busy, setBusy] = useState<ReadonlySet<string>>(() => new Set())
  const [actionErrors, setActionErrors] = useState<Record<string, string>>({})
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [soundOn, setSoundOn] = useState(readSoundPref)
  const [mediaFailed, setMediaFailed] = useState<Record<string, string>>({})
  const [mediaAttempt, setMediaAttempt] = useState<Record<string, number>>({})
  const [streamFilter, setStreamFilter] = useState('all')
  const [showRejected, setShowRejected] = useState(false)
  const now = useNow(30_000)

  const clipsRef = useRef<LiveClip[]>([])
  const epochRef = useRef(0)
  const inFlightRef = useRef(false)
  const queuedRef = useRef(false)
  const mountedRef = useRef(true)
  const loadedRef = useRef(false)
  const refreshRef = useRef<() => Promise<void>>(async () => undefined)
  const knownStatus = useRef(new Map<string, LiveClipStatus>())
  const primedRef = useRef(false)
  const videos = useRef(new Map<string, HTMLVideoElement>())
  const discardButtons = useRef(new Map<string, HTMLButtonElement>())
  const headingRef = useRef<HTMLHeadingElement>(null)
  const ids = { heading: useId(), rejected: useId(), filter: useId() }

  useEffect(() => {
    clipsRef.current = clips
  }, [clips])

  const setBusyKey = useCallback((key: string, on: boolean) => {
    setBusy((current) => {
      const next = new Set(current)
      if (on) next.add(key)
      else next.delete(key)
      return next
    })
  }, [])

  const setActionError = useCallback((key: string, message: string | null) => {
    setActionErrors((current) => {
      if (message === null && !(key in current)) return current
      const next = { ...current }
      if (message === null) delete next[key]
      else next[key] = message
      return next
    })
  }, [])

  /** Every local change bumps the epoch, so a poll already in flight is dropped. */
  const upsertClip = useCallback((clip: LiveClip) => {
    epochRef.current += 1
    setClips((current) => upsertById(current, clip).sort(newestFirst))
  }, [])

  const patchClip = useCallback((id: string, patch: Record<string, unknown>) => {
    epochRef.current += 1
    setClips((current) => current.map((row) => (row.id === id ? mergeClipFields(row, patch) : row)))
  }, [])

  const upsertUpload = useCallback((upload: LiveUpload) => {
    epochRef.current += 1
    setUploads((current) => upsertById(current, upload))
  }, [])

  const refreshUploads = useCallback(() => {
    liveApi
      .listUploads()
      .then((list) => {
        if (!mountedRef.current) return
        epochRef.current += 1
        setUploads(list.map(coerceUpload).filter((row): row is LiveUpload => row !== null))
      })
      .catch(() => undefined)
  }, [])

  const fetchOne = useCallback(
    (id: string) => {
      liveApi
        .getClip(id)
        .then((row) => {
          const clip = coerceClip(row)
          if (clip && mountedRef.current) upsertClip(clip)
        })
        .catch(() => undefined)
    },
    [upsertClip],
  )

  // Poll + first load. The first failure is a banner; later ones only mark the
  // counts line "reconnecting…" rather than flickering good cards away.
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
        const [wallRes, rejectedRes, uploadsRes] = await Promise.allSettled([
          liveApi.listClips({ limit: CLIP_LIMIT, status: WALL_STATUSES }),
          liveApi.listClips({ limit: REJECTED_LIMIT, status: ['rejected'] }),
          liveApi.listUploads(),
        ])
        if (!mountedRef.current) return
        if (epoch !== epochRef.current) {
          queuedRef.current = true
          return
        }
        if (wallRes.status === 'rejected') {
          if (loadedRef.current) setStale(true)
          else setLoadError(asError(wallRes.reason))
          loadedRef.current = true
          setLoaded(true)
          return
        }
        const wall = wallRes.value.map(coerceClip).filter((c): c is LiveClip => c !== null)
        const rejected =
          rejectedRes.status === 'fulfilled'
            ? rejectedRes.value.map(coerceClip).filter((c): c is LiveClip => c !== null)
            : null
        setClips((current) => {
          // Older servers ignore `status`, so both lists may hold every row:
          // merge by id and let the views split by status.
          const merged = new Map<string, LiveClip>()
          for (const clip of rejected ?? current.filter((c) => c.status === 'rejected')) merged.set(clip.id, clip)
          for (const clip of wall) merged.set(clip.id, clip)
          return [...merged.values()].sort(newestFirst)
        })
        if (rejected) setRejectedCapped(rejected.length >= REJECTED_LIMIT)
        if (uploadsRes.status === 'fulfilled') {
          setUploads(uploadsRes.value.map(coerceUpload).filter((u): u is LiveUpload => u !== null))
        }
        setLoadError(null)
        setStale(false)
        loadedRef.current = true
        setLoaded(true)
      } finally {
        inFlightRef.current = false
        if (queuedRef.current && mountedRef.current) {
          queuedRef.current = false
          void refresh()
        }
      }
    }
    refreshRef.current = refresh

    void refresh()
    const timer = window.setInterval(() => {
      if (!document.hidden) void refresh()
    }, POLL_MS)
    return () => {
      mountedRef.current = false
      window.clearInterval(timer)
    }
  }, [])

  // The shared live feed.
  useEffect(() => {
    const onEvent = (event: LiveEvent) => {
      const payload = event.data
      switch (event.type) {
        case 'snapshot':
          // Every (re)connect opens with one: refetch what the gap missed.
          void refreshRef.current()
          return
        case 'clip':
        case 'clip_status': {
          const full = coerceClip(payload) ?? coerceClip(payload.clip)
          if (full) {
            upsertClip(full)
            return
          }
          const id = clipIdOf(payload)
          if (!id) return
          if (clipsRef.current.some((row) => row.id === id)) patchClip(id, payload)
          else fetchOne(id)
          return
        }
        case 'upload': {
          const full = coerceUpload(payload) ?? coerceUpload(payload.upload)
          if (full) upsertUpload(full)
          else refreshUploads() // a thin frame from the auto-uploader
          return
        }
      }
    }
    return subscribeLiveEvents(onEvent)
  }, [fetchOne, patchClip, refreshUploads, upsertClip, upsertUpload])

  // A removed stream takes its clips with it: drop them now, not at the next poll.
  useEffect(() => {
    if (!removedStream) return
    const { id } = removedStream
    epochRef.current += 1
    setClips((current) => current.filter((clip) => clip.stream_id !== id))
    setStreamFilter((current) => (current === id ? 'all' : current))
  }, [removedStream])

  // Announce clips that turn ready while the page is open (not the first load).
  useEffect(() => {
    if (!loaded) return
    const known = knownStatus.current
    const firstPass = !primedRef.current
    primedRef.current = true
    const fresh: LiveClip[] = []
    for (const clip of clips) {
      const before = known.get(clip.id)
      if (!firstPass && clip.status === 'ready' && before !== 'ready') fresh.push(clip)
      known.set(clip.id, clip.status)
    }
    if (fresh.length === 1) {
      const from = labelFor(streams, fresh[0].stream_id)
      announce(`New clip ready from ${from}: ${fresh[0].title || 'Untitled clip'}.`)
    } else if (fresh.length > 1) {
      announce(`${fresh.length} new clips ready.`)
    }
  }, [clips, loaded, streams])

  useEffect(() => {
    writeSoundPref(soundOn)
  }, [soundOn])

  // ---- actions --------------------------------------------------------------

  const upload = useCallback(
    async (clip: LiveClip, platform: LivePlatform) => {
      const key = `${clip.id}:upload:${platform}`
      const name = platformLabel(platform)
      setActionError(key, null)
      setBusyKey(key, true)
      try {
        const record = coerceUpload(await liveApi.uploadClip(clip.id, platform))
        if (record) upsertUpload(record)
        else refreshUploads()
        announce(`${quoted(clip)} uploaded to ${name}.`)
      } catch (error) {
        const message =
          error instanceof Error ? describeUploadError(error.message, platform).message : errorText(error)
        setActionError(key, message)
        announce(`Upload of ${quoted(clip)} to ${name} failed: ${message}`)
        // The server records refusals and failures as rows; show the latest.
        refreshUploads()
      } finally {
        setBusyKey(key, false)
      }
    },
    [refreshUploads, setActionError, setBusyKey, upsertUpload],
  )

  const discard = useCallback(
    async (clip: LiveClip) => {
      const key = `${clip.id}:discard`
      setActionError(key, null)
      setBusyKey(key, true)
      videos.current.get(clip.id)?.pause()
      try {
        const updated = coerceClip(await liveApi.discardClip(clip.id))
        setConfirmId(null)
        if (updated) upsertClip(updated)
        else patchClip(clip.id, { status: 'rejected', reason: 'discarded by user' })
        announce(`Discarded ${quoted(clip)}. It’s listed under rejected.`)
        // The card leaves the wall; give focus a stable place to land.
        headingRef.current?.focus()
      } catch (error) {
        setConfirmId(null)
        setActionError(key, `Couldn’t discard: ${errorText(error)}`)
        announce(`Couldn’t discard ${quoted(clip)}: ${errorText(error)}`)
        requestAnimationFrame(() => discardButtons.current.get(clip.id)?.focus())
      } finally {
        setBusyKey(key, false)
      }
    },
    [patchClip, setActionError, setBusyKey, upsertClip],
  )

  const retryRender = useCallback(
    async (clip: LiveClip) => {
      const key = `${clip.id}:retry`
      setActionError(key, null)
      setBusyKey(key, true)
      try {
        const updated = coerceClip(await liveApi.retryClip(clip.id))
        if (updated) upsertClip(updated)
        else patchClip(clip.id, { status: 'queued', error: null })
        announce(`${quoted(clip)} is queued to render again.`)
      } catch (error) {
        // FastAPI answers an unknown route with a bare "Not Found".
        const unsupported =
          errorStatus(error) === 405 || (errorStatus(error) === 404 && errorText(error) === 'Not Found')
        const message = unsupported ? 'This server can’t retry renders yet.' : errorText(error)
        setActionError(key, `Couldn’t retry: ${message}`)
        announce(`Couldn’t retry ${quoted(clip)}: ${message}`)
      } finally {
        setBusyKey(key, false)
      }
    },
    [patchClip, setActionError, setBusyKey, upsertClip],
  )

  const cancelDiscard = useCallback((id: string) => {
    setConfirmId(null)
    requestAnimationFrame(() => discardButtons.current.get(id)?.focus())
  }, [])

  const handlePlay = useCallback((id: string) => {
    // One preview at a time — two clips talking over each other helps no one.
    for (const [otherId, video] of videos.current) if (otherId !== id && !video.paused) video.pause()
    setPlayingId(id)
  }, [])

  const handleStop = useCallback((id: string) => {
    setPlayingId((current) => (current === id ? null : current))
  }, [])

  const handleMediaError = useCallback((clip: LiveClip) => {
    // Keyed to the row version: a later update (the file landed) clears it.
    setMediaFailed((current) => ({ ...current, [clip.id]: clip.updated_at }))
  }, [])

  const retryMedia = useCallback((id: string) => {
    setMediaFailed((current) => {
      const next = { ...current }
      delete next[id]
      return next
    })
    setMediaAttempt((current) => ({ ...current, [id]: (current[id] ?? 0) + 1 }))
  }, [])

  const registerVideo = useCallback((id: string, element: HTMLVideoElement | null) => {
    if (element) videos.current.set(id, element)
    else videos.current.delete(id)
  }, [])

  const registerDiscard = useCallback((id: string, element: HTMLButtonElement | null) => {
    if (element) discardButtons.current.set(id, element)
    else discardButtons.current.delete(id)
  }, [])

  // ---- derived --------------------------------------------------------------

  const streamOptions = useMemo(() => {
    const counts = new Map<string, number>()
    for (const clip of clips) {
      if (clip.status !== 'rejected') counts.set(clip.stream_id, (counts.get(clip.stream_id) ?? 0) + 1)
    }
    const order = [...streams.map((stream) => stream.id), ...[...counts.keys()].filter((id) => !streams.some((s) => s.id === id))]
    return order.map((id) => ({ id, label: labelFor(streams, id), count: counts.get(id) ?? 0 }))
  }, [clips, streams])

  const inFilter = (clip: LiveClip) => streamFilter === 'all' || clip.stream_id === streamFilter
  const wall = clips.filter((clip) => clip.status !== 'rejected' && inFilter(clip))
  const rejected = clips.filter((clip) => clip.status === 'rejected' && inFilter(clip))
  const readyCount = wall.filter((clip) => clip.status === 'ready').length
  const workingCount = wall.filter((clip) => IN_PROGRESS.includes(clip.status)).length
  const failedCount = wall.filter((clip) => clip.status === 'failed').length
  const filterName = streamFilter === 'all' ? null : labelFor(streams, streamFilter)
  const rejectedLabel = `${rejected.length}${rejectedCapped && streamFilter === 'all' ? '+' : ''}`

  const summary = [
    plural(readyCount, 'ready clip'),
    workingCount > 0 ? `${workingCount} in progress` : null,
    failedCount > 0 ? `${failedCount} failed` : null,
  ]
    .filter(Boolean)
    .join(' · ')

  return (
    <section aria-labelledby={ids.heading}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 border-b border-ink-800 pb-2">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h2 id={ids.heading} ref={headingRef} tabIndex={-1} className="eyebrow">
            Clips
          </h2>
          {loaded && !loadError && (
            <span className="numeric text-xs text-ink-500">
              {summary}
              {stale && <span className="text-sodium-600"> · reconnecting…</span>}
            </span>
          )}
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-xs text-ink-400">
          <input
            type="checkbox"
            checked={showRejected}
            onChange={(event) => setShowRejected(event.target.checked)}
            className="size-3.5 accent-sodium-500"
          />
          Show rejected ({rejectedLabel})
        </label>
      </div>

      {streamOptions.length > 1 && (
        <div role="group" aria-labelledby={ids.filter} className="mt-4 flex flex-wrap items-center gap-2">
          <span id={ids.filter} className="sr-only">
            Show clips from
          </span>
          <FilterChip active={streamFilter === 'all'} onClick={() => setStreamFilter('all')}>
            All streams
          </FilterChip>
          {streamOptions.map((option) => (
            <FilterChip
              key={option.id}
              active={streamFilter === option.id}
              onClick={() => setStreamFilter(option.id)}
            >
              {option.label}
              <span className="numeric ml-1.5 text-ink-500">{option.count}</span>
            </FilterChip>
          ))}
        </div>
      )}

      {loadError && (
        <div className="mt-6 max-w-3xl space-y-3">
          <ErrorNote error={loadError} />
          <button className="btn btn-ghost" onClick={() => void refreshRef.current()}>
            Try again
          </button>
        </div>
      )}

      {!loaded ? (
        <p className="pt-10 text-sm text-ink-500">Loading clips…</p>
      ) : loadError ? null : wall.length === 0 ? (
        <EmptyWall streamName={filterName} liveEnabled={liveEnabled} />
      ) : (
        <ul
          aria-label={filterName ? `Clips from ${filterName}` : 'Clips'}
          className="mt-8 grid grid-cols-1 gap-x-5 gap-y-10 min-[420px]:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5"
        >
          {wall.map((clip, index) => (
            <li key={clip.id} className="min-w-0">
              <ClipCard
                clip={clip}
                index={index}
                streamLabel={labelFor(streams, clip.stream_id)}
                now={now}
                minScore={minScore}
                uploads={uploads.filter((upload) => upload.clip_id === clip.id)}
                busy={busy}
                actionErrors={actionErrors}
                confirming={confirmId === clip.id}
                playing={playingId === clip.id}
                soundOn={soundOn}
                mediaBroken={clip.status === 'ready' && mediaFailed[clip.id] === clip.updated_at}
                mediaAttempt={mediaAttempt[clip.id] ?? 0}
                onUpload={upload}
                onAskDiscard={setConfirmId}
                onCancelDiscard={cancelDiscard}
                onDiscard={discard}
                onRetryRender={retryRender}
                onPlay={handlePlay}
                onStop={handleStop}
                onToggleSound={() => setSoundOn((current) => !current)}
                onMediaError={handleMediaError}
                onRetryMedia={retryMedia}
                onOpenUploadSettings={onOpenUploadSettings}
                registerVideo={registerVideo}
                registerDiscard={registerDiscard}
              />
            </li>
          ))}
        </ul>
      )}

      {showRejected && loaded && !loadError && (
        <section aria-labelledby={ids.rejected} className="rise mt-16">
          <div className="flex items-baseline justify-between gap-4 border-b border-ink-800 pb-2">
            <h3 id={ids.rejected} className="eyebrow">
              Rejected moments
            </h3>
            <span className="numeric text-xs text-ink-500">{rejectedLabel}</span>
          </div>
          <p className="mt-2 max-w-2xl text-xs leading-relaxed text-ink-500">
            Windows that scored under the bar of {minScore}, never reached the judge, or were
            discarded — kept with their reason, so a quiet stream never looks like a bug.
          </p>
          {rejected.length === 0 ? (
            <p className="pt-6 text-sm text-ink-500">
              Nothing rejected{filterName ? ` from ${filterName}` : ''} yet.
            </p>
          ) : (
            <ul className="mt-2">
              {rejected.map((clip) => (
                <RejectedRow
                  key={clip.id}
                  clip={clip}
                  streamLabel={labelFor(streams, clip.stream_id)}
                  now={now}
                  minScore={minScore}
                />
              ))}
            </ul>
          )}
        </section>
      )}
    </section>
  )
}

function labelFor(streams: LiveStream[], streamId: string): string {
  const stream = streams.find((row) => row.id === streamId)
  if (stream) return streamName(stream)
  return streamId ? `stream ${streamId.slice(0, 6)}` : 'unknown stream'
}

function FilterChip({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      aria-pressed={active}
      onClick={onClick}
      className={`max-w-[16rem] truncate rounded-full border px-3 py-1 text-xs transition-colors duration-200 ${
        active
          ? 'border-sodium-600 bg-sodium-500/10 text-sodium-400'
          : 'border-ink-700 text-ink-400 hover:border-ink-600 hover:text-ink-200'
      }`}
    >
      {children}
    </button>
  )
}

// --------------------------------------------------------------------------
// One card
// --------------------------------------------------------------------------

interface ClipCardProps {
  clip: LiveClip
  index: number
  streamLabel: string
  now: number
  minScore: number
  uploads: LiveUpload[]
  busy: ReadonlySet<string>
  actionErrors: Record<string, string>
  confirming: boolean
  playing: boolean
  soundOn: boolean
  mediaBroken: boolean
  mediaAttempt: number
  onUpload: (clip: LiveClip, platform: LivePlatform) => void
  onAskDiscard: (id: string) => void
  onCancelDiscard: (id: string) => void
  onDiscard: (clip: LiveClip) => void
  onRetryRender: (clip: LiveClip) => void
  onPlay: (id: string) => void
  onStop: (id: string) => void
  onToggleSound: () => void
  onMediaError: (clip: LiveClip) => void
  onRetryMedia: (id: string) => void
  onOpenUploadSettings: () => void
  registerVideo: (id: string, element: HTMLVideoElement | null) => void
  registerDiscard: (id: string, element: HTMLButtonElement | null) => void
}

function ClipCard(props: ClipCardProps) {
  const { clip } = props
  const title = clip.title || (clip.status === 'detecting' ? 'New moment' : 'Untitled clip')
  const ready = clip.status === 'ready'
  const showVideo = ready && !props.mediaBroken
  const durationS = Math.max(0, clip.clip_end_s - clip.clip_start_s)
  const discarding = props.busy.has(`${clip.id}:discard`)
  const retrying = props.busy.has(`${clip.id}:retry`)
  const canDiscard = clip.status === 'ready' || clip.status === 'failed'
  const errors = [props.actionErrors[`${clip.id}:discard`], props.actionErrors[`${clip.id}:retry`]].filter(
    (message): message is string => Boolean(message),
  )

  return (
    <article
      className="group rise"
      style={{ animationDelay: `${Math.min(props.index, 10) * 35}ms` }}
    >
      <div className="relative aspect-[9/16] overflow-hidden rounded-[2px] border border-ink-800 bg-ink-850 transition-colors duration-200 group-hover:border-ink-700">
        {showVideo ? (
          <PreviewVideo
            key={`${clip.id}:${props.mediaAttempt}`}
            clip={clip}
            attempt={props.mediaAttempt}
            soundOn={props.soundOn}
            playing={props.playing}
            onPlay={props.onPlay}
            onStop={props.onStop}
            onError={props.onMediaError}
            register={props.registerVideo}
          />
        ) : props.mediaBroken ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center">
            <p className="text-[0.6875rem] font-semibold uppercase tracking-widest text-ink-400">
              preview unavailable
            </p>
            <p className="text-xs leading-snug text-ink-500">The file may still be moving into place.</p>
            <button
              onClick={() => props.onRetryMedia(clip.id)}
              className="mt-1 text-xs text-sodium-500 underline underline-offset-4 hover:text-sodium-400"
              aria-label={`Try loading ${quoted(clip)} again`}
            >
              Try again
            </button>
          </div>
        ) : (
          <TilePlaceholder clip={clip} />
        )}

        {clip.status !== 'detecting' && clip.score > 0 && (
          <ScoreBadge score={clip.score} minScore={props.minScore} />
        )}
        {durationS > 0 && clip.status !== 'detecting' && (
          <span className="numeric pointer-events-none absolute right-2 bottom-2 rounded-[2px] bg-ink-900/75 px-1.5 py-0.5 text-[0.6875rem] leading-none text-ink-100">
            {formatDuration(durationS)}
          </span>
        )}
        {showVideo && (
          <button
            onClick={props.onToggleSound}
            aria-pressed={props.soundOn}
            aria-label="Preview sound"
            title={props.soundOn ? 'Sound on — click to mute previews' : 'Muted — click to hear previews'}
            className={`absolute bottom-2 left-2 grid size-7 place-items-center rounded-full bg-ink-900/75 transition-colors hover:bg-ink-900 ${
              props.soundOn ? 'text-sodium-400' : 'text-ink-200'
            }`}
          >
            <SpeakerIcon muted={!props.soundOn} />
          </button>
        )}
        {showVideo && !props.playing && <PlayGlyph />}
      </div>

      <div className="mt-3 min-w-0">
        <h3 className="line-clamp-2 text-[0.9375rem] leading-snug text-ink-100" title={title}>
          {title}
        </h3>
        <p className="mt-1 truncate text-xs text-ink-500">
          <span title={props.streamLabel}>{props.streamLabel}</span>
          <span aria-hidden> · </span>
          <time dateTime={clip.created_at} title={absoluteTime(clip.created_at)}>
            {relativeTime(clip.created_at, props.now)}
          </time>
        </p>

        {clip.hook && (
          <p className="mt-2 line-clamp-2 border-l-2 border-ink-700 pl-2.5 font-display text-base italic leading-snug text-ink-300">
            “{clip.hook}”
          </p>
        )}

        {props.confirming ? (
          <ConfirmInline
            question="Discard this clip?"
            detail="It moves to the rejected list."
            confirmLabel="Discard"
            confirmName={`Discard ${quoted(clip)}`}
            cancelLabel="Keep"
            busy={discarding}
            onConfirm={() => props.onDiscard(clip)}
            onCancel={() => props.onCancelDiscard(clip.id)}
            className="mt-3"
          />
        ) : (
          <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
            <StatusLabel status={clip.status} interrupted={clip.status === 'failed' && wasInterrupted(clip.error)} />
            <span className="ml-auto flex items-center gap-3">
              {ready && (
                <a
                  href={liveApi.clipMediaUrl(clip.id)}
                  target="_blank"
                  rel="noreferrer"
                  aria-label={`Open ${quoted(clip)} in a new tab`}
                  className="text-ink-400 transition-colors hover:text-ink-100"
                >
                  open ↗
                </a>
              )}
              {clip.status === 'failed' && (
                <button
                  onClick={() => props.onRetryRender(clip)}
                  disabled={retrying}
                  aria-label={`Retry ${quoted(clip)}`}
                  className="inline-flex items-center gap-1.5 text-sodium-500 transition-colors hover:text-sodium-400 disabled:opacity-50"
                >
                  {retrying && <Spinner />}
                  {retrying ? 'retrying…' : 'retry'}
                </button>
              )}
              {canDiscard && (
                <button
                  ref={(element) => props.registerDiscard(clip.id, element)}
                  onClick={() => props.onAskDiscard(clip.id)}
                  disabled={discarding}
                  aria-label={`Discard ${quoted(clip)}`}
                  className="text-ink-400 transition-colors hover:text-signal-bad disabled:opacity-50"
                >
                  {discarding ? 'discarding…' : 'discard'}
                </button>
              )}
            </span>
          </div>
        )}

        {ready && (
          <UploadRow
            clip={clip}
            uploads={props.uploads}
            busy={props.busy}
            actionErrors={props.actionErrors}
            onUpload={props.onUpload}
            onOpenUploadSettings={props.onOpenUploadSettings}
          />
        )}

        {errors.map((message) => (
          <p key={message} className="mt-2 line-clamp-3 break-words text-xs leading-snug text-signal-bad" title={message}>
            {message}
          </p>
        ))}
      </div>
    </article>
  )
}

function PreviewVideo({
  clip,
  attempt,
  soundOn,
  playing,
  onPlay,
  onStop,
  onError,
  register,
}: {
  clip: LiveClip
  attempt: number
  soundOn: boolean
  playing: boolean
  onPlay: (id: string) => void
  onStop: (id: string) => void
  onError: (clip: LiveClip) => void
  register: (id: string, element: HTMLVideoElement | null) => void
}) {
  const ref = useRef<HTMLVideoElement | null>(null)

  // `muted` is a property React doesn't keep in sync after mount.
  useEffect(() => {
    if (ref.current) ref.current.muted = !soundOn
  }, [soundOn])

  const toggle = () => {
    const video = ref.current
    if (!video) return
    if (video.paused) void video.play().catch(() => undefined)
    else video.pause()
  }

  return (
    <video
      ref={(element) => {
        ref.current = element
        if (element) element.muted = !soundOn
        register(clip.id, element)
      }}
      src={liveApi.clipMediaUrl(clip.id, attempt)}
      preload="metadata"
      playsInline
      role="button"
      tabIndex={0}
      aria-label={`${playing ? 'Pause' : 'Play'} ${quoted(clip)}`}
      className="h-full w-full cursor-pointer object-cover"
      onClick={toggle}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          toggle()
        }
      }}
      onPlay={() => onPlay(clip.id)}
      onPause={() => onStop(clip.id)}
      onEnded={() => onStop(clip.id)}
      onError={() => onError(clip)}
    />
  )
}

/** The tile before media exists, or after a render failed. */
function TilePlaceholder({ clip }: { clip: LiveClip }) {
  if (clip.status === 'failed') {
    const stopped = wasInterrupted(clip.error)
    const why = describeFailure(clip.error)
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center">
        <p
          className={`text-[0.6875rem] font-semibold uppercase tracking-widest ${stopped ? 'text-sodium-600' : 'text-signal-bad'}`}
        >
          {stopped ? 'interrupted' : 'failed'}
        </p>
        {why && (
          <p className="line-clamp-4 break-words text-xs leading-snug text-ink-400" title={clip.error ?? undefined}>
            {why}
          </p>
        )}
      </div>
    )
  }
  const copy =
    clip.status === 'rendering'
      ? 'Rendering — reframe, captions, export'
      : clip.status === 'queued'
        ? 'Waiting for a render worker'
        : 'The judge is weighing this moment'
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-5 text-center">
      <span
        aria-hidden
        className={`h-1 w-20 rounded-full ${
          clip.status === 'rendering'
            ? 'animate-pulse bg-sodium-500'
            : clip.status === 'queued'
              ? 'animate-pulse bg-sodium-700'
              : 'animate-pulse bg-ink-600'
        }`}
      />
      <p className="text-xs leading-snug text-ink-400">{copy}</p>
    </div>
  )
}

function ScoreBadge({ score, minScore }: { score: number; minScore: number }) {
  const tone =
    score >= minScore
      ? 'border-sodium-700 text-sodium-400'
      : score >= minScore - 20
        ? 'border-ink-700 text-ink-100'
        : 'border-ink-700 text-ink-400'
  return (
    <span
      title={`Judge score ${score} — the bar is ${minScore}`}
      className={`numeric pointer-events-none absolute top-2 left-2 rounded-[2px] border bg-ink-900/75 px-1.5 py-0.5 font-display text-sm leading-none ${tone}`}
    >
      <span className="sr-only">Judge score </span>
      {score}
    </span>
  )
}

function PlayGlyph() {
  return (
    <span aria-hidden className="pointer-events-none absolute inset-0 grid place-items-center">
      <span className="grid size-11 place-items-center rounded-full bg-ink-900/60 text-ink-100 transition-transform duration-200 group-hover:scale-105">
        <svg viewBox="0 0 12 14" className="ml-0.5 size-3.5 fill-current">
          <path d="M0 0 L12 7 L0 14 Z" />
        </svg>
      </span>
    </span>
  )
}

const STATUS_LOOK: Record<LiveClipStatus, { label: string; tone: string; pulse: boolean }> = {
  detecting: { label: 'detecting', tone: 'text-ink-300', pulse: true },
  queued: { label: 'queued', tone: 'text-sodium-600', pulse: true },
  rendering: { label: 'rendering', tone: 'text-sodium-500', pulse: true },
  ready: { label: 'ready', tone: 'text-signal-good', pulse: false },
  failed: { label: 'failed', tone: 'text-signal-bad', pulse: false },
  rejected: { label: 'rejected', tone: 'text-ink-500', pulse: false },
}

function StatusLabel({ status, interrupted = false }: { status: LiveClipStatus; interrupted?: boolean }) {
  // Statuses come straight off the wire; one the UI hasn't met degrades to plain text.
  const look = interrupted
    ? { label: 'interrupted', tone: 'text-sodium-600', pulse: false }
    : (STATUS_LOOK[status] ?? { label: String(status), tone: 'text-ink-400', pulse: false })
  return (
    <span
      className={`flex items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-[0.14em] ${look.tone}`}
    >
      {look.pulse && <span aria-hidden className="size-1 animate-pulse rounded-full bg-current" />}
      {look.label}
    </span>
  )
}

// --------------------------------------------------------------------------
// Uploads on a card
// --------------------------------------------------------------------------

function UploadRow({
  clip,
  uploads,
  busy,
  actionErrors,
  onUpload,
  onOpenUploadSettings,
}: {
  clip: LiveClip
  uploads: LiveUpload[]
  busy: ReadonlySet<string>
  actionErrors: Record<string, string>
  onUpload: (clip: LiveClip, platform: LivePlatform) => void
  onOpenUploadSettings: () => void
}) {
  const lines: { platform: LivePlatform; message: string; setup: boolean; tone: 'bad' | 'muted' }[] = []

  const controls = PLATFORMS.map(({ key, label }) => {
    const record = pickUpload(uploads, key)
    const posting = busy.has(`${clip.id}:upload:${key}`)
    const postError = actionErrors[`${clip.id}:upload:${key}`]
    if (!posting) {
      if (postError) {
        lines.push({ platform: key, message: postError, setup: describeUploadError(postError, key).setup, tone: 'bad' })
      } else if (record && (record.status === 'failed' || record.status === 'skipped')) {
        const friendly = describeUploadError(record.error, key)
        lines.push({
          platform: key,
          message: friendly.message,
          setup: friendly.setup,
          tone: record.status === 'failed' ? 'bad' : 'muted',
        })
      }
    }
    return (
      <UploadControl
        key={key}
        clip={clip}
        platform={key}
        label={label}
        record={record}
        posting={posting}
        failedNow={Boolean(postError)}
        onUpload={() => onUpload(clip, key)}
      />
    )
  })

  const needsSetup = lines.some((line) => line.setup)

  return (
    <div className="mt-2.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
        <span className="text-ink-500">Post to</span>
        {controls}
      </div>
      {lines.map((line) => (
        <p
          key={line.platform}
          className={`mt-1.5 line-clamp-3 break-words text-xs leading-snug ${line.tone === 'bad' ? 'text-signal-bad' : 'text-ink-400'}`}
          title={line.message}
        >
          {!line.message.includes(platformLabel(line.platform)) && (
            <span className="text-ink-300">{platformLabel(line.platform)}: </span>
          )}
          {line.message}
        </p>
      ))}
      {needsSetup && (
        <button
          onClick={onOpenUploadSettings}
          className="mt-1.5 text-xs text-sodium-500 underline underline-offset-4 hover:text-sodium-400"
        >
          Open upload settings
        </button>
      )}
    </div>
  )
}

function UploadControl({
  clip,
  platform,
  label,
  record,
  posting,
  failedNow,
  onUpload,
}: {
  clip: LiveClip
  platform: LivePlatform
  label: string
  record: LiveUpload | null
  posting: boolean
  failedNow: boolean
  onUpload: () => void
}) {
  if (posting || record?.status === 'uploading' || record?.status === 'pending') {
    return (
      <span className="inline-flex items-center gap-1.5 text-ink-300">
        <Spinner />
        {label}
        <span className="sr-only"> upload in progress</span>
      </span>
    )
  }
  if (record?.status === 'done' && !failedNow) {
    return record.upload_url ? (
      <a
        href={record.upload_url}
        target="_blank"
        rel="noreferrer"
        aria-label={`Open ${quoted(clip)} on ${label}`}
        className="text-sodium-500 underline underline-offset-4 transition-colors hover:text-sodium-400"
      >
        {label} ↗
      </a>
    ) : (
      <span className="text-signal-good">
        {label} ✓<span className="sr-only"> uploaded</span>
      </span>
    )
  }
  if (failedNow || record?.status === 'failed' || record?.status === 'skipped') {
    const skipped = !failedNow && record?.status === 'skipped'
    const reason = failedNow ? '' : describeUploadError(record?.error, platform).message
    return (
      <span className="inline-flex items-baseline gap-1.5">
        <span className={skipped ? 'text-ink-400' : 'text-signal-bad'} title={reason || undefined}>
          {label} {skipped ? 'skipped' : 'failed'}
        </span>
        <button
          onClick={onUpload}
          aria-label={`Retry uploading ${quoted(clip)} to ${label}`}
          className="text-ink-200 underline underline-offset-2 transition-colors hover:text-sodium-400"
        >
          retry
        </button>
      </span>
    )
  }
  return (
    <button
      onClick={onUpload}
      aria-label={`Upload ${quoted(clip)} to ${label}`}
      title={`Upload to ${label}`}
      className="text-ink-200 transition-colors hover:text-sodium-400"
    >
      {label}
    </button>
  )
}

// --------------------------------------------------------------------------
// Rejected windows and the empty wall
// --------------------------------------------------------------------------

function RejectedRow({
  clip,
  streamLabel,
  now,
  minScore,
}: {
  clip: LiveClip
  streamLabel: string
  now: number
  minScore: number
}) {
  const judged = wasJudged(clip)
  return (
    <li className="grid grid-cols-[2.75rem_minmax(0,1fr)] gap-x-4 border-b border-ink-850 py-3.5 sm:grid-cols-[2.75rem_minmax(0,1fr)_auto]">
      <span
        className="numeric self-start font-display text-2xl leading-none text-ink-500"
        title={judged ? `Judge score ${clip.score}` : 'Not judged'}
      >
        {judged ? clip.score : '—'}
        <span className="sr-only">{judged ? ' points' : ' not judged'}</span>
      </span>
      <div className="min-w-0">
        <p className="truncate text-sm text-ink-200">{clip.title || 'Untitled moment'}</p>
        <p className="mt-0.5 text-xs leading-snug text-ink-400">{describeRejection(clip, minScore)}</p>
        <p className="mt-1 truncate text-xs text-ink-500 sm:hidden">
          {streamLabel} · {relativeTime(clip.created_at, now)}
        </p>
      </div>
      <p className="hidden text-right text-xs text-ink-500 sm:block">
        <span className="block max-w-[14rem] truncate" title={streamLabel}>
          {streamLabel}
        </span>
        <time dateTime={clip.created_at} title={absoluteTime(clip.created_at)}>
          {relativeTime(clip.created_at, now)}
        </time>
      </p>
    </li>
  )
}

function EmptyWall({ streamName: name, liveEnabled }: { streamName: string | null; liveEnabled: boolean | null }) {
  if (name) {
    return (
      <div className="max-w-xl pt-12 pb-4">
        <p className="font-display text-2xl leading-tight text-ink-200">Nothing from {name} yet.</p>
        <p className="mt-3 text-sm leading-relaxed text-ink-400">
          Its clips land here as soon as the judge accepts a moment.
        </p>
      </div>
    )
  }
  return (
    <div className="max-w-xl pt-12 pb-4">
      <p className="font-display text-3xl leading-tight text-ink-200">No clips yet.</p>
      {liveEnabled === false ? (
        <p className="mt-4 text-sm leading-relaxed text-ink-400">
          Live Watch is off. Once it’s on and a stream on your list goes live, moments the
          judge accepts are cut and land here on their own.
        </p>
      ) : (
        <>
          <p className="mt-4 text-sm leading-relaxed text-ink-400">
            Detection runs while your streams are live — when a moment spikes and the judge
            agrees, the clip is cut and lands here on its own.
          </p>
          <p className="mt-3 text-sm leading-relaxed text-ink-400">
            Keep this page open: cards appear as moments are detected and fill in as they render.
          </p>
        </>
      )}
    </div>
  )
}
