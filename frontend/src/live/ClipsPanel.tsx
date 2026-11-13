/**
 * Live Watch — the clip wall.
 *
 * Detection answers continuously while streams are live, so this gallery is a
 * feed rather than an index: new clips ride in over SSE ('clip'), state
 * changes land in place ('clip_status'), and a 10s poll backfills whatever a
 * dropped connection missed. Everything the operator needs to act on a clip —
 * preview, upload, open, discard — lives on its card; nothing navigates away.
 *
 * Self-contained by work order: imports only react and the live API client.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  liveApi,
  subscribeLiveEvents,
  type LiveClip,
  type LiveClipStatus,
  type LiveEvent,
  type LiveUpload,
} from '../live/api'

/** How many clips the wall asks for in one go. */
const CLIP_LIMIT = 100

/** Poll safety net under the SSE feed. */
const POLL_MS = 10_000

type Platform = 'youtube' | 'instagram'

const PLATFORMS: { key: Platform; label: string }[] = [
  { key: 'youtube', label: 'YouTube' },
  { key: 'instagram', label: 'Instagram' },
]

const CLIP_STATUSES: readonly LiveClipStatus[] = [
  'detecting',
  'queued',
  'rendering',
  'ready',
  'failed',
  'rejected',
]

function isClipStatus(value: unknown): value is LiveClipStatus {
  return typeof value === 'string' && CLIP_STATUSES.some((status) => status === value)
}

const UPLOAD_STATUSES: readonly LiveUpload['status'][] = [
  'pending',
  'uploading',
  'done',
  'failed',
  'skipped',
]

function isUploadStatus(value: unknown): value is LiveUpload['status'] {
  return typeof value === 'string' && UPLOAD_STATUSES.some((status) => status === value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function asString(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function asNumber(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function asNullableString(value: unknown): string | null {
  return value === null ? null : asString(value)
}

function clipIdOf(value: Record<string, unknown>): string | null {
  const candidate = value.clip_id ?? value.id
  return typeof candidate === 'string' && candidate ? candidate : null
}

/**
 * Normalize an SSE/REST payload into a full LiveClip. Several hands publish
 * events; anything that isn't a whole row returns null and the caller falls
 * back to either a targeted getClip or a deeper check — never a broken card.
 */
function coerceClip(value: unknown): LiveClip | null {
  if (!isRecord(value)) return null
  const id = clipIdOf(value)
  if (!id || !isClipStatus(value.status)) return null
  return {
    id,
    stream_id: asString(value.stream_id),
    status: value.status,
    title: asString(value.title),
    score: asNumber(value.score),
    reason: asString(value.reason),
    hook: asString(value.hook),
    trigger_start_s: asNumber(value.trigger_start_s),
    trigger_end_s: asNumber(value.trigger_end_s),
    clip_start_s: asNumber(value.clip_start_s),
    clip_end_s: asNumber(value.clip_end_s),
    caption_style: asString(value.caption_style),
    ratio: asString(value.ratio),
    error: asNullableString(value.error),
    created_at: asString(value.created_at),
    updated_at: asString(value.updated_at),
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
    upload_url: asString(value.upload_url),
    platform_media_id: asString(value.platform_media_id),
    error: asNullableString(value.error),
    created_at: asString(value.created_at),
    updated_at: asString(value.updated_at),
  }
}

/** Patch the well-typed fields a thin 'clip_status' payload may carry. */
function mergeClipFields(clip: LiveClip, patch: Record<string, unknown>): LiveClip {
  return {
    ...clip,
    status: isClipStatus(patch.status) ? patch.status : clip.status,
    title: typeof patch.title === 'string' ? patch.title : clip.title,
    score:
      typeof patch.score === 'number' && Number.isFinite(patch.score) ? patch.score : clip.score,
    hook: typeof patch.hook === 'string' ? patch.hook : clip.hook,
    reason: typeof patch.reason === 'string' ? patch.reason : clip.reason,
    clip_start_s:
      typeof patch.clip_start_s === 'number' && Number.isFinite(patch.clip_start_s)
        ? patch.clip_start_s
        : clip.clip_start_s,
    clip_end_s:
      typeof patch.clip_end_s === 'number' && Number.isFinite(patch.clip_end_s)
        ? patch.clip_end_s
        : clip.clip_end_s,
    error: typeof patch.error === 'string' || patch.error === null ? patch.error : clip.error,
    updated_at: typeof patch.updated_at === 'string' ? patch.updated_at : clip.updated_at,
  }
}

function newestFirst(a: LiveClip, b: LiveClip): number {
  const ta = Date.parse(a.created_at)
  const tb = Date.parse(b.created_at)
  if (Number.isFinite(ta) && Number.isFinite(tb) && ta !== tb) return tb - ta
  return b.created_at.localeCompare(a.created_at)
}

function sortedNewestFirst(clips: LiveClip[]): LiveClip[] {
  return [...clips].sort(newestFirst)
}

function describeError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message
  if (isRecord(error) && typeof error.message === 'string') return error.message
  return String(error)
}

/** ApiError carries the actionable fix on `.hint`; primitives carry nothing. */
function errorHint(error: unknown): string {
  return isRecord(error) && typeof error.hint === 'string' && error.hint ? error.hint : ''
}

function describeActionError(error: unknown): string {
  const hint = errorHint(error)
  const message = describeError(error)
  return hint ? `${message} — ${hint}` : message
}

function addToSet(set: ReadonlySet<string>, value: string): Set<string> {
  const next = new Set(set)
  next.add(value)
  return next
}

function removeFromSet(set: ReadonlySet<string>, value: string): Set<string> {
  const next = new Set(set)
  next.delete(value)
  return next
}

/** m:ss — live clips are short; the hour case never arrives here. */
function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

/** Prefer a completed record so a retried upload never hides the live link. */
function pickUpload(uploads: LiveUpload[], platform: Platform): LiveUpload | null {
  const mine = uploads.filter((upload) => upload.platform === platform)
  const done = mine.find((upload) => upload.status === 'done')
  if (done) return done
  return [...mine].sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0] ?? null
}

function toggleVideo(video: HTMLVideoElement) {
  if (video.paused) void video.play().catch(() => undefined)
  else video.pause()
}

export function ClipsPanel() {
  const [clips, setClips] = useState<LiveClip[]>([])
  const [uploads, setUploads] = useState<LiveUpload[]>([])
  const [loaded, setLoaded] = useState(false)
  const [stale, setStale] = useState(false)
  const [loadError, setLoadError] = useState<unknown>(null)
  const [uploading, setUploading] = useState<ReadonlySet<string>>(() => new Set<string>())
  const [uploadErrors, setUploadErrors] = useState<Record<string, string>>({})
  const [discarding, setDiscarding] = useState<ReadonlySet<string>>(() => new Set<string>())
  const [discardErrors, setDiscardErrors] = useState<Record<string, string>>({})
  const [confirmId, setConfirmId] = useState<string | null>(null)
  const [playingIds, setPlayingIds] = useState<ReadonlySet<string>>(() => new Set<string>())
  const [mediaFailed, setMediaFailed] = useState<ReadonlySet<string>>(() => new Set<string>())

  // Ref mirrors let the SSE handler make synchronous "do we have this row?"
  // decisions without re-subscribing on every render.
  const clipsRef = useRef<LiveClip[]>([])
  const loadedRef = useRef(false)

  useEffect(() => {
    clipsRef.current = clips
  }, [clips])

  const upsertClip = useCallback((clip: LiveClip) => {
    setClips((current) => {
      const index = current.findIndex((row) => row.id === clip.id)
      if (index === -1) return [clip, ...current]
      const next = [...current]
      next[index] = clip
      return next
    })
    if (clip.status === 'ready') setMediaFailed((current) => removeFromSet(current, clip.id))
  }, [])

  const upsertUpload = useCallback((upload: LiveUpload) => {
    setUploads((current) => {
      const index = current.findIndex((row) => row.id === upload.id)
      if (index === -1) return [...current, upload]
      const next = [...current]
      next[index] = upload
      return next
    })
  }, [])

  const refetchClips = useCallback(() => {
    liveApi
      .listClips(CLIP_LIMIT)
      .then((list) => setClips(sortedNewestFirst(list)))
      .catch(() => undefined)
  }, [])

  // Initial load doubles as the 10s SSE fallback. First failure surfaces a
  // banner; later ones dim the counts line to "reconnecting…" rather than
  // flickering good data away.
  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [clipList, uploadList] = await Promise.all([
          liveApi.listClips(CLIP_LIMIT),
          liveApi.listUploads(),
        ])
        if (cancelled) return
        setClips(sortedNewestFirst(clipList))
        setUploads(uploadList)
        setStale(false)
        setLoadError(null)
        if (!loadedRef.current) {
          loadedRef.current = true
          setLoaded(true)
        }
      } catch (error) {
        if (cancelled) return
        if (!loadedRef.current) {
          loadedRef.current = true
          setLoaded(true)
          setLoadError(error)
        } else {
          setStale(true)
        }
      }
    }

    void load()
    const timer = window.setInterval(() => void load(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  const handleEvent = useCallback(
    (event: LiveEvent) => {
      if (event.type === 'clip' || event.type === 'clip_status') {
        const payload = event.data
        const full = coerceClip(payload) ?? coerceClip(isRecord(payload) ? payload.clip : undefined)
        if (full) return upsertClip(full)

        // A thin payload: patch the row in place when we hold it, otherwise
        // pull the whole clip so the card can appear at all.
        const clipId = isRecord(payload) ? clipIdOf(payload) : null
        const id = clipId ?? (typeof event.clip_id === 'string' && event.clip_id ? event.clip_id : null)
        if (!id) return
        if (clipsRef.current.some((row) => row.id === id)) {
          const patch = isRecord(payload) ? payload : {}
          setClips((current) =>
            current.map((row) => (row.id === id ? mergeClipFields(row, patch) : row)),
          )
          if (isClipStatus(patch.status) && patch.status === 'ready') {
            setMediaFailed((current) => removeFromSet(current, id))
          }
        } else {
          liveApi
            .getClip(id)
            .then(upsertClip)
            .catch(() => undefined)
        }
        return
      }

      if (event.type === 'upload') {
        const payload = event.data
        const full =
          coerceUpload(payload) ?? coerceUpload(isRecord(payload) ? payload.upload : undefined)
        if (full) upsertUpload(full)
        return
      }

      // Snapshots bring their own truth; tolerate either or both lists.
      if (event.type === 'snapshot' && isRecord(event.data)) {
        const clipList = event.data.clips
        if (Array.isArray(clipList)) {
          const coerced = clipList.map(coerceClip).filter((clip): clip is LiveClip => clip !== null)
          setClips(sortedNewestFirst(coerced))
        }
        const uploadList = event.data.uploads
        if (Array.isArray(uploadList)) {
          const coerced = uploadList
            .map(coerceUpload)
            .filter((upload): upload is LiveUpload => upload !== null)
          setUploads(coerced)
        }
      }
    },
    [upsertClip, upsertUpload],
  )

  useEffect(() => {
    const unsubscribe = subscribeLiveEvents(handleEvent)
    return () => unsubscribe()
  }, [handleEvent])

  const upload = useCallback(
    async (clip: LiveClip, platform: Platform) => {
      const key = `${clip.id}:${platform}`
      setUploadErrors((current) => {
        if (!(key in current)) return current
        const next = { ...current }
        delete next[key]
        return next
      })
      setUploading((current) => addToSet(current, key))
      try {
        upsertUpload(await liveApi.uploadClip(clip.id, platform))
      } catch (error) {
        setUploadErrors((current) => ({ ...current, [key]: describeActionError(error) }))
      } finally {
        setUploading((current) => removeFromSet(current, key))
      }
    },
    [upsertUpload],
  )

  const discard = useCallback(
    async (clip: LiveClip) => {
      setConfirmId(null)
      setDiscardErrors((current) => {
        if (!(clip.id in current)) return current
        const next = { ...current }
        delete next[clip.id]
        return next
      })
      setDiscarding((current) => addToSet(current, clip.id))
      setPlayingIds((current) => removeFromSet(current, clip.id))
      try {
        const updated = coerceClip(await liveApi.discardClip(clip.id))
        if (updated) upsertClip(updated)
        else refetchClips()
      } catch (error) {
        setDiscardErrors((current) => ({ ...current, [clip.id]: describeActionError(error) }))
      } finally {
        setDiscarding((current) => removeFromSet(current, clip.id))
      }
    },
    [refetchClips, upsertClip],
  )

  const askDiscard = useCallback((id: string) => setConfirmId(id), [])
  const cancelDiscard = useCallback(() => setConfirmId(null), [])

  const handlePlayingChange = useCallback((id: string, playing: boolean) => {
    setPlayingIds((current) => (playing ? addToSet(current, id) : removeFromSet(current, id)))
  }, [])

  const handleMediaError = useCallback((id: string) => {
    setMediaFailed((current) => addToSet(current, id))
  }, [])

  const readyCount = clips.filter((clip) => clip.status === 'ready').length

  return (
    <section aria-label="Live clips" className="pt-8">
      <RightsNotice />

      <div className="mt-8 flex items-baseline justify-between gap-4 border-b border-ink-800 pb-3">
        <h2 className="eyebrow">Clips</h2>
        <p className="numeric text-xs text-ink-400">
          {clips.length} {clips.length === 1 ? 'clip' : 'clips'}
          {readyCount > 0 && ` · ${readyCount} ready`}
          {stale && ' · reconnecting…'}
        </p>
      </div>

      {loadError !== null && (
        <div className="mt-6">
          <ErrorBanner error={loadError} onDismiss={() => setLoadError(null)} />
        </div>
      )}

      {!loaded && clips.length === 0 ? (
        <p className="pt-12 text-sm text-ink-400">Loading clips…</p>
      ) : clips.length === 0 ? (
        <EmptyWall />
      ) : (
        <div className="mt-8 grid grid-cols-2 gap-x-5 gap-y-10 md:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
          {clips.map((clip, index) => (
            <ClipCard
              key={clip.id}
              clip={clip}
              index={index}
              uploads={uploads}
              busyUploads={uploading}
              uploadErrors={uploadErrors}
              discardErrors={discardErrors}
              discarding={discarding}
              confirmId={confirmId}
              playing={playingIds}
              mediaFailed={mediaFailed}
              onUpload={upload}
              onAskDiscard={askDiscard}
              onCancelDiscard={cancelDiscard}
              onDiscard={discard}
              onPlayingChange={handlePlayingChange}
              onMediaError={handleMediaError}
            />
          ))}
        </div>
      )}
    </section>
  )
}

/** Preview tiles before media exists, and after it failed to load. */
function TilePlaceholder({ clip }: { clip: LiveClip }) {
  if (clip.status === 'failed') {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center">
        <p className="text-[0.6875rem] font-semibold uppercase tracking-widest text-signal-bad">
          render failed
        </p>
        {clip.error && (
          <p className="line-clamp-3 text-xs leading-snug text-ink-400" title={clip.error}>
            {clip.error}
          </p>
        )}
      </div>
    )
  }
  if (clip.status === 'rejected') {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center">
        <p className="text-[0.6875rem] font-semibold uppercase tracking-widest text-ink-600">
          not clipped
        </p>
        <p className="text-xs leading-snug text-ink-600">
          the judge scored this moment below the bar
        </p>
      </div>
    )
  }
  const copy =
    clip.status === 'rendering'
      ? 'rendering the clip'
      : clip.status === 'queued'
        ? 'queued for a worker'
        : 'capturing the moment'
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-5 text-center">
      <span
        aria-hidden
        className={`h-1 w-20 rounded-full ${
          clip.status === 'detecting' ? 'bg-ink-600' : 'animate-pulse bg-sodium-700'
        }`}
      />
      <p className="text-xs leading-snug text-ink-400">{copy}</p>
    </div>
  )
}

interface ClipCardProps {
  clip: LiveClip
  index: number
  uploads: LiveUpload[]
  busyUploads: ReadonlySet<string>
  uploadErrors: Record<string, string>
  discardErrors: Record<string, string>
  discarding: ReadonlySet<string>
  confirmId: string | null
  playing: ReadonlySet<string>
  mediaFailed: ReadonlySet<string>
  onUpload: (clip: LiveClip, platform: Platform) => void
  onAskDiscard: (id: string) => void
  onCancelDiscard: () => void
  onDiscard: (clip: LiveClip) => void
  onPlayingChange: (id: string, playing: boolean) => void
  onMediaError: (id: string) => void
}

function ClipCard(props: ClipCardProps) {
  const { clip, index } = props
  const ready = clip.status === 'ready'
  const mediaBroken = ready && props.mediaFailed.has(clip.id)
  const showVideo = ready && !mediaBroken
  const isPlaying = props.playing.has(clip.id)
  const durationS = Math.max(0, clip.clip_end_s - clip.clip_start_s)
  const mine = props.uploads.filter((upload) => upload.clip_id === clip.id)
  const uploadError =
    PLATFORMS.map((platform) => props.uploadErrors[`${clip.id}:${platform.key}`]).find(Boolean) ??
    null
  const discardError = props.discardErrors[clip.id] ?? null
  const dropping = props.discarding.has(clip.id)
  const asking = props.confirmId === clip.id

  return (
    <article
      className={`group rise min-w-0 ${clip.status === 'rejected' ? 'opacity-50' : ''}`}
      style={{ animationDelay: `${Math.min(index, 10) * 35}ms` }}
    >
      <div className="relative aspect-[9/16] overflow-hidden rounded-[2px] border border-ink-800 bg-ink-850 transition-colors duration-200 group-hover:border-ink-700">
        {showVideo ? (
          <video
            src={liveApi.clipMediaUrl(clip.id)}
            preload="metadata"
            muted
            playsInline
            role="button"
            tabIndex={0}
            aria-label={`${isPlaying ? 'Pause' : 'Play'} ${clip.title || 'clip'}`}
            className="h-full w-full cursor-pointer object-cover"
            onClick={(event) => toggleVideo(event.currentTarget)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                toggleVideo(event.currentTarget)
              }
            }}
            onPlay={() => props.onPlayingChange(clip.id, true)}
            onPause={() => props.onPlayingChange(clip.id, false)}
            onEnded={() => props.onPlayingChange(clip.id, false)}
            onError={() => props.onMediaError(clip.id)}
          />
        ) : mediaBroken ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-5 text-center">
            <p className="text-[0.6875rem] font-semibold uppercase tracking-widest text-ink-600">
              media unavailable
            </p>
            <p className="text-xs leading-snug text-ink-600">
              the file may still be moving into place — try opening it
            </p>
          </div>
        ) : (
          <TilePlaceholder clip={clip} />
        )}

        <ScoreBadge score={clip.score} />
        {durationS > 0 && (
          <span className="numeric absolute bottom-2 right-2 rounded-[2px] bg-ink-900/70 px-1.5 py-0.5 text-[0.6875rem] leading-none text-ink-100">
            {formatDuration(durationS)}
          </span>
        )}
        {showVideo && !isPlaying && <PlayGlyph />}
      </div>

      <div className="mt-3 min-w-0">
        <p className="line-clamp-2 text-[0.9375rem] leading-snug text-ink-100" title={clip.title || undefined}>
          {clip.title || 'Untitled clip'}
        </p>

        {clip.hook && (
          <p className="mt-1.5 line-clamp-2 border-l-2 border-ink-700 pl-2.5 font-display text-base italic leading-snug text-ink-300">
            “{clip.hook}”
          </p>
        )}

        <div className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
          <StatusPill status={clip.status} />
          <span className="ml-auto flex items-center gap-3">
            {showVideo ? (
              <a
                href={liveApi.clipMediaUrl(clip.id)}
                target="_blank"
                rel="noreferrer"
                title="Open in a new tab"
                className="text-ink-400 transition-colors hover:text-ink-100"
              >
                open ↗
              </a>
            ) : (
              <span className="text-ink-600" title="Media exists once the clip is ready">
                open ↗
              </span>
            )}

            {dropping ? (
              <span className="text-signal-bad">dropping…</span>
            ) : asking ? (
              <span className="flex items-center gap-2">
                <span className="text-signal-bad">drop it?</span>
                <button
                  onClick={() => props.onDiscard(clip)}
                  className="text-signal-bad underline underline-offset-2"
                >
                  yes
                </button>
                <button
                  onClick={props.onCancelDiscard}
                  className="text-ink-400 transition-colors hover:text-ink-100"
                >
                  no
                </button>
              </span>
            ) : (
              <button
                onClick={() => props.onAskDiscard(clip.id)}
                title="Drop this clip without rendering"
                className="text-ink-400 transition-colors hover:text-signal-bad"
              >
                discard
              </button>
            )}
          </span>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {PLATFORMS.map((platform) => {
            const record = pickUpload(mine, platform.key)
            const busy =
              props.busyUploads.has(`${clip.id}:${platform.key}`) ||
              record?.status === 'uploading' ||
              record?.status === 'pending'
            return (
              <UploadControl
                key={platform.key}
                label={platform.label}
                record={record}
                busy={busy}
                canUpload={ready && !busy}
                onClick={() => props.onUpload(clip, platform.key)}
              />
            )
          })}
        </div>

        {uploadError && (
          <p className="mt-2 line-clamp-2 break-words text-xs leading-snug text-signal-bad" title={uploadError}>
            {uploadError}
          </p>
        )}
        {discardError && (
          <p className="mt-2 line-clamp-2 break-words text-xs leading-snug text-signal-bad" title={discardError}>
            {discardError}
          </p>
        )}
      </div>
    </article>
  )
}

function ScoreBadge({ score }: { score: number }) {
  const tone =
    score >= 70
      ? 'border-sodium-700 text-sodium-400'
      : score >= 50
        ? 'border-ink-700 text-ink-100'
        : 'border-ink-700 text-ink-400'
  return (
    <span
      title={`Judge score ${score}`}
      className={`numeric absolute left-2 top-2 rounded-[2px] border bg-ink-900/70 px-1.5 py-0.5 font-display text-sm leading-none ${tone}`}
    >
      {score}
    </span>
  )
}

function PlayGlyph() {
  return (
    <span className="pointer-events-none absolute inset-0 grid place-items-center">
      <span className="grid h-11 w-11 place-items-center rounded-full bg-ink-900/60 text-ink-100 transition-transform duration-200 group-hover:scale-105">
        <svg viewBox="0 0 12 14" aria-hidden className="ml-0.5 h-3.5 w-3.5 fill-current">
          <path d="M0 0 L12 7 L0 14 Z" />
        </svg>
      </span>
    </span>
  )
}

const STATUS_PRESENTATION: Record<LiveClipStatus, { label: string; className: string; pulse: boolean }> = {
  detecting: { label: 'detecting', className: 'text-ink-300', pulse: true },
  queued: { label: 'queued', className: 'text-sodium-600', pulse: true },
  rendering: { label: 'rendering', className: 'text-sodium-500', pulse: true },
  ready: { label: 'ready', className: 'text-signal-good', pulse: false },
  failed: { label: 'failed', className: 'text-signal-bad', pulse: false },
  rejected: { label: 'rejected', className: 'text-ink-600', pulse: false },
}

function StatusPill({ status }: { status: LiveClipStatus }) {
  // Statuses come straight off the wire; if the backend ever grows one the
  // UI hasn't met, degrade to plain muted text instead of crashing.
  const present = STATUS_PRESENTATION[status] ?? {
    label: String(status),
    className: 'text-ink-400',
    pulse: false,
  }
  return (
    <span
      className={`flex items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-widest ${present.className}`}
    >
      {present.pulse && <span aria-hidden className="h-1 w-1 animate-pulse rounded-full bg-current" />}
      {present.label}
    </span>
  )
}

function UploadControl({
  label,
  record,
  busy,
  canUpload,
  onClick,
}: {
  label: string
  record: LiveUpload | null
  busy: boolean
  canUpload: boolean
  onClick: () => void
}) {
  if (busy) {
    return (
      <span className="flex items-center gap-1.5 text-ink-300">
        <Spinner />
        {label}…
      </span>
    )
  }
  if (record?.status === 'done') {
    return record.upload_url ? (
      <a
        href={record.upload_url}
        target="_blank"
        rel="noreferrer"
        className="text-sodium-500 underline underline-offset-4 transition-colors hover:text-sodium-400"
      >
        {label} ↗
      </a>
    ) : (
      <span className="text-signal-good" title="Uploaded">
        {label} ✓
      </span>
    )
  }
  if (record?.status === 'failed') {
    return (
      <span className="text-signal-bad" title={record.error || 'Upload failed'}>
        {label} failed
      </span>
    )
  }
  if (record?.status === 'skipped') {
    return (
      <span className="text-ink-600" title={record.error || 'Skipped'}>
        {label} skipped
      </span>
    )
  }
  return (
    <button
      onClick={onClick}
      disabled={!canUpload}
      title={canUpload ? `Upload to ${label}` : 'Clip must be ready before upload'}
      className="text-ink-400 transition-colors hover:text-ink-100 disabled:text-ink-600"
    >
      {label}
    </button>
  )
}

function Spinner() {
  return (
    <span
      aria-hidden
      className="inline-block h-3 w-3 animate-spin rounded-full border border-ink-600 border-t-sodium-500"
    />
  )
}

function EmptyWall() {
  return (
    <div className="max-w-xl pb-6 pt-14">
      <h3 className="font-display text-3xl leading-tight text-ink-200">No clips yet.</h3>
      <p className="mt-4 text-sm leading-relaxed text-ink-400">
        Detection runs while streams are live — when a moment spikes and the judge
        agrees, the clip lands here on its own.
      </p>
      <p className="mt-3 text-sm leading-relaxed text-ink-400">
        Leave this page open: new cards appear as they are detected, and what the live
        feed misses a ten-second refresh brings in.
      </p>
    </div>
  )
}

function ErrorBanner({ error, onDismiss }: { error: unknown; onDismiss: () => void }) {
  const hint = errorHint(error)
  return (
    <div className="border-l-2 border-signal-bad pl-4">
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm font-medium text-ink-100">{describeError(error)}</p>
        <button onClick={onDismiss} aria-label="Dismiss" className="btn btn-quiet shrink-0">
          ✕
        </button>
      </div>
      {hint && (
        <p className="mt-2 max-w-prose whitespace-pre-line text-sm leading-relaxed text-ink-400">
          {hint}
        </p>
      )}
    </div>
  )
}

function RightsNotice() {
  return (
    <div className="sticky top-0 z-10 bg-ink-900 pb-2 pt-1">
      <p className="flex items-baseline gap-3 text-xs leading-relaxed text-ink-400">
        <span className="eyebrow shrink-0">Rights</span>
        <span>
          You are responsible for having the rights or permission to clip and repost a
          creator's content, and for following each platform's terms of service.
        </span>
      </p>
    </div>
  )
}
