import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

import { formatDuration } from '../api'
import { ErrorNote } from '../components/ErrorNote'
import { gtaApi } from '../gta/api'
import { SendToGtaButton } from '../gta/SendToGtaButton'
import { HeatChart } from './HeatChart'
import { compactNumber, exploreApi, timeAgo, type ExploreItem, type PeekResult } from './api'

/** What the panel needs to start showing something before Peek returns. */
export interface PeekTarget {
  source: ExploreItem['source']
  id: string
  url: string
  title?: string
  thumbnail?: string | null
  channel?: string | null
  is_live?: boolean
}

function youtubeEmbed(id: string, start: number | null) {
  const params = new URLSearchParams({ rel: '0', modestbranding: '1' })
  if (start !== null) {
    params.set('start', String(Math.floor(start)))
    params.set('autoplay', '1')
  }
  return `https://www.youtube-nocookie.com/embed/${encodeURIComponent(id)}?${params}`
}

/**
 * Right-hand drawer: watch the video, see where viewers rewatch, and send it
 * to the clipper. A peak click reloads the embed at that second (the iframe
 * `start` parameter), which needs no player API.
 */
export function PeekPanel({ target, onClose }: { target: PeekTarget; onClose: () => void }) {
  const navigate = useNavigate()
  const [peek, setPeek] = useState<PeekResult | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [start, setStart] = useState<number | null>(null)
  const [queued, setQueued] = useState<'idle' | 'busy' | 'done'>('idle')
  const [queueError, setQueueError] = useState<Error | null>(null)

  useEffect(() => {
    let stale = false
    setPeek(null)
    setError(null)
    setStart(null)
    setQueued('idle')
    setQueueError(null)
    exploreApi
      .peek(target.url)
      .then((p) => !stale && setPeek(p))
      .catch((e) => !stale && setError(e as Error))
    return () => {
      stale = true
    }
  }, [target.url])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const isYouTube = target.source === 'youtube'
  const videoId = peek?.id || target.id
  const title = peek?.title ?? target.title ?? 'Loading…'
  const isLive = peek?.is_live ?? target.is_live ?? false
  const url = peek?.url ?? target.url
  const thumbnail = peek?.thumbnail ?? target.thumbnail ?? null

  const meta = peek
    ? [
        peek.channel,
        peek.views != null ? `${compactNumber(peek.views)} views` : null,
        peek.likes != null ? `${compactNumber(peek.likes)} likes` : null,
        timeAgo(peek.upload_date) || null,
        peek.duration_s ? formatDuration(peek.duration_s) : null,
      ].filter(Boolean)
    : [target.channel].filter(Boolean)

  const clipThis = () => navigate(`/?url=${encodeURIComponent(url)}`)

  const sendToGta = async () => {
    setQueued('busy')
    setQueueError(null)
    try {
      const result = await gtaApi.addStreams([url], {})
      if (result.added.length === 0 && result.skipped.length > 0) {
        throw new Error(result.skipped[0].reason)
      }
      setQueued('done')
    } catch (e) {
      setQueueError(e as Error)
      setQueued('idle')
    }
  }

  return (
    <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true" aria-label={title}>
      <button type="button" className="absolute inset-0 bg-scrim" onClick={onClose} aria-label="Close preview" />
      <aside className="relative flex h-full w-full max-w-2xl flex-col overflow-y-auto border-l border-line bg-surface shadow-[var(--shadow-pop)]">
        <div className="flex items-center justify-between border-b border-ink-800 px-6 py-3">
          <span className="eyebrow">Peek</span>
          <button type="button" onClick={onClose} className="btn btn-quiet" aria-label="Close">
            ✕
          </button>
        </div>

        <div className="aspect-video w-full bg-black">
          {isYouTube ? (
            <iframe
              key={`${videoId}-${start ?? 'none'}`}
              src={youtubeEmbed(videoId, start)}
              title={title}
              className="h-full w-full"
              allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
              allowFullScreen
            />
          ) : thumbnail ? (
            <a href={url} target="_blank" rel="noreferrer" className="block h-full w-full">
              <img src={thumbnail} alt="" referrerPolicy="no-referrer" className="h-full w-full object-cover" />
            </a>
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-ink-500">No preview</div>
          )}
        </div>

        <div className="space-y-8 px-6 py-6">
          <div>
            <h2 className="font-display text-2xl leading-tight text-ink-100 md:text-3xl">{title}</h2>
            <p className="numeric mt-2 flex flex-wrap gap-x-3 text-xs text-ink-400">
              {meta.map((m) => (
                <span key={String(m)}>{m}</span>
              ))}
              {isLive && <span className="font-semibold uppercase text-signal-bad">Live now</span>}
            </p>
          </div>

          {/* Actions */}
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              {isYouTube ? (
                <button type="button" className="btn btn-primary" onClick={clipThis} disabled={isLive}>
                  Clip this video
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => void sendToGta()}
                  disabled={isLive || queued !== 'idle'}
                >
                  {queued === 'busy' ? 'Queueing…' : queued === 'done' ? 'Queued' : 'Send to GTA Mode'}
                </button>
              )}
              {isYouTube && <SendToGtaButton url={url} disabled={isLive} />}
              <a href={url} target="_blank" rel="noreferrer" className="btn btn-ghost">
                Open on {isYouTube ? 'YouTube' : 'Twitch'} ↗
              </a>
            </div>
            {isLive ? (
              <p className="text-xs leading-relaxed text-ink-500">
                This is live right now, so there is no finished video to clip yet.{' '}
                <Link to="/live" className="text-sodium-500 underline underline-offset-4">
                  Live Watch
                </Link>{' '}
                can clip it as it streams.
              </p>
            ) : isYouTube ? (
              <p className="text-xs leading-relaxed text-ink-500">
                Opens New with this link filled in, so you can set options before starting.
              </p>
            ) : (
              <p className="text-xs leading-relaxed text-ink-500">
                Normal mode takes YouTube links. Twitch broadcasts go through GTA Mode's stream queue.
                {queued === 'done' && (
                  <>
                    {' '}
                    <Link to="/gta" className="text-sodium-500 underline underline-offset-4">
                      Open GTA Mode
                    </Link>
                  </>
                )}
              </p>
            )}
            {queueError && <ErrorNote error={queueError} onDismiss={() => setQueueError(null)} />}
          </div>

          {/* Most replayed */}
          <section>
            <p className="eyebrow">Most replayed</p>
            {error ? (
              <div className="mt-3">
                <ErrorNote error={error} />
              </div>
            ) : !peek ? (
              <div className="mt-3 space-y-2" aria-live="polite">
                <div className="h-24 animate-pulse bg-ink-850" />
                <p className="text-xs text-ink-500">Reading the video's details…</p>
              </div>
            ) : peek.heatmap.length === 0 ? (
              <p className="mt-3 text-sm text-ink-400">
                {isYouTube
                  ? 'No "Most replayed" graph for this video yet. YouTube shows it only once enough people have watched.'
                  : 'Twitch has no "Most replayed" graph. AutoClip finds the moments from the stream itself.'}
              </p>
            ) : (
              <div className="mt-4">
                <HeatChart
                  heatmap={peek.heatmap}
                  peaks={isYouTube ? peek.peaks : []}
                  duration={peek.duration_s}
                  position={start}
                  onSeek={(t) => isYouTube && setStart(t)}
                />
                {peek.peaks.length > 0 && (
                  <ol className="mt-5 divide-y divide-ink-800 border-y border-ink-800">
                    {peek.peaks.map((p, i) => (
                      <li key={i}>
                        <button
                          type="button"
                          onClick={() => isYouTube && setStart(p.start_s)}
                          className="group flex w-full items-baseline gap-4 py-2.5 text-left"
                        >
                          <span className="numeric w-5 text-sm font-bold text-sodium-500">{i + 1}</span>
                          <span className="numeric w-28 shrink-0 text-sm text-ink-200 group-hover:text-sodium-400">
                            {formatDuration(p.start_s)}–{formatDuration(p.end_s)}
                          </span>
                          <span className="min-w-0 flex-1 truncate text-xs text-ink-500">{p.reasons.join(' · ')}</span>
                          <span className="numeric text-xs text-ink-400">{Math.round(p.value * 100)}</span>
                        </button>
                      </li>
                    ))}
                  </ol>
                )}
              </div>
            )}
          </section>

          {peek && peek.chapters.length > 0 && (
            <section>
              <p className="eyebrow">Chapters</p>
              <ol className="mt-3 divide-y divide-ink-800 border-y border-ink-800">
                {peek.chapters.map((c, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      onClick={() => isYouTube && setStart(c.start_s)}
                      className="group flex w-full items-baseline gap-4 py-2 text-left"
                      disabled={!isYouTube}
                    >
                      <span className="numeric w-16 shrink-0 text-xs text-ink-400 group-hover:text-sodium-400">
                        {formatDuration(c.start_s)}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-sm text-ink-200">{c.title}</span>
                    </button>
                  </li>
                ))}
              </ol>
            </section>
          )}

          {peek?.description && (
            <section>
              <p className="eyebrow">Description</p>
              <p className="mt-3 whitespace-pre-line break-words text-xs leading-relaxed text-ink-400 [overflow-wrap:anywhere]">
                {peek.description}
              </p>
            </section>
          )}
        </div>
      </aside>
    </div>
  )
}
