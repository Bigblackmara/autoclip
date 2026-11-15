/** Display helpers shared by the Live Watch panels. */

import type { LiveClip, LiveStream } from './api'

/** A hand-typed URL without a scheme still reads as its host, or as itself. */
export function hostOf(url: string): string {
  if (!url) return '(no url)'
  try {
    return new URL(url).host || url
  } catch {
    try {
      return new URL(`https://${url}`).host || url
    } catch {
      return url
    }
  }
}

/** The name a stream goes by everywhere in the UI: its label, else its host. */
export function streamName(stream: Pick<LiveStream, 'label' | 'url'>): string {
  return stream.label.trim() || hostOf(stream.url)
}

export function formatMB(mb: number): string {
  if (!Number.isFinite(mb) || mb <= 0) return '0 MB'
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return mb < 10 ? `${mb.toFixed(1)} MB` : `${Math.round(mb)} MB`
}

export function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`
}

/** "just now", "4 min ago", "3 h ago", "yesterday", then a date. */
export function relativeTime(iso: string, now: number): string {
  const at = Date.parse(iso)
  if (!Number.isFinite(at)) return ''
  const seconds = Math.round((now - at) / 1000)
  if (seconds < 45) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} h ago`
  const days = Math.round(hours / 24)
  if (days === 1) return 'yesterday'
  if (days < 7) return `${days} days ago`
  return new Date(at).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

/** Full local timestamp, for tooltips next to a relative time. */
export function absoluteTime(iso: string): string {
  const at = Date.parse(iso)
  if (!Number.isFinite(at)) return iso
  return new Date(at).toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDate(iso: string | null): string {
  if (!iso) return 'an unknown date'
  const at = Date.parse(iso)
  if (!Number.isFinite(at)) return iso
  return new Date(at).toLocaleDateString(undefined, {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  })
}

/** A failure that is really a stop: the app shut down or the work was cancelled. */
export function wasInterrupted(error: string | null): boolean {
  const text = (error ?? '').trim().toLowerCase()
  return text === 'interrupted' || text.startsWith('cancelled')
}

/** A failed clip's error, said plainly. Unknown errors pass through. */
export function describeFailure(error: string | null): string {
  const text = (error ?? '').trim()
  const lower = text.toLowerCase()
  if (lower === 'interrupted') return 'AutoClip stopped before this clip was finished.'
  if (lower === 'cancelled during render') return 'Cancelled mid-render when Live Watch stopped.'
  if (lower === 'cancelled before verdict') return 'Stopped before the judge’s verdict.'
  return text
}

const NOT_JUDGED = /budget exhausted|budget unavailable|^rate limited|^judge (unavailable|error)/

/** False when the window was refused before the judge ever saw it (its score is 0, not a verdict). */
export function wasJudged(clip: LiveClip): boolean {
  return !NOT_JUDGED.test(clip.reason.trim().toLowerCase())
}

/**
 * Why a window was not clipped, in a sentence. Known server reasons are
 * rewritten; the judge's own rationale is shown as it came.
 */
export function describeRejection(clip: LiveClip, minScore: number): string {
  const reason = clip.reason.trim()
  const lower = reason.toLowerCase()
  if (lower === 'discarded by user') return 'Discarded by you.'
  const budget = /judge (minute|hour|day) budget exhausted/.exec(lower)
  if (budget) return `Not judged — this stream’s judge budget for the ${budget[1]} was used up.`
  if (lower.includes('budget unavailable')) return 'Not judged — the judge budget couldn’t be checked.'
  if (lower.startsWith('judge unavailable') || lower.startsWith('judge error')) {
    const detail = reason.replace(/^judge (unavailable|error):?\s*/i, '')
    return `Not judged — the AI provider failed${detail ? `: ${detail}` : '.'}`
  }
  if (lower.includes('queue full') || lower.includes('disk pressure')) {
    return 'Passed the judge, but the render queue was full.'
  }
  const below = /below judge_min_score \((\d+)\)/.exec(lower)
  if (below) return `Scored ${clip.score}, under the bar of ${below[1]}.`
  if (reason) return reason
  if (clip.score > 0) return `Scored ${clip.score}, under the bar of ${minScore}.`
  return 'Not clipped.'
}
