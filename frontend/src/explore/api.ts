/** Explore API client and types (backend: autoclip/api/explore.py). */

import { request } from '../api'

export type ExploreSource = 'youtube' | 'twitch'
export type ItemKind = 'video' | 'short' | 'channel' | 'live' | 'vod' | 'clip'
export type YouTubeType = 'videos' | 'shorts' | 'channels' | 'live'
export type Duration = 'any' | 'short' | 'medium' | 'long'
export type Sort = 'relevance' | 'views' | 'date'
export type Uploaded = 'any' | 'today' | 'week' | 'month' | 'year'
export type ChannelTab = 'videos' | 'shorts' | 'streams' | 'vods' | 'clips'
export type ClipRange = '24hr' | '7d' | '30d' | 'all'

export interface ExploreItem {
  kind: ItemKind
  source: ExploreSource
  id: string
  url: string
  title: string
  channel: string | null
  channel_id: string | null
  channel_url: string | null
  thumbnail: string | null
  duration_s: number | null
  views: number | null
  subscribers: number | null
  upload_date: string | null
  is_live: boolean
  description: string | null
}

export interface ExploreChannel {
  source: ExploreSource
  title: string
  url: string
  avatar: string | null
  banner: string | null
  subscribers: number | null
  description: string | null
}

export interface ChannelPage {
  channel: ExploreChannel
  tab: ChannelTab
  items: ExploreItem[]
}

export interface SearchResult {
  items: ExploreItem[]
  twitch_search: boolean
  note: string | null
}

export interface HeatPoint {
  start_s: number
  end_s: number
  value: number
}

export interface Peak {
  start_s: number
  end_s: number
  peak_s: number
  value: number
  reasons: string[]
}

export interface PeekResult {
  platform: string
  id: string
  url: string
  title: string
  channel: string | null
  channel_url: string | null
  duration_s: number
  views: number | null
  likes: number | null
  upload_date: string | null
  thumbnail: string | null
  is_live: boolean
  description: string
  chapters: { start_s: number; end_s: number; title: string }[]
  heatmap: HeatPoint[]
  peaks: Peak[]
}

export interface SearchParams {
  q: string
  source: ExploreSource
  type?: YouTubeType
  duration?: Duration
  sort?: Sort
  uploaded?: Uploaded
  limit?: number
}

const qs = (params: Record<string, string | number | undefined>) => {
  const out = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') out.set(key, String(value))
  }
  return out.toString()
}

export const exploreApi = {
  status: () => request<{ twitch_search: boolean }>('/api/explore/status'),
  search: (p: SearchParams) => request<SearchResult>(`/api/explore/search?${qs({ ...p })}`),
  channel: (url: string, tab: ChannelTab, opts: { range?: ClipRange; sort?: 'time' | 'views' } = {}) =>
    request<ChannelPage>(`/api/explore/channel?${qs({ url, tab, ...opts })}`),
  peek: (url: string) => request<PeekResult>(`/api/explore/peek?${qs({ url })}`),
}

/** A pasted link, recognised so it opens directly instead of being searched as text. */
export type LinkHit =
  | { kind: 'video'; source: ExploreSource; id: string; url: string }
  | { kind: 'channel'; url: string }

export function parseLink(text: string): LinkHit | null {
  const t = text.trim()
  let m = t.match(/(?:youtube\.com\/(?:watch\?(?:.*&)?v=|live\/|embed\/|shorts\/)|youtu\.be\/)([\w-]{11})/)
  if (m) return { kind: 'video', source: 'youtube', id: m[1], url: `https://www.youtube.com/watch?v=${m[1]}` }
  m = t.match(/twitch\.tv\/videos\/(\d+)/)
  if (m) return { kind: 'video', source: 'twitch', id: `v${m[1]}`, url: `https://www.twitch.tv/videos/${m[1]}` }
  m = t.match(/(?:clips\.twitch\.tv\/|twitch\.tv\/\w+\/clip\/)([\w-]+)/)
  if (m) return { kind: 'video', source: 'twitch', id: m[1], url: t.startsWith('http') ? t : `https://${t}` }
  if (/youtube\.com\/(@|channel\/|c\/|user\/)/i.test(t) || /twitch\.tv\/[a-z0-9_]+\/?$/i.test(t)) {
    return { kind: 'channel', url: t.startsWith('http') ? t : `https://${t}` }
  }
  return null
}

/** 1234567 -> "1.2M". */
export function compactNumber(n: number | null | undefined): string {
  if (n == null) return ''
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(n)
}

/** "2026-11-01" -> "2 weeks ago". */
export function timeAgo(isoDate: string | null): string {
  if (!isoDate) return ''
  const then = new Date(isoDate).getTime()
  if (Number.isNaN(then)) return ''
  const days = Math.floor((Date.now() - then) / 86_400_000)
  if (days < 1) return 'today'
  if (days < 2) return 'yesterday'
  if (days < 14) return `${days} days ago`
  if (days < 60) return `${Math.floor(days / 7)} weeks ago`
  if (days < 730) return `${Math.floor(days / 30)} months ago`
  return `${Math.floor(days / 365)} years ago`
}
