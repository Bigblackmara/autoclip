import { useCallback, useEffect, useState } from 'react'

import { ErrorNote } from '../components/ErrorNote'
import { Avatar, ResultGrid } from '../explore/Cards'
import { PeekPanel, type PeekTarget } from '../explore/PeekPanel'
import {
  compactNumber,
  exploreApi,
  parseLink,
  type ChannelPage,
  type ChannelTab,
  type ClipRange,
  type Duration,
  type ExploreItem,
  type ExploreSource,
  type Sort,
  type Uploaded,
  type YouTubeType,
} from '../explore/api'

const YT_TYPES: { value: YouTubeType; label: string }[] = [
  { value: 'videos', label: 'Videos' },
  { value: 'shorts', label: 'Shorts' },
  { value: 'channels', label: 'Channels' },
  { value: 'live', label: 'Live' },
]

const DURATIONS: { value: Duration; label: string }[] = [
  { value: 'any', label: 'Any length' },
  { value: 'short', label: 'Under 4 min' },
  { value: 'medium', label: '4–20 min' },
  { value: 'long', label: 'Over 20 min' },
]

const SORTS: { value: Sort; label: string }[] = [
  { value: 'relevance', label: 'Relevance' },
  { value: 'views', label: 'Most viewed' },
  { value: 'date', label: 'Newest' },
]

const UPLOADED: { value: Uploaded; label: string }[] = [
  { value: 'any', label: 'Any time' },
  { value: 'today', label: 'Today' },
  { value: 'week', label: 'This week' },
  { value: 'month', label: 'This month' },
  { value: 'year', label: 'This year' },
]

const CLIP_RANGES: { value: ClipRange; label: string }[] = [
  { value: '24hr', label: 'Last 24 hours' },
  { value: '7d', label: 'Last 7 days' },
  { value: '30d', label: 'Last 30 days' },
  { value: 'all', label: 'All time' },
]

const SUGGESTIONS = ['podcast', 'interview', 'stand up comedy', 'startup founder', 'gaming highlights']

interface ChannelView {
  source: ExploreSource
  url: string
  title: string
}

/**
 * Explore: find something worth clipping.
 *
 * Search YouTube (with the same filters as its own search page) or open a
 * Twitch channel's past broadcasts, then Peek at a video to see where people
 * rewatch before sending it to the clipper.
 */
export function Explore() {
  const [source, setSource] = useState<ExploreSource>('youtube')
  const [ytType, setYtType] = useState<YouTubeType>('videos')
  const [duration, setDuration] = useState<Duration>('any')
  const [sort, setSort] = useState<Sort>('relevance')
  const [uploaded, setUploaded] = useState<Uploaded>('any')

  const [input, setInput] = useState('')
  const [query, setQuery] = useState('')
  const [items, setItems] = useState<ExploreItem[] | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [twitchSearch, setTwitchSearch] = useState(false)

  const [channel, setChannel] = useState<ChannelView | null>(null)
  const [peek, setPeek] = useState<PeekTarget | null>(null)

  useEffect(() => {
    exploreApi
      .status()
      .then((s) => setTwitchSearch(s.twitch_search))
      .catch(() => undefined)
  }, [])

  // Re-run the search whenever the query, tab or a filter changes.
  useEffect(() => {
    if (!query) return
    let stale = false
    setLoading(true)
    setError(null)
    setNote(null)
    exploreApi
      .search({ q: query, source, type: ytType, duration, sort, uploaded })
      .then((result) => {
        if (stale) return
        setTwitchSearch(result.twitch_search)
        setNote(result.note)
        // Without Helix credentials a Twitch search is a channel name: open it.
        if (source === 'twitch' && !result.twitch_search && result.items.length === 1) {
          const only = result.items[0]
          setChannel({ source: 'twitch', url: only.url, title: only.title })
        }
        setItems(result.items)
      })
      .catch((e) => !stale && setError(e as Error))
      .finally(() => !stale && setLoading(false))
    return () => {
      stale = true
    }
  }, [query, source, ytType, duration, sort, uploaded])

  const submit = (text: string) => {
    const value = text.trim()
    if (!value) return
    const link = parseLink(value)
    if (link?.kind === 'video') {
      setPeek({ source: link.source, id: link.id, url: link.url })
      return
    }
    if (link?.kind === 'channel') {
      const isTwitch = /twitch\.tv/i.test(link.url)
      setSource(isTwitch ? 'twitch' : 'youtube')
      setQuery('')
      setChannel({ source: isTwitch ? 'twitch' : 'youtube', url: link.url, title: link.url })
      return
    }
    setChannel(null)
    setQuery(value)
  }

  const switchSource = (next: ExploreSource) => {
    if (next === source) return
    setSource(next)
    setChannel(null)
    setQuery('')
    setItems(null)
    setNote(null)
    setError(null)
  }

  const openItem = useCallback((item: ExploreItem) => {
    if (item.kind === 'channel') {
      setChannel({ source: item.source, url: item.url, title: item.title })
      return
    }
    setPeek({
      source: item.source,
      id: item.id,
      url: item.url,
      title: item.title,
      thumbnail: item.thumbnail,
      channel: item.channel,
      is_live: item.is_live,
    })
  }, [])

  const openChannelOf = useCallback((item: ExploreItem) => {
    const url = item.kind === 'channel' ? item.url : item.channel_url
    if (url) setChannel({ source: item.source, url, title: item.channel ?? item.title })
  }, [])

  return (
    <div className="pt-14">
      <div className="rise max-w-3xl">
        <p className="eyebrow">Explore · YouTube & Twitch · No API key</p>
        <h1 className="mt-5 font-display text-[clamp(2.5rem,6vw,4.75rem)] leading-[0.95] text-ink-100">
          Find something
          <br />
          worth <span className="italic text-sodium-500">clipping</span>.
        </h1>
      </div>

      {/* Source switch */}
      <div className="rise mt-12 flex items-baseline gap-6 border-b border-ink-800" style={{ animationDelay: '60ms' }}>
        {(['youtube', 'twitch'] as const).map((s) => (
          <TabButton key={s} active={source === s} onClick={() => switchSource(s)} large>
            {s === 'youtube' ? 'YouTube' : 'Twitch'}
          </TabButton>
        ))}
      </div>

      <section className="rise mt-8" style={{ animationDelay: '120ms' }}>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit(input)
          }}
          className="flex max-w-3xl items-end gap-4"
        >
          <label htmlFor="explore-q" className="sr-only">
            Search
          </label>
          <input
            id="explore-q"
            className="field font-display text-xl md:text-2xl"
            placeholder={
              source === 'youtube'
                ? 'Search YouTube, or paste a video or channel link'
                : twitchSearch
                  ? 'Search Twitch channels'
                  : 'Twitch channel name, e.g. shroud'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            autoComplete="off"
            spellCheck={false}
          />
          <button type="submit" className="btn btn-primary shrink-0" disabled={!input.trim()}>
            {loading ? 'Searching…' : 'Search'}
          </button>
        </form>

        {source === 'youtube' ? (
          <div className="mt-6 flex flex-wrap items-end gap-x-8 gap-y-4">
            <div className="flex items-baseline gap-5">
              {YT_TYPES.map((t) => (
                <TabButton
                  key={t.value}
                  active={ytType === t.value}
                  onClick={() => {
                    setYtType(t.value)
                    setChannel(null)
                  }}
                >
                  {t.label}
                </TabButton>
              ))}
            </div>
            <div className="flex flex-wrap items-end gap-6">
              {ytType === 'videos' && (
                <FilterSelect label="Length" value={duration} options={DURATIONS} onChange={setDuration} />
              )}
              {ytType !== 'channels' && (
                <>
                  <FilterSelect label="Sort" value={sort} options={SORTS} onChange={setSort} />
                  <FilterSelect label="Uploaded" value={uploaded} options={UPLOADED} onChange={setUploaded} />
                </>
              )}
            </div>
          </div>
        ) : (
          !twitchSearch && (
            <p className="mt-3 max-w-3xl text-xs leading-relaxed text-ink-500">
              Type a channel name to see its past broadcasts and top clips. Keyword search for
              channels needs Twitch app credentials (Client ID and secret), saved in GTA Mode
              settings. Live broadcasts are hidden until they end.
            </p>
          )
        )}
      </section>

      <div className="mt-10">
        {error && (
          <div className="mb-8 max-w-3xl">
            <ErrorNote error={error} onDismiss={() => setError(null)} />
          </div>
        )}

        {channel ? (
          <ChannelBrowser
            key={channel.url}
            view={channel}
            onBack={items ? () => setChannel(null) : null}
            onOpen={openItem}
          />
        ) : loading && !items ? (
          <SkeletonGrid />
        ) : items ? (
          <div className={loading ? 'opacity-50 transition-opacity' : 'transition-opacity'}>
            {note && <p className="mb-6 text-xs text-ink-500">{note}</p>}
            {items.length === 0 ? (
              <p className="text-sm text-ink-400">Nothing found. Try other words or loosen the filters.</p>
            ) : (
              <ResultGrid items={items} onOpen={openItem} onChannel={openChannelOf} />
            )}
          </div>
        ) : (
          source === 'youtube' && (
            <div className="max-w-3xl">
              <p className="eyebrow">Try</p>
              <div className="mt-3 flex flex-wrap gap-2">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => {
                      setInput(s)
                      setDuration('long')
                      setSort('views')
                      setYtType('videos')
                      submit(s)
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
              <p className="mt-4 text-xs leading-relaxed text-ink-500">
                Long, well-watched videos give the clipper the most to work with. Click any result to
                see its "Most replayed" graph before you clip it.
              </p>
            </div>
          )
        )}
      </div>

      {peek && <PeekPanel target={peek} onClose={() => setPeek(null)} />}
    </div>
  )
}

function ChannelBrowser({
  view,
  onBack,
  onOpen,
}: {
  view: ChannelView
  onBack: (() => void) | null
  onOpen: (item: ExploreItem) => void
}) {
  const tabs: { value: ChannelTab; label: string }[] =
    view.source === 'twitch'
      ? [
          { value: 'vods', label: 'Past broadcasts' },
          { value: 'clips', label: 'Top clips' },
        ]
      : [
          { value: 'videos', label: 'Videos' },
          { value: 'shorts', label: 'Shorts' },
          { value: 'streams', label: 'Streams' },
        ]
  const [tab, setTab] = useState<ChannelTab>(tabs[0].value)
  const [range, setRange] = useState<ClipRange>('7d')
  const [twitchSort, setTwitchSort] = useState<'time' | 'views'>('time')
  const [page, setPage] = useState<ChannelPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    let stale = false
    setLoading(true)
    setError(null)
    const opts = view.source === 'twitch' ? (tab === 'clips' ? { range } : { sort: twitchSort }) : {}
    exploreApi
      .channel(view.url, tab, opts)
      .then((p) => !stale && setPage(p))
      .catch((e) => !stale && setError(e as Error))
      .finally(() => !stale && setLoading(false))
    return () => {
      stale = true
    }
  }, [view.url, view.source, tab, range, twitchSort])

  const info = page?.channel
  return (
    <div>
      {onBack && (
        <button type="button" onClick={onBack} className="btn btn-quiet -ml-1 mb-6">
          ← Back to results
        </button>
      )}

      {info?.banner && (
        <div className="mb-6 h-24 overflow-hidden rounded-[2px] border border-ink-800 md:h-32">
          <img src={info.banner} alt="" referrerPolicy="no-referrer" className="h-full w-full object-cover" />
        </div>
      )}
      <div className="flex items-center gap-5">
        <Avatar src={info?.avatar ?? null} name={info?.title ?? view.title} size={64} />
        <div className="min-w-0">
          <h2 className="truncate font-display text-3xl text-ink-100 md:text-4xl">{info?.title ?? view.title}</h2>
          <p className="numeric mt-1 text-xs text-ink-500">
            {info?.subscribers != null && `${compactNumber(info.subscribers)} subscribers · `}
            <a href={info?.url ?? view.url} target="_blank" rel="noreferrer" className="hover:text-sodium-400">
              {view.source === 'twitch' ? 'twitch.tv' : 'youtube.com'} ↗
            </a>
          </p>
        </div>
      </div>

      <div className="mt-8 flex flex-wrap items-end justify-between gap-4 border-b border-ink-800">
        <div className="flex items-baseline gap-6">
          {tabs.map((t) => (
            <TabButton key={t.value} active={tab === t.value} onClick={() => setTab(t.value)}>
              {t.label}
            </TabButton>
          ))}
        </div>
        {view.source === 'twitch' && (
          <div className="mb-2">
            {tab === 'clips' ? (
              <FilterSelect label="Range" value={range} options={CLIP_RANGES} onChange={setRange} />
            ) : (
              <FilterSelect
                label="Sort"
                value={twitchSort}
                options={[
                  { value: 'time', label: 'Newest' },
                  { value: 'views', label: 'Most viewed' },
                ]}
                onChange={setTwitchSort}
              />
            )}
          </div>
        )}
      </div>

      <div className="mt-8">
        {error ? (
          <div className="max-w-3xl">
            <ErrorNote error={error} />
          </div>
        ) : loading && !page ? (
          <SkeletonGrid />
        ) : page && page.items.length === 0 ? (
          <p className="text-sm text-ink-400">Nothing on this tab.</p>
        ) : (
          page && (
            <div className={loading ? 'opacity-50 transition-opacity' : 'transition-opacity'}>
              <ResultGrid items={page.items} onOpen={onOpen} onChannel={() => undefined} />
            </div>
          )
        )}
      </div>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  large,
  children,
}: {
  active: boolean
  onClick: () => void
  large?: boolean
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        '-mb-px border-b-2 pb-2 font-medium transition-colors duration-200',
        large ? 'text-base' : 'text-sm',
        active ? 'border-sodium-500 text-sodium-500' : 'border-transparent text-ink-400 hover:text-ink-200',
      ].join(' ')}
    >
      {children}
    </button>
  )
}

function FilterSelect<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
}) {
  return (
    <label className="block min-w-36">
      <span className="eyebrow">{label}</span>
      <select
        className="field mt-1 cursor-pointer text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function SkeletonGrid() {
  return (
    <ul className="grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4" aria-hidden>
      {Array.from({ length: 8 }, (_, i) => (
        <li key={i}>
          <div className="aspect-video animate-pulse bg-ink-850" />
          <div className="mt-3 h-3 w-4/5 animate-pulse bg-ink-850" />
          <div className="mt-2 h-3 w-2/5 animate-pulse bg-ink-850" />
        </li>
      ))}
    </ul>
  )
}
