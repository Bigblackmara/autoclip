import { useCallback, useEffect, useState } from 'react'

import { ErrorNote } from '../components/ErrorNote'
import {
  ArrowLeftIcon,
  Button,
  Card,
  CompassIcon,
  EmptyState,
  Field,
  Input,
  PageHeader,
  SearchIcon,
  Segmented,
  Select,
  Skeleton,
  Tabs,
} from '../ui'
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
    <div>
      <PageHeader
        title="Explore"
        description="Find something worth clipping on YouTube or Twitch. Click a result to see where viewers rewatch before you clip it."
      />

      <Card padding="md">
        <Tabs
          label="Where to search"
          variant="pill"
          items={[
            { key: 'youtube', label: 'YouTube' },
            { key: 'twitch', label: 'Twitch' },
          ]}
          value={source}
          onChange={switchSource}
        />

        <form
          onSubmit={(e) => {
            e.preventDefault()
            submit(input)
          }}
          className="mt-4 flex flex-col gap-3 sm:flex-row"
        >
          <label htmlFor="explore-q" className="sr-only">
            Search
          </label>
          <div className="relative min-w-0 flex-1">
            <SearchIcon size={18} className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-fg-subtle" />
            <Input
              id="explore-q"
              className="py-3 pl-11 text-base"
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
          </div>
          <Button type="submit" variant="primary" size="lg" icon={SearchIcon} loading={loading} disabled={!input.trim()}>
            {loading ? 'Searching…' : 'Search'}
          </Button>
        </form>

        {source === 'youtube' ? (
          <div className="mt-4 flex flex-wrap items-end gap-x-6 gap-y-4">
            <div>
              <p className="mb-1.5 text-sm font-medium text-fg">Show</p>
              <Segmented
                label="Result type"
                options={YT_TYPES}
                value={ytType}
                onChange={(value) => {
                  setYtType(value)
                  setChannel(null)
                }}
              />
            </div>
            {ytType === 'videos' && (
              <FilterSelect label="Length" value={duration} options={DURATIONS} onChange={setDuration} />
            )}
            {ytType !== 'channels' && (
              <>
                <FilterSelect label="Sort by" value={sort} options={SORTS} onChange={setSort} />
                <FilterSelect label="Uploaded" value={uploaded} options={UPLOADED} onChange={setUploaded} />
              </>
            )}
          </div>
        ) : (
          !twitchSearch && (
            <p className="mt-3 max-w-3xl text-xs leading-relaxed text-fg-muted">
              Type a channel name to see its past broadcasts and top clips. Keyword search for channels needs Twitch
              app credentials (Client ID and secret), saved in GTA Mode → Settings. Live broadcasts are hidden until
              they end.
            </p>
          )
        )}
      </Card>

      <div className="mt-8">
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
            {note && <p className="mb-6 text-xs text-fg-muted">{note}</p>}
            {items.length === 0 ? (
              <EmptyState
                compact
                icon={SearchIcon}
                title="Nothing found"
                description="Try other words or loosen the filters."
              />
            ) : (
              <ResultGrid items={items} onOpen={openItem} onChannel={openChannelOf} />
            )}
          </div>
        ) : (
          source === 'youtube' && (
            <EmptyState
              icon={CompassIcon}
              title="Start with a search"
              description="Long, well-watched videos give the clipper the most to work with. Try one of these:"
              action={SUGGESTIONS.map((s) => (
                <Button
                  key={s}
                  size="sm"
                  onClick={() => {
                    setInput(s)
                    setDuration('long')
                    setSort('views')
                    setYtType('videos')
                    submit(s)
                  }}
                >
                  {s}
                </Button>
              ))}
            />
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
        <Button variant="ghost" icon={ArrowLeftIcon} onClick={onBack} className="-ml-2 mb-4">
          Back to results
        </Button>
      )}

      <Card padding="none" className="overflow-hidden">
        {info?.banner && (
          <div className="h-24 overflow-hidden md:h-32">
            <img src={info.banner} alt="" referrerPolicy="no-referrer" className="h-full w-full object-cover" />
          </div>
        )}
        <div className="flex items-center gap-4 p-5 sm:p-6">
          <Avatar src={info?.avatar ?? null} name={info?.title ?? view.title} size={56} />
          <div className="min-w-0">
            <h2 className="truncate text-xl font-bold text-fg md:text-2xl">{info?.title ?? view.title}</h2>
            <p className="numeric mt-1 text-sm text-fg-muted">
              {info?.subscribers != null && `${compactNumber(info.subscribers)} subscribers · `}
              <a href={info?.url ?? view.url} target="_blank" rel="noreferrer" className="link">
                {view.source === 'twitch' ? 'twitch.tv' : 'youtube.com'} ↗
              </a>
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-end justify-between gap-4 px-5 sm:px-6">
          <Tabs label="Channel content" items={tabs.map((t) => ({ key: t.value, label: t.label }))} value={tab} onChange={setTab} className="border-b-0" />
          {view.source === 'twitch' && (
            <div className="mb-3">
              {tab === 'clips' ? (
                <FilterSelect label="Range" value={range} options={CLIP_RANGES} onChange={setRange} />
              ) : (
                <FilterSelect
                  label="Sort by"
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
      </Card>

      <div className="mt-8">
        {error ? (
          <div className="max-w-3xl">
            <ErrorNote error={error} />
          </div>
        ) : loading && !page ? (
          <SkeletonGrid />
        ) : page && page.items.length === 0 ? (
          <EmptyState compact title="Nothing on this tab" />
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
    <Field label={label} className="min-w-40">
      <Select value={value} onChange={(e) => onChange(e.target.value as T)} options={options} />
    </Field>
  )
}

function SkeletonGrid() {
  return (
    <ul className="grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4" aria-hidden>
      {Array.from({ length: 8 }, (_, i) => (
        <li key={i}>
          <Skeleton className="aspect-video h-auto w-full rounded-xl" />
          <Skeleton className="mt-3 h-3 w-4/5" />
          <Skeleton className="mt-2 h-3 w-2/5" />
        </li>
      ))}
    </ul>
  )
}
