import { formatDuration } from '../api'
import { compactNumber, timeAgo, type ExploreItem } from './api'

/** Result grid: wide cards for videos / VODs / clips, tall ones for Shorts. */
export function ResultGrid({
  items,
  onOpen,
  onChannel,
}: {
  items: ExploreItem[]
  onOpen: (item: ExploreItem) => void
  onChannel: (item: ExploreItem) => void
}) {
  const channels = items.filter((i) => i.kind === 'channel')
  const shorts = items.filter((i) => i.kind === 'short')
  const wide = items.filter((i) => i.kind !== 'channel' && i.kind !== 'short')

  return (
    <div className="space-y-10">
      {channels.length > 0 && (
        <ul className="grid gap-x-8 gap-y-1 md:grid-cols-2 xl:grid-cols-3">
          {channels.map((item) => (
            <li key={`${item.source}-${item.id}`}>
              <ChannelRow item={item} onOpen={() => onChannel(item)} />
            </li>
          ))}
        </ul>
      )}
      {wide.length > 0 && (
        <ul className="grid grid-cols-1 gap-x-6 gap-y-8 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
          {wide.map((item) => (
            <li key={`${item.source}-${item.id}`}>
              <VideoCard item={item} onOpen={() => onOpen(item)} onChannel={() => onChannel(item)} />
            </li>
          ))}
        </ul>
      )}
      {shorts.length > 0 && (
        <ul className="grid grid-cols-2 gap-x-4 gap-y-6 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6">
          {shorts.map((item) => (
            <li key={`${item.source}-${item.id}`}>
              <VideoCard item={item} tall onOpen={() => onOpen(item)} onChannel={() => onChannel(item)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function VideoCard({
  item,
  tall,
  onOpen,
  onChannel,
}: {
  item: ExploreItem
  tall?: boolean
  onOpen: () => void
  onChannel: () => void
}) {
  const meta = [
    item.views != null ? `${compactNumber(item.views)} views` : '',
    timeAgo(item.upload_date),
  ].filter(Boolean)

  return (
    <div className="group">
      <button type="button" onClick={onOpen} className="block w-full text-left" aria-label={`Peek: ${item.title}`}>
        <div
          className={[
            'relative overflow-hidden rounded-[2px] border border-ink-800 bg-ink-850 transition-colors duration-200 group-hover:border-ink-600',
            tall ? 'aspect-[9/16]' : 'aspect-video',
          ].join(' ')}
        >
          {item.thumbnail ? (
            <img
              src={item.thumbnail}
              alt=""
              loading="lazy"
              referrerPolicy="no-referrer"
              className="h-full w-full object-cover transition-transform duration-500 ease-out group-hover:scale-[1.02]"
            />
          ) : (
            <div className="flex h-full items-center justify-center text-xs text-ink-600">
              {item.source === 'twitch' ? 'Twitch' : 'No preview'}
            </div>
          )}
          {item.is_live ? (
            <span className="absolute bottom-1.5 right-1.5 bg-signal-bad px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-ink-900">
              Live
            </span>
          ) : (
            item.duration_s != null && (
              <span className="numeric absolute bottom-1.5 right-1.5 bg-ink-900/85 px-1.5 py-0.5 text-[11px] font-medium text-ink-100">
                {formatDuration(item.duration_s)}
              </span>
            )
          )}
          {(item.kind === 'vod' || item.kind === 'clip') && (
            <span className="absolute left-1.5 top-1.5 bg-ink-900/85 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-ink-300">
              {item.kind === 'vod' ? 'Past broadcast' : 'Clip'}
            </span>
          )}
        </div>
        <p className="mt-2.5 line-clamp-2 text-sm font-medium leading-snug text-ink-100 group-hover:text-sodium-400">
          {item.title}
        </p>
      </button>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-2 text-xs text-ink-500">
        {item.channel &&
          (item.channel_url ? (
            <button type="button" onClick={onChannel} className="truncate text-ink-400 hover:text-sodium-400">
              {item.channel}
            </button>
          ) : (
            <span className="truncate text-ink-400">{item.channel}</span>
          ))}
        {meta.map((m) => (
          <span key={m} className="numeric">
            {m}
          </span>
        ))}
      </div>
    </div>
  )
}

function ChannelRow({ item, onOpen }: { item: ExploreItem; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="group flex w-full items-center gap-4 border-b border-ink-800 py-3 text-left"
    >
      <Avatar src={item.thumbnail} name={item.title} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-display text-xl text-ink-100 group-hover:text-sodium-400">
          {item.title}
        </span>
        <span className="block truncate text-xs text-ink-500">
          {item.subscribers != null ? `${compactNumber(item.subscribers)} subscribers` : item.description}
        </span>
      </span>
      <span className="text-xs text-ink-500 group-hover:text-sodium-400">Browse →</span>
    </button>
  )
}

export function Avatar({ src, name, size = 44 }: { src: string | null; name: string; size?: number }) {
  return src ? (
    <img
      src={src}
      alt=""
      referrerPolicy="no-referrer"
      className="shrink-0 rounded-full border border-ink-800 object-cover"
      style={{ width: size, height: size }}
    />
  ) : (
    <span
      className="flex shrink-0 items-center justify-center rounded-full border border-ink-700 bg-ink-850 font-display text-lg text-ink-300"
      style={{ width: size, height: size }}
    >
      {name.slice(0, 1).toUpperCase()}
    </span>
  )
}
