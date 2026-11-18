import { cx } from './cx'

/** A shimmering placeholder block while something loads. */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cx('skeleton h-4', className)} />
}

/** A card-shaped loading placeholder with a few text lines. */
export function SkeletonCard({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={cx('card p-5 sm:p-6', className)} aria-hidden>
      <Skeleton className="h-5 w-1/3" />
      <div className="mt-4 space-y-2.5">
        {Array.from({ length: lines }, (_, i) => (
          <Skeleton key={i} className={cx('h-3.5', i === lines - 1 ? 'w-2/3' : 'w-full')} />
        ))}
      </div>
    </div>
  )
}

/** Full-page loading state: a header placeholder plus a couple of cards. */
export function PageSkeleton({ cards = 2 }: { cards?: number }) {
  return (
    <div role="status" aria-label="Loading">
      <Skeleton className="h-8 w-64" />
      <Skeleton className="mt-3 h-4 w-96 max-w-full" />
      <div className="mt-8 space-y-5">
        {Array.from({ length: cards }, (_, i) => (
          <SkeletonCard key={i} lines={i === 0 ? 4 : 3} />
        ))}
      </div>
    </div>
  )
}
