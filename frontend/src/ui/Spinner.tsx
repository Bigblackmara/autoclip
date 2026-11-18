import { cx } from './cx'

/** Small circular spinner; inherits the text colour. */
export function Spinner({ size = 16, className }: { size?: number; className?: string }) {
  return (
    <span
      aria-hidden
      style={{ width: size, height: size }}
      className={cx(
        'inline-block shrink-0 animate-spin rounded-full border-2 border-current border-r-transparent opacity-80',
        className,
      )}
    />
  )
}

/** A thin determinate progress bar (0–1). */
export function ProgressBar({
  value,
  className,
  tone = 'accent',
  label,
}: {
  value: number
  className?: string
  tone?: 'accent' | 'success' | 'danger'
  label?: string
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      aria-label={label}
      className={cx('h-1.5 w-full overflow-hidden rounded-full bg-subtle', className)}
    >
      <div
        className={cx(
          'h-full rounded-full transition-[width] duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]',
          tone === 'accent' && 'bg-accent',
          tone === 'success' && 'bg-success',
          tone === 'danger' && 'bg-danger',
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}
