import type { HTMLAttributes, ReactNode } from 'react'

import { cx } from './cx'
import type { IconComponent } from './icons'

export interface CardProps extends Omit<HTMLAttributes<HTMLElement>, 'title'> {
  /** Optional heading drawn at the top of the card. */
  title?: ReactNode
  /** One line under the title, in plain language. */
  description?: ReactNode
  icon?: IconComponent
  /** Buttons / links on the right of the header. */
  actions?: ReactNode
  /** Inner padding: `none` for edge-to-edge lists and media. */
  padding?: 'none' | 'sm' | 'md' | 'lg'
  as?: 'section' | 'div' | 'article' | 'aside'
}

const PAD = { none: '', sm: 'p-4', md: 'p-5 sm:p-6', lg: 'p-6 sm:p-8' }

/** A white rounded panel — the basic building block of every page. */
export function Card({
  title,
  description,
  icon: Icon,
  actions,
  padding = 'md',
  as: Tag = 'section',
  className,
  children,
  ...rest
}: CardProps) {
  const hasHeader = title || description || actions
  return (
    <Tag className={cx('card min-w-0', PAD[padding], className)} {...rest}>
      {hasHeader && (
        <div className={cx('flex flex-wrap items-start justify-between gap-x-4 gap-y-2', children ? 'mb-5' : '', padding === 'none' && 'px-5 pt-5 sm:px-6')}>
          <div className="flex min-w-0 items-start gap-3">
            {Icon && (
              <span className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent-text">
                <Icon size={17} />
              </span>
            )}
            <div className="min-w-0">
              {title && <h2 className="text-base font-semibold leading-snug text-fg">{title}</h2>}
              {description && <p className="mt-0.5 text-sm leading-relaxed text-fg-muted">{description}</p>}
            </div>
          </div>
          {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </Tag>
  )
}

/** A heading for a group of content that isn't inside a card. */
export function SectionTitle({
  title,
  description,
  actions,
  className,
}: {
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
  className?: string
}) {
  return (
    <div className={cx('mb-4 flex flex-wrap items-end justify-between gap-3', className)}>
      <div className="min-w-0">
        <h2 className="text-lg font-semibold leading-snug text-fg">{title}</h2>
        {description && <p className="mt-0.5 text-sm text-fg-muted">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

/** Label / value list, e.g. "This machine" details. */
export function KeyValue({ items, className }: { items: { label: string; value: ReactNode }[]; className?: string }) {
  return (
    <dl className={cx('grid gap-x-8 gap-y-0 text-sm sm:grid-cols-2', className)}>
      {items.map((item) => (
        <div key={item.label} className="flex items-baseline justify-between gap-4 border-b border-line py-2.5">
          <dt className="text-fg-muted">{item.label}</dt>
          <dd className="numeric min-w-0 truncate text-right font-medium text-fg">{item.value}</dd>
        </div>
      ))}
    </dl>
  )
}
