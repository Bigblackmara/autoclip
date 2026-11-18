import { useId, useRef, type ReactNode } from 'react'

import { cx } from './cx'
import type { IconComponent } from './icons'

export interface TabItem<K extends string> {
  key: K
  label: ReactNode
  icon?: IconComponent
  /** Small count or dot after the label. */
  badge?: ReactNode
}

/**
 * Accessible tab bar (arrow keys, Home/End). Panels are rendered by the
 * caller; give each panel `{...tabPanelProps(idBase, key)}`.
 *
 * `variant="underline"` for page sub-navigation, `"pill"` for compact
 * switches inside a card. On narrow screens the bar scrolls sideways.
 */
export function Tabs<K extends string>({
  items,
  value,
  onChange,
  label,
  idBase,
  variant = 'underline',
  className,
}: {
  items: readonly TabItem<K>[]
  value: K
  onChange: (key: K) => void
  /** Accessible name of the tab list. */
  label: string
  /** Stable id prefix shared with the panels (see tabPanelProps). */
  idBase?: string
  /** `side`: a scrolling pill row on small screens, a vertical list from `lg` up (settings-style). */
  variant?: 'underline' | 'pill' | 'side'
  className?: string
}) {
  const fallback = useId()
  const base = idBase ?? fallback
  const refs = useRef(new Map<K, HTMLButtonElement>())

  const onKeyDown = (event: React.KeyboardEvent, index: number) => {
    const last = items.length - 1
    const target =
      event.key === 'ArrowRight' || event.key === 'ArrowDown'
        ? index === last
          ? 0
          : index + 1
        : event.key === 'ArrowLeft' || event.key === 'ArrowUp'
          ? index === 0
            ? last
            : index - 1
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? last
              : null
    if (target === null) return
    event.preventDefault()
    const next = items[target].key
    onChange(next)
    refs.current.get(next)?.focus()
  }

  return (
    <div
      role="tablist"
      aria-label={label}
      className={cx(
        'flex max-w-full overflow-x-auto [scrollbar-width:none]',
        variant === 'underline' && 'gap-1 border-b border-line',
        variant === 'pill' && 'inline-flex gap-1 rounded-[var(--radius-control)] bg-subtle p-1',
        variant === 'side' && 'gap-1 lg:flex-col lg:overflow-visible',
        className,
      )}
    >
      {items.map((item, index) => {
        const active = item.key === value
        const Icon = item.icon
        return (
          <button
            key={item.key}
            ref={(el) => {
              if (el) refs.current.set(item.key, el)
              else refs.current.delete(item.key)
            }}
            type="button"
            role="tab"
            id={`${base}-tab-${item.key}`}
            aria-selected={active}
            aria-controls={`${base}-panel-${item.key}`}
            tabIndex={active ? 0 : -1}
            onClick={() => onChange(item.key)}
            onKeyDown={(e) => onKeyDown(e, index)}
            className={cx(
              'inline-flex shrink-0 items-center gap-2 whitespace-nowrap text-sm font-medium transition-colors duration-150',
              variant === 'underline'
                ? cx(
                    '-mb-px border-b-2 px-3 py-2.5',
                    active ? 'border-accent text-accent-text' : 'border-transparent text-fg-muted hover:border-line-strong hover:text-fg',
                  )
                : variant === 'pill'
                  ? cx('rounded-lg px-3 py-1.5', active ? 'bg-surface text-fg shadow-sm ring-1 ring-line' : 'text-fg-muted hover:text-fg')
                  : cx(
                      'rounded-[var(--radius-control)] px-3 py-2 lg:w-full',
                      active ? 'bg-accent-soft text-accent-text' : 'text-fg-muted hover:bg-subtle hover:text-fg',
                    ),
            )}
          >
            {Icon && <Icon size={16} />}
            {item.label}
            {item.badge}
          </button>
        )
      })}
    </div>
  )
}

/** Props for a tab panel matching a <Tabs idBase>. */
export function tabPanelProps(idBase: string, key: string) {
  return {
    role: 'tabpanel' as const,
    id: `${idBase}-panel-${key}`,
    'aria-labelledby': `${idBase}-tab-${key}`,
  }
}
