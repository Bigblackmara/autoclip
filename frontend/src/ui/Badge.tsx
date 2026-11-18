import type { ReactNode } from 'react'

import { cx } from './cx'

export type Tone = 'neutral' | 'accent' | 'success' | 'warn' | 'danger' | 'info'

const TONE: Record<Tone, string> = {
  neutral: 'bg-subtle text-fg-muted ring-line',
  accent: 'bg-accent-soft text-accent-text ring-accent-line',
  success: 'bg-success-soft text-success ring-success/25',
  warn: 'bg-warn-soft text-warn ring-warn/25',
  danger: 'bg-danger-soft text-danger ring-danger/25',
  info: 'bg-info-soft text-info ring-info/25',
}

const DOT: Record<Tone, string> = {
  neutral: 'bg-fg-subtle',
  accent: 'bg-accent',
  success: 'bg-success',
  warn: 'bg-warn',
  danger: 'bg-danger',
  info: 'bg-info',
}

/** Small status pill. `dot` adds a coloured dot; `pulse` animates it (live / running). */
export function Badge({
  tone = 'neutral',
  dot,
  pulse,
  className,
  title,
  children,
}: {
  tone?: Tone
  dot?: boolean
  pulse?: boolean
  className?: string
  title?: string
  children: ReactNode
}) {
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset',
        TONE[tone],
        className,
      )}
    >
      {dot && (
        <span className="relative inline-flex size-1.5">
          {pulse && <span className={cx('absolute inset-0 animate-ping rounded-full opacity-60', DOT[tone])} />}
          <span className={cx('relative size-1.5 rounded-full', DOT[tone])} />
        </span>
      )}
      {children}
    </span>
  )
}
