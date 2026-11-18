import type { ReactNode } from 'react'

import { cx } from './cx'
import { AlertCircleIcon, AlertIcon, CheckCircleIcon, InfoIcon, type IconComponent } from './icons'

/** "Nothing here yet" with a friendly line and the one action that fixes it. */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
  compact,
}: {
  icon?: IconComponent
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
  compact?: boolean
}) {
  return (
    <div
      className={cx(
        'flex flex-col items-center justify-center rounded-[var(--radius-card)] border border-dashed border-line-strong text-center',
        compact ? 'px-4 py-8' : 'px-6 py-14',
        className,
      )}
    >
      {Icon && (
        <span className="mb-4 grid size-12 place-items-center rounded-2xl bg-accent-soft text-accent-text">
          <Icon size={22} />
        </span>
      )}
      <p className="text-base font-semibold text-fg">{title}</p>
      {description && <p className="mt-1.5 max-w-md text-sm leading-relaxed text-fg-muted">{description}</p>}
      {action && <div className="mt-5 flex flex-wrap justify-center gap-2">{action}</div>}
    </div>
  )
}

const CALLOUT = {
  info: { box: 'border-info/25 bg-info-soft', icon: InfoIcon, iconClass: 'text-info' },
  warn: { box: 'border-warn/30 bg-warn-soft', icon: AlertIcon, iconClass: 'text-warn' },
  danger: { box: 'border-danger/30 bg-danger-soft', icon: AlertCircleIcon, iconClass: 'text-danger' },
  success: { box: 'border-success/30 bg-success-soft', icon: CheckCircleIcon, iconClass: 'text-success' },
} as const

/** A tinted notice box: tips (info), heads-ups (warn), errors (danger), confirmations (success). */
export function Callout({
  tone = 'info',
  title,
  children,
  action,
  onDismiss,
  className,
  role,
}: {
  tone?: keyof typeof CALLOUT
  title?: ReactNode
  children?: ReactNode
  action?: ReactNode
  onDismiss?: () => void
  className?: string
  role?: 'alert' | 'status'
}) {
  const look = CALLOUT[tone]
  const Icon = look.icon
  return (
    <div role={role} className={cx('flex gap-3 rounded-xl border px-4 py-3', look.box, className)}>
      <Icon size={18} className={cx('mt-0.5', look.iconClass)} />
      <div className="min-w-0 flex-1 text-sm leading-relaxed text-fg">
        {title && <p className="font-semibold">{title}</p>}
        {children && <div className={cx(title ? 'mt-0.5 text-fg-muted' : '', '[overflow-wrap:anywhere]')}>{children}</div>}
        {action && <div className="mt-2.5 flex flex-wrap gap-2">{action}</div>}
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="-mr-1 -mt-0.5 grid size-7 shrink-0 place-items-center rounded-md text-fg-muted transition-colors hover:bg-surface/60 hover:text-fg"
        >
          ✕
        </button>
      )}
    </div>
  )
}
