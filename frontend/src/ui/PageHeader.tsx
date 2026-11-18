import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

import { cx } from './cx'
import { ArrowLeftIcon } from './icons'

export interface PageHeaderProps {
  title: ReactNode
  /** One line saying what this page is for, in plain language. */
  description?: ReactNode
  /** Small line above the title (status, category). */
  eyebrow?: ReactNode
  /** Back link for detail pages: { to: '/projects', label: 'All projects' }. */
  back?: { to: string; label: string }
  /** Main actions, right-aligned on wide screens. */
  actions?: ReactNode
  /** Rendered under the header (e.g. <Tabs>). */
  children?: ReactNode
  className?: string
}

/** The top of every page: title, one-line description, optional back link and actions. */
export function PageHeader({ title, description, eyebrow, back, actions, children, className }: PageHeaderProps) {
  return (
    <header className={cx('mb-6 sm:mb-8', className)}>
      {back && (
        <Link
          to={back.to}
          className="mb-3 inline-flex items-center gap-1.5 rounded-md text-sm font-medium text-fg-muted transition-colors hover:text-accent-text"
        >
          <ArrowLeftIcon size={16} />
          {back.label}
        </Link>
      )}
      {/* Phones: actions sit under the description, so the text keeps the full width. */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
        <div className="min-w-0 flex-1">
          {eyebrow && <div className="mb-1.5 text-sm font-medium text-fg-muted">{eyebrow}</div>}
          <h1 className="text-2xl font-bold leading-tight tracking-tight text-fg [overflow-wrap:anywhere] sm:text-[1.75rem]">
            {title}
          </h1>
          {description && <p className="mt-1.5 max-w-3xl text-[0.9375rem] leading-relaxed text-fg-muted">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children && <div className="mt-5">{children}</div>}
    </header>
  )
}
