import type { ReactNode } from 'react'
import { Link, matchPath, useLocation } from 'react-router-dom'

import { cx } from './cx'
import type { IconComponent } from './icons'

export interface SidebarItem {
  label: string
  path: string
  icon: IconComponent
  section: string
  /** Exact match only (for "/"). */
  end?: boolean
  /** Extra path patterns that also light this item up, e.g. ['/jobs/*']. */
  activeOn?: string[]
}

/** Whether `item` is the current page. Query strings in `path` are ignored. */
export function useIsActive() {
  const { pathname } = useLocation()
  return (item: SidebarItem) => {
    const base = item.path.split('?')[0]
    if (matchPath({ path: base, end: item.end ?? false }, pathname)) return true
    return (item.activeOn ?? []).some((pattern) => matchPath({ path: pattern, end: false }, pathname))
  }
}

/**
 * The navigation list, grouped by section. Used by the desktop sidebar and
 * the phone drawer alike. `collapsed` shows icons only (labels become
 * tooltips).
 */
export function SidebarNav({
  sections,
  items,
  collapsed = false,
  onNavigate,
}: {
  sections: readonly string[]
  items: readonly SidebarItem[]
  collapsed?: boolean
  onNavigate?: () => void
}) {
  const isActive = useIsActive()
  return (
    <nav aria-label="Main" className="flex flex-col gap-5">
      {sections.map((section) => {
        const inSection = items.filter((i) => i.section === section)
        if (inSection.length === 0) return null
        return (
          <div key={section}>
            {collapsed ? (
              <div className="mx-auto mb-2 h-px w-6 bg-line" aria-hidden />
            ) : (
              <p className="mb-1.5 px-3 text-[0.6875rem] font-semibold tracking-wider text-fg-subtle uppercase">{section}</p>
            )}
            <ul className="flex flex-col gap-0.5">
              {inSection.map((item) => {
                const active = isActive(item)
                const Icon = item.icon
                return (
                  <li key={item.path}>
                    <Link
                      to={item.path}
                      onClick={onNavigate}
                      aria-current={active ? 'page' : undefined}
                      title={collapsed ? item.label : undefined}
                      className={cx(
                        'group flex items-center gap-3 rounded-[var(--radius-control)] text-sm font-medium transition-colors duration-150',
                        collapsed ? 'mx-auto size-10 justify-center' : 'px-3 py-2',
                        active ? 'bg-accent-soft text-accent-text' : 'text-fg-muted hover:bg-subtle hover:text-fg',
                      )}
                    >
                      <Icon size={18} className={active ? 'text-accent-text' : 'text-fg-subtle group-hover:text-fg-muted'} />
                      {collapsed ? <span className="sr-only">{item.label}</span> : <span className="truncate">{item.label}</span>}
                    </Link>
                  </li>
                )
              })}
            </ul>
          </div>
        )
      })}
    </nav>
  )
}

/** Fixed left rail on wide screens. */
export function Sidebar({
  collapsed,
  header,
  footer,
  children,
}: {
  collapsed: boolean
  header: ReactNode
  footer?: ReactNode
  children: ReactNode
}) {
  return (
    <aside
      className={cx(
        'fixed inset-y-0 left-0 z-30 hidden flex-col border-r border-line bg-surface transition-[width] duration-200 lg:flex',
        collapsed ? 'w-[4.5rem]' : 'w-64',
      )}
    >
      <div className={cx('flex h-16 shrink-0 items-center', collapsed ? 'justify-center px-2' : 'px-5')}>{header}</div>
      <div className={cx('min-h-0 flex-1 overflow-y-auto pb-4', collapsed ? 'px-2' : 'px-3')}>{children}</div>
      {footer && <div className={cx('shrink-0 border-t border-line', collapsed ? 'px-2 py-3' : 'px-3 py-3')}>{footer}</div>}
    </aside>
  )
}
