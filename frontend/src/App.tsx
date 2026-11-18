import { useEffect, useState } from 'react'
import { Link, Outlet, ScrollRestoration, useLocation } from 'react-router-dom'

import { api, type SystemStatus } from './api'
import { NAV_ITEMS, NAV_SECTIONS } from './nav'
import { isRouteRegistered } from './routes'
import {
  Badge,
  CpuIcon,
  IconButton,
  LogoMark,
  MenuIcon,
  MoonIcon,
  PanelLeftIcon,
  Sheet,
  Sidebar,
  SidebarNav,
  SunIcon,
  ToastProvider,
  XIcon,
  cx,
  useTheme,
} from './ui'

const COLLAPSE_KEY = 'autoclip.sidebar.collapsed'

/** Nav entries whose page exists on this build (see nav.tsx). */
const VISIBLE_ITEMS = NAV_ITEMS.filter((item) => isRouteRegistered(item.path))

/**
 * Application shell: a grouped left sidebar on wide screens (collapsible to
 * icons), a top bar with a slide-in drawer on phones and tablets, toasts, and
 * the page outlet.
 */
export function App() {
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [theme, toggleTheme] = useTheme()
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === '1'
    } catch {
      return false
    }
  })
  const [drawer, setDrawer] = useState(false)
  const location = useLocation()

  useEffect(() => {
    api.system().then(setSystem).catch(() => setSystem(null))
  }, [])

  // Close the drawer whenever the page changes (back button included).
  useEffect(() => setDrawer(false), [location.pathname])

  const toggleCollapsed = () =>
    setCollapsed((v) => {
      try {
        localStorage.setItem(COLLAPSE_KEY, v ? '0' : '1')
      } catch {
        /* ignore */
      }
      return !v
    })

  const brand = (compact: boolean) => (
    <Link to="/" className="flex items-center gap-2.5 rounded-lg" aria-label="AutoClip home">
      <LogoMark size={compact ? 32 : 30} />
      {!compact && <span className="text-[1.0625rem] font-bold tracking-tight text-fg">AutoClip</span>}
    </Link>
  )

  const footer = (compact: boolean) => (
    <div className={cx('flex gap-1', compact ? 'flex-col items-center' : 'items-center justify-between')}>
      {system && !compact && <SystemBadge system={system} />}
      {system && compact && (
        <Link
          to="/settings?tab=system"
          title={system.ready ? 'Ready' : 'Not ready — open Settings'}
          className="grid size-9 place-items-center rounded-lg text-fg-subtle hover:bg-subtle"
        >
          <span className={cx('size-2 rounded-full', system.ready ? 'bg-success' : 'bg-danger')} />
        </Link>
      )}
      <div className={cx('flex gap-1', compact && 'flex-col')}>
        <IconButton
          icon={theme === 'dark' ? SunIcon : MoonIcon}
          label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          onClick={toggleTheme}
          size="sm"
        />
        <IconButton
          icon={PanelLeftIcon}
          label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          onClick={toggleCollapsed}
          size="sm"
          className="hidden lg:inline-flex"
        />
      </div>
    </div>
  )

  return (
    <ToastProvider>
      <div className="min-h-screen bg-page">
        <a
          href="#main"
          className="sr-only z-50 rounded-lg bg-surface px-3 py-2 text-sm font-medium focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
        >
          Skip to content
        </a>

        <Sidebar collapsed={collapsed} header={brand(collapsed)} footer={footer(collapsed)}>
          <SidebarNav sections={NAV_SECTIONS} items={VISIBLE_ITEMS} collapsed={collapsed} />
        </Sidebar>

        {/* Phone / tablet top bar */}
        <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-line bg-surface/90 px-3 backdrop-blur lg:hidden">
          <IconButton icon={MenuIcon} label="Open menu" onClick={() => setDrawer(true)} />
          {brand(false)}
          <div className="ml-auto flex items-center gap-1">
            {system && !system.ready && (
              <Link to="/settings?tab=system">
                <Badge tone="danger" dot>
                  Not ready
                </Badge>
              </Link>
            )}
            <IconButton
              icon={theme === 'dark' ? SunIcon : MoonIcon}
              label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              onClick={toggleTheme}
            />
          </div>
        </header>

        <Sheet open={drawer} onClose={() => setDrawer(false)} label="Menu">
          <div className="flex h-14 shrink-0 items-center justify-between border-b border-line px-4">
            {brand(false)}
            <IconButton icon={XIcon} label="Close menu" onClick={() => setDrawer(false)} />
          </div>
          <div className="flex-1 px-3 py-4">
            <SidebarNav sections={NAV_SECTIONS} items={VISIBLE_ITEMS} onNavigate={() => setDrawer(false)} />
          </div>
          {system && (
            <div className="border-t border-line px-4 py-3">
              <SystemBadge system={system} />
            </div>
          )}
        </Sheet>

        <div className={cx('transition-[padding] duration-200', collapsed ? 'lg:pl-[4.5rem]' : 'lg:pl-64')}>
          <main id="main" className="mx-auto w-full max-w-[1440px] px-4 pt-5 pb-20 sm:px-6 sm:pt-7 lg:px-10 lg:pt-9">
            <Outlet />
          </main>
        </div>
        <ScrollRestoration />
      </div>
    </ToastProvider>
  )
}

/**
 * Compact machine status: whether ffmpeg can render and whether the GPU is
 * used — the two things that change how long everything takes.
 */
function SystemBadge({ system }: { system: SystemStatus }) {
  const accel = system.accel.toUpperCase()
  const gpu = system.gpu_name && accel === 'CUDA' ? system.gpu_name.replace('NVIDIA GeForce ', '') : null
  return (
    <Link
      to="/settings?tab=system"
      className="flex min-w-0 items-center gap-2.5 rounded-lg px-2 py-1.5 transition-colors hover:bg-subtle"
      title={system.ready ? 'All required components present' : 'Something is missing — open Settings → This machine'}
    >
      <span className="grid size-7 shrink-0 place-items-center rounded-md bg-subtle text-fg-muted">
        <CpuIcon size={15} />
      </span>
      <span className="min-w-0">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-fg">
          <span className={cx('size-1.5 rounded-full', system.ready ? 'bg-success' : 'bg-danger')} />
          {system.ready ? 'Ready' : 'Not ready'}
        </span>
        <span className="numeric block truncate text-[0.6875rem] text-fg-muted">
          {accel}
          {gpu && ` · ${gpu}`}
        </span>
      </span>
    </Link>
  )
}
