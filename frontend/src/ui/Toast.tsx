import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'

import { cx } from './cx'
import { AlertCircleIcon, CheckCircleIcon, InfoIcon, XIcon } from './icons'

export type ToastTone = 'success' | 'error' | 'info'

interface ToastItem {
  id: number
  tone: ToastTone
  title: string
  description?: string
}

interface ToastApi {
  show: (tone: ToastTone, title: string, description?: string) => void
  success: (title: string, description?: string) => void
  error: (title: string, description?: string) => void
  info: (title: string, description?: string) => void
}

const ToastContext = createContext<ToastApi | null>(null)

/**
 * `const toast = useToast(); toast.success('Saved')`. Works anywhere under
 * <ToastProvider> (the app shell). Outside it, calls are silently ignored.
 */
export function useToast(): ToastApi {
  return useContext(ToastContext) ?? NOOP
}

const NOOP: ToastApi = { show: () => undefined, success: () => undefined, error: () => undefined, info: () => undefined }

const LOOK: Record<ToastTone, { icon: typeof InfoIcon; color: string }> = {
  success: { icon: CheckCircleIcon, color: 'text-success' },
  error: { icon: AlertCircleIcon, color: 'text-danger' },
  info: { icon: InfoIcon, color: 'text-info' },
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const nextId = useRef(0)

  const dismiss = useCallback((id: number) => setItems((all) => all.filter((t) => t.id !== id)), [])

  const show = useCallback(
    (tone: ToastTone, title: string, description?: string) => {
      const id = (nextId.current += 1)
      // The same message again replaces the old one instead of stacking up
      // (e.g. "Saved" after every field in a row).
      setItems((all) => [
        ...all.filter((t) => !(t.title === title && t.tone === tone && t.description === description)).slice(-3),
        { id, tone, title, description },
      ])
      window.setTimeout(() => dismiss(id), tone === 'error' ? 9000 : 4000)
    },
    [dismiss],
  )

  const api = useMemo<ToastApi>(
    () => ({
      show,
      success: (t, d) => show('success', t, d),
      error: (t, d) => show('error', t, d),
      info: (t, d) => show('info', t, d),
    }),
    [show],
  )

  return (
    <ToastContext.Provider value={api}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-3 bottom-3 z-[70] flex flex-col items-stretch gap-2 sm:inset-x-auto sm:right-5 sm:bottom-5 sm:w-96"
      >
        {items.map((t) => {
          const look = LOOK[t.tone]
          const Icon = look.icon
          return (
            <div
              key={t.id}
              role={t.tone === 'error' ? 'alert' : 'status'}
              className="pointer-events-auto flex items-start gap-3 rounded-xl border border-line bg-surface px-4 py-3 shadow-[var(--shadow-pop)] [animation:toast-in_260ms_var(--ease-out-expo)_both]"
            >
              <Icon size={18} className={cx('mt-0.5', look.color)} />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-fg">{t.title}</p>
                {t.description && (
                  <p className="mt-0.5 whitespace-pre-line text-sm leading-snug text-fg-muted [overflow-wrap:anywhere]">
                    {t.description}
                  </p>
                )}
              </div>
              <button
                type="button"
                onClick={() => dismiss(t.id)}
                aria-label="Dismiss"
                className="-mr-1 grid size-7 shrink-0 place-items-center rounded-md text-fg-subtle transition-colors hover:bg-subtle hover:text-fg"
              >
                <XIcon size={15} />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}
