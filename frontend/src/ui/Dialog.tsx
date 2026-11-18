import { useEffect, useId, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

import { cx } from './cx'
import { XIcon } from './icons'

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/** Escape closes, focus moves in on open and back out on close, Tab stays inside. */
function useModal(open: boolean, onClose: () => void, panel: React.RefObject<HTMLElement>) {
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    const node = panel.current
    const first = node?.querySelector<HTMLElement>('[data-autofocus]') ?? node?.querySelector<HTMLElement>(FOCUSABLE)
    ;(first ?? node)?.focus()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.stopPropagation()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab' || !node) return
      const items = [...node.querySelectorAll<HTMLElement>(FOCUSABLE)].filter((el) => el.offsetParent !== null)
      if (items.length === 0) return
      const head = items[0]
      const tail = items[items.length - 1]
      if (event.shiftKey && document.activeElement === head) {
        event.preventDefault()
        tail.focus()
      } else if (!event.shiftKey && document.activeElement === tail) {
        event.preventDefault()
        head.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
      previous?.focus?.()
    }
  }, [open, panel])
}

/** Centred modal dialog with a title, optional description and a footer of buttons. */
export function Dialog({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  size = 'md',
}: {
  open: boolean
  onClose: () => void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  footer?: ReactNode
  size?: 'sm' | 'md' | 'lg'
}) {
  const panel = useRef<HTMLDivElement>(null)
  const titleId = useId()
  const descId = useId()
  useModal(open, onClose, panel)
  if (!open) return null
  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-end justify-center p-3 sm:items-center sm:p-6">
      <div className="absolute inset-0 bg-scrim [animation:fade-in_160ms_ease_both]" onClick={onClose} aria-hidden />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descId : undefined}
        tabIndex={-1}
        className={cx(
          'relative w-full rounded-2xl border border-line bg-surface p-5 shadow-[var(--shadow-pop)] outline-none [animation:toast-in_220ms_var(--ease-out-expo)_both] sm:p-6',
          size === 'sm' && 'max-w-sm',
          size === 'md' && 'max-w-lg',
          size === 'lg' && 'max-w-2xl',
        )}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h2 id={titleId} className="text-lg font-semibold text-fg">
              {title}
            </h2>
            {description && (
              <p id={descId} className="mt-1 text-sm leading-relaxed text-fg-muted">
                {description}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-mr-1 -mt-1 grid size-8 shrink-0 place-items-center rounded-lg text-fg-muted hover:bg-subtle hover:text-fg"
          >
            <XIcon size={17} />
          </button>
        </div>
        {children && <div className="mt-4">{children}</div>}
        {footer && <div className="mt-6 flex flex-wrap justify-end gap-2">{footer}</div>}
      </div>
    </div>,
    document.body,
  )
}

/** A panel that slides in from the side (mobile navigation, previews). */
export function Sheet({
  open,
  onClose,
  label,
  side = 'left',
  children,
  className,
}: {
  open: boolean
  onClose: () => void
  /** Accessible name. */
  label: string
  side?: 'left' | 'right'
  children: ReactNode
  className?: string
}) {
  const panel = useRef<HTMLDivElement>(null)
  useModal(open, onClose, panel)
  if (!open) return null
  return createPortal(
    <div className="fixed inset-0 z-[60]">
      <div className="absolute inset-0 bg-scrim [animation:fade-in_160ms_ease_both]" onClick={onClose} aria-hidden />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={label}
        tabIndex={-1}
        className={cx(
          'absolute inset-y-0 flex w-[min(20rem,86vw)] flex-col overflow-y-auto bg-surface shadow-[var(--shadow-pop)] outline-none',
          side === 'left' ? 'left-0 [animation:slide-in-left_240ms_var(--ease-out-expo)_both]' : 'right-0',
          className,
        )}
      >
        {children}
      </div>
    </div>,
    document.body,
  )
}
