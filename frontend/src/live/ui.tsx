/**
 * Small UI pieces the Live Watch panels share: the screen-reader announcer,
 * the toast, the inline confirm, and a couple of glyphs.
 */

import { useEffect, useId, useRef, useState } from 'react'

// --------------------------------------------------------------------------
// Announcements — one polite live region for the whole page
// --------------------------------------------------------------------------

const announceListeners = new Set<(message: string) => void>()

/** Queue a polite screen-reader announcement (new clips, action results). */
export function announce(message: string) {
  for (const listener of announceListeners) listener(message)
}

/**
 * The polite live region. Each message is its own node and expires after a
 * few seconds, so two announcements in a row are both read rather than the
 * second overwriting the first.
 */
export function LiveAnnouncer() {
  const [items, setItems] = useState<{ id: number; text: string }[]>([])
  const nextId = useRef(0)

  useEffect(() => {
    const timers = new Set<number>()
    const listener = (text: string) => {
      const id = (nextId.current += 1)
      setItems((current) => [...current.slice(-4), { id, text }])
      const timer = window.setTimeout(() => {
        timers.delete(timer)
        setItems((current) => current.filter((item) => item.id !== id))
      }, 7_000)
      timers.add(timer)
    }
    announceListeners.add(listener)
    return () => {
      announceListeners.delete(listener)
      for (const timer of timers) window.clearTimeout(timer)
    }
  }, [])

  return (
    <div className="sr-only" aria-live="polite">
      {items.map((item) => (
        <p key={item.id}>{item.text}</p>
      ))}
    </div>
  )
}

// --------------------------------------------------------------------------
// Toast
// --------------------------------------------------------------------------

export interface ToastMessage {
  id: number
  tone: 'good' | 'bad'
  text: string
}

/** A single, self-dismissing notice pinned to the bottom of the viewport. */
export function Toast({ toast, onDismiss }: { toast: ToastMessage | null; onDismiss: () => void }) {
  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(onDismiss, toast.tone === 'bad' ? 12_000 : 7_000)
    return () => window.clearTimeout(timer)
  }, [toast, onDismiss])

  if (!toast) return null
  return (
    <div className="pointer-events-none fixed inset-x-4 bottom-4 z-40 flex justify-end sm:inset-x-auto sm:right-6 sm:bottom-6">
      <div
        role="status"
        className={`rise pointer-events-auto flex max-w-md items-start gap-4 border-l-2 bg-ink-800 py-3 pr-2 pl-4 shadow-[0_12px_40px_-12px_rgb(0_0_0/0.7)] ${
          toast.tone === 'good' ? 'border-signal-good' : 'border-signal-bad'
        }`}
      >
        <p className="pt-0.5 text-sm leading-relaxed text-ink-100">{toast.text}</p>
        <button onClick={onDismiss} className="btn btn-quiet shrink-0 px-2 py-0.5" aria-label="Dismiss">
          ✕
        </button>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Inline confirm
// --------------------------------------------------------------------------

/**
 * A destructive action's second step, in place. Focus lands on the confirm
 * button the moment it appears (as with a native confirm dialog) and Escape
 * backs out; the caller puts focus back where it came from.
 */
export function ConfirmInline({
  question,
  detail,
  confirmLabel,
  confirmName,
  cancelLabel = 'Cancel',
  busy = false,
  onConfirm,
  onCancel,
  className = '',
}: {
  question: string
  detail?: string
  confirmLabel: string
  /** Accessible name for the confirm button when the visible label is terse. */
  confirmName?: string
  cancelLabel?: string
  busy?: boolean
  onConfirm: () => void
  onCancel: () => void
  className?: string
}) {
  const confirmRef = useRef<HTMLButtonElement>(null)
  const questionId = useId()

  useEffect(() => {
    confirmRef.current?.focus()
  }, [])

  return (
    <div
      role="group"
      aria-labelledby={questionId}
      onKeyDown={(event) => {
        if (event.key === 'Escape') {
          event.stopPropagation()
          onCancel()
        }
      }}
      className={`border-l-2 border-signal-bad pl-3 ${className}`}
    >
      <p id={questionId} className="text-xs leading-snug text-ink-100">
        {question}
      </p>
      {detail && <p className="mt-0.5 text-xs leading-snug text-ink-500">{detail}</p>}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button
          ref={confirmRef}
          onClick={onConfirm}
          disabled={busy}
          aria-label={confirmName}
          className="btn border border-signal-bad/50 bg-signal-bad/10 px-3 py-1 text-xs text-signal-bad hover:bg-signal-bad/20"
        >
          {confirmLabel}
        </button>
        <button onClick={onCancel} disabled={busy} className="btn btn-quiet px-2 py-1 text-xs">
          {cancelLabel}
        </button>
      </div>
    </div>
  )
}

// --------------------------------------------------------------------------
// Glyphs and tiny hooks
// --------------------------------------------------------------------------

export function Spinner({ className = '' }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={`inline-block size-3 shrink-0 animate-spin rounded-full border border-ink-600 border-t-sodium-500 ${className}`}
    />
  )
}

export function SpeakerIcon({ muted }: { muted: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
      <path
        d="M11 5 6 9H3v6h3l5 4V5Z"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      {muted ? (
        <path d="m16 9 5 6m0-6-5 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      ) : (
        <>
          <path d="M15.5 9.5a3.5 3.5 0 0 1 0 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          <path d="M18.5 7a7 7 0 0 1 0 10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </>
      )}
    </svg>
  )
}

/** Re-render every `intervalMs`, for relative timestamps. */
export function useNow(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}

/** scrollIntoView that respects reduced motion. */
export function scrollToElement(element: Element, block: ScrollLogicalPosition = 'center') {
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
  element.scrollIntoView({ block, behavior: reduce ? 'auto' : 'smooth' })
}
