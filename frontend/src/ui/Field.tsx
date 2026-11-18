import {
  forwardRef,
  useId,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from 'react'

import { cx } from './cx'

/**
 * Label + control + helper text. Pass the control as children; the label is
 * linked to it through `htmlFor` (give the control the same `id`), or, when
 * no id is given, by wrapping it.
 */
export function Field({
  label,
  hint,
  error,
  id,
  className,
  children,
  optional,
}: {
  label: ReactNode
  hint?: ReactNode
  error?: ReactNode
  id?: string
  className?: string
  optional?: boolean
  children: ReactNode
}) {
  const body = (
    <>
      <span className="mb-1.5 flex items-baseline gap-2 text-sm font-medium text-fg">
        {label}
        {optional && <span className="text-xs font-normal text-fg-subtle">optional</span>}
      </span>
      {children}
      {error ? (
        <span className="mt-1.5 block text-xs leading-snug text-danger" role="alert">
          {error}
        </span>
      ) : (
        hint && <span className="mt-1.5 block text-xs leading-snug text-fg-muted">{hint}</span>
      )}
    </>
  )
  return (
    <label htmlFor={id} className={cx('block min-w-0', className)}>
      {body}
    </label>
  )
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(
  { className, type = 'text', ...rest },
  ref,
) {
  return <input ref={ref} type={type} className={cx('field', type === 'number' && 'numeric', className)} {...rest} />
})

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea(
  { className, ...rest },
  ref,
) {
  return <textarea ref={ref} className={cx('field min-h-24 leading-relaxed', className)} {...rest} />
})

export interface SelectOption {
  value: string
  label: string
  disabled?: boolean
}

export const Select = forwardRef<
  HTMLSelectElement,
  SelectHTMLAttributes<HTMLSelectElement> & { options?: SelectOption[] }
>(function Select({ className, options, children, ...rest }, ref) {
  return (
    <select ref={ref} className={cx('field cursor-pointer', className)} {...rest}>
      {options
        ? options.map((o) => (
            <option key={o.value} value={o.value} disabled={o.disabled}>
              {o.label}
            </option>
          ))
        : children}
    </select>
  )
})

/**
 * An on/off switch with a label and optional helper line. Renders a real
 * checkbox (role="switch") so forms, keyboard and screen readers just work.
 */
export function Toggle({
  label,
  hint,
  checked,
  onChange,
  disabled,
  id,
  className,
  inputRef,
}: {
  label: ReactNode
  hint?: ReactNode
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  id?: string
  className?: string
  inputRef?: React.Ref<HTMLInputElement>
}) {
  const fallback = useId()
  const inputId = id ?? fallback
  const hintId = `${inputId}-hint`
  return (
    <div className={cx('flex items-start gap-3', disabled && 'opacity-60', className)}>
      <span className="relative mt-0.5 inline-flex shrink-0">
        <input
          ref={inputRef}
          id={inputId}
          type="checkbox"
          role="switch"
          checked={checked}
          disabled={disabled}
          aria-describedby={hint ? hintId : undefined}
          onChange={(e) => onChange(e.target.checked)}
          className="peer absolute inset-0 z-10 size-full cursor-pointer opacity-0 disabled:cursor-not-allowed"
        />
        <span
          aria-hidden
          className="h-5 w-9 rounded-full bg-line-strong transition-colors duration-200 peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent"
        />
        <span
          aria-hidden
          className="pointer-events-none absolute top-0.5 left-0.5 size-4 rounded-full bg-white shadow-sm transition-transform duration-200 peer-checked:translate-x-4"
        />
      </span>
      <span className="min-w-0">
        <label htmlFor={inputId} className="cursor-pointer text-sm font-medium leading-snug text-fg">
          {label}
        </label>
        {hint && (
          <span id={hintId} className="mt-0.5 block text-xs leading-snug text-fg-muted">
            {hint}
          </span>
        )}
      </span>
    </div>
  )
}

/** Checkbox with a label and optional helper text. */
export function Checkbox({
  label,
  hint,
  checked,
  onChange,
  disabled,
  className,
}: {
  label: ReactNode
  hint?: ReactNode
  checked: boolean
  onChange: (checked: boolean) => void
  disabled?: boolean
  className?: string
}) {
  return (
    <label className={cx('flex cursor-pointer items-start gap-3', disabled && 'cursor-not-allowed opacity-60', className)}>
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 size-4 shrink-0 rounded"
      />
      <span className="min-w-0">
        <span className="text-sm leading-snug text-fg">{label}</span>
        {hint && <span className="mt-0.5 block text-xs leading-snug text-fg-muted">{hint}</span>}
      </span>
    </label>
  )
}

/** A row of mutually exclusive choices (radio group look-alike). */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  size = 'md',
  className,
  disabled,
}: {
  options: readonly { value: T; label: ReactNode; disabled?: boolean; title?: string }[]
  value: T
  onChange: (value: T) => void
  /** Accessible group name. */
  label: string
  size?: 'sm' | 'md'
  className?: string
  disabled?: boolean
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cx('inline-flex max-w-full flex-wrap gap-1 rounded-[var(--radius-control)] bg-subtle p-1', className)}
    >
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={o.title}
            disabled={disabled || o.disabled}
            onClick={() => onChange(o.value)}
            className={cx(
              'rounded-lg font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-1.5 text-sm',
              active ? 'bg-surface text-fg shadow-sm ring-1 ring-line' : 'text-fg-muted hover:text-fg',
            )}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
