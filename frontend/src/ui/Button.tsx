import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { Link, type LinkProps } from 'react-router-dom'

import { cx } from './cx'
import type { IconComponent } from './icons'
import { Spinner } from './Spinner'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type ButtonSize = 'sm' | 'md' | 'lg'

const VARIANT: Record<ButtonVariant, string> = {
  primary: 'btn-primary',
  secondary: 'btn-ghost',
  ghost: 'btn-quiet',
  danger: 'btn-danger',
}
const SIZE: Record<ButtonSize, string> = { sm: 'btn-sm', md: '', lg: 'btn-lg' }

export function buttonClass(variant: ButtonVariant = 'secondary', size: ButtonSize = 'md', extra = '') {
  return cx('btn', VARIANT[variant], SIZE[size], extra)
}

interface CommonProps {
  variant?: ButtonVariant
  size?: ButtonSize
  /** Icon drawn before the label. */
  icon?: IconComponent
  /** Icon drawn after the label. */
  iconRight?: IconComponent
  /** Shows a spinner and disables the button. */
  loading?: boolean
  block?: boolean
  children?: ReactNode
}

export type ButtonProps = CommonProps & ButtonHTMLAttributes<HTMLButtonElement>

/**
 * The one button. `primary` for the main action on a screen (one per area),
 * `secondary` for everything else, `ghost` for low-emphasis actions in lists,
 * `danger` for destructive confirmations.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon: Icon, iconRight: IconRight, loading, block, className, children, disabled, type = 'button', ...rest },
  ref,
) {
  const iconSize = size === 'sm' ? 15 : 17
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={buttonClass(variant, size, cx(block && 'w-full', className))}
      {...rest}
    >
      {loading ? <Spinner size={iconSize - 2} /> : Icon && <Icon size={iconSize} />}
      {children}
      {IconRight && <IconRight size={iconSize} />}
    </button>
  )
})

export type ButtonLinkProps = CommonProps & LinkProps

/** A router link that looks like a button. */
export function ButtonLink({ variant = 'secondary', size = 'md', icon: Icon, iconRight: IconRight, block, className, children, ...rest }: ButtonLinkProps) {
  const iconSize = size === 'sm' ? 15 : 17
  return (
    <Link className={buttonClass(variant, size, cx(block && 'w-full', className))} {...rest}>
      {Icon && <Icon size={iconSize} />}
      {children}
      {IconRight && <IconRight size={iconSize} />}
    </Link>
  )
}

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: IconComponent
  /** Required: the accessible name (also shown as a tooltip). */
  label: string
  variant?: 'ghost' | 'secondary'
  size?: 'sm' | 'md'
}

/** Square icon-only button with a required accessible label. */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { icon: Icon, label, variant = 'ghost', size = 'md', className, type = 'button', ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      aria-label={label}
      title={label}
      className={cx(
        'inline-flex shrink-0 items-center justify-center rounded-[var(--radius-control)] text-fg-muted transition-colors duration-150 hover:bg-subtle hover:text-fg disabled:cursor-not-allowed disabled:opacity-45',
        variant === 'secondary' && 'border border-line-strong bg-surface',
        size === 'sm' ? 'size-8' : 'size-9',
        className,
      )}
      {...rest}
    >
      <Icon size={size === 'sm' ? 16 : 18} />
    </button>
  )
})
