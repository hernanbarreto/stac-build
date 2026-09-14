/**
 * Button — primary (brand orange, inverse text, the one allowed glow),
 * secondary (surface-2 + control border), ghost, danger. Sizes follow the
 * density (--ctl-h / --ctl-h-sm). `loading` swaps the icon for a spinner and
 * keeps the text (§6).
 */
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { Loader2 } from 'lucide-react'

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export type ControlSize = 'sm' | 'md'

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  size?: ControlSize
  icon?: ReactNode
  iconRight?: ReactNode
  loading?: boolean
  block?: boolean
  active?: boolean
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', icon, iconRight, loading = false, block = false, active = false, className = '', children, disabled, type = 'button', ...rest },
  ref,
) {
  const cls = [
    'stac-btn',
    `stac-btn--${variant}`,
    `stac-btn--${size}`,
    block ? 'stac-btn--block' : '',
    active ? 'stac-btn--active' : '',
    loading ? 'stac-btn--loading' : '',
    !children ? 'stac-btn--icon-only' : '',
    className,
  ].filter(Boolean).join(' ')
  return (
    <button ref={ref} type={type} className={cls} disabled={disabled || loading} aria-busy={loading || undefined} aria-pressed={active || undefined} {...rest}>
      {loading ? <Loader2 className="stac-btn__icon stac-spin" aria-hidden /> : icon ? <span className="stac-btn__icon" aria-hidden>{icon}</span> : null}
      {children != null && <span className="stac-btn__label">{children}</span>}
      {iconRight && <span className="stac-btn__icon" aria-hidden>{iconRight}</span>}
    </button>
  )
})
