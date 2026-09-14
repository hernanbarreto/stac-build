/**
 * IconButton — 28/32 px square button with a mandatory tooltip (§6). The
 * `label` is both the tooltip and the aria-label. `badge` shows a small
 * count (activity bar). `tone` colours the icon for the measurement layer.
 */
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { Loader2 } from 'lucide-react'
import { Tooltip, type TooltipSide } from './Tooltip'
import type { ControlSize } from './Button'

export interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  label: string
  icon: ReactNode
  shortcut?: string
  size?: ControlSize | 'lg'
  variant?: 'ghost' | 'secondary' | 'danger' | 'primary'
  tone?: 'default' | 'measure' | 'brand'
  active?: boolean
  loading?: boolean
  badge?: number | string
  tooltipSide?: TooltipSide
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, shortcut, size = 'md', variant = 'ghost', tone = 'default', active = false, loading = false, badge, tooltipSide, className = '', disabled, type = 'button', ...rest },
  ref,
) {
  const cls = [
    'stac-iconbtn',
    `stac-iconbtn--${size}`,
    `stac-iconbtn--${variant}`,
    tone !== 'default' ? `stac-iconbtn--${tone}` : '',
    active ? 'stac-iconbtn--active' : '',
    className,
  ].filter(Boolean).join(' ')
  return (
    <Tooltip label={label} shortcut={shortcut} side={tooltipSide}>
      <button ref={ref} type={type} className={cls} aria-label={label} aria-pressed={active || undefined} disabled={disabled || loading} {...rest}>
        {loading ? <Loader2 className="stac-spin" aria-hidden /> : icon}
        {badge != null && badge !== 0 && badge !== '' && <span className="stac-iconbtn__badge">{badge}</span>}
      </button>
    </Tooltip>
  )
})
