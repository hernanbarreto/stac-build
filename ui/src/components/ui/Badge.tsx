/**
 * Badge / Tag — state (ok / warn / err / info / brand / neutral) rendered as
 * soft fill + text-safe foreground; categorical variant takes an instance
 * colour (runtime data) through the `--badge-color` variable and shows it as
 * a dot beside neutral text, never as coloured text on a coloured fill (§6).
 */
import type { HTMLAttributes, ReactNode } from 'react'

export type BadgeTone = 'neutral' | 'ok' | 'warn' | 'err' | 'info' | 'brand' | 'measure'

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone
  /** categorical colour from the data (hex string) — rendered as a dot */
  color?: string
  dot?: boolean
  mono?: boolean
  size?: 'sm' | 'md'
  children?: ReactNode
}

export function Badge({ tone = 'neutral', color, dot = false, mono = false, size = 'md', className = '', children, ...rest }: BadgeProps) {
  const cls = ['stac-badge', `stac-badge--${tone}`, `stac-badge--${size}`, color ? 'stac-badge--cat' : '', mono ? 'stac-mono' : '', className].filter(Boolean).join(' ')
  return (
    <span className={cls} style={{ '--badge-color': color ?? 'currentColor' } as React.CSSProperties} {...rest}>
      {(dot || color) && <span className="stac-badge__dot" aria-hidden />}
      {children}
    </span>
  )
}

/** Plain colour dot for lists (instance colour from data). */
export function ColorDot({ color, className = '', size = 'md' }: { color: string; className?: string; size?: 'sm' | 'md' | 'lg' }) {
  return <span className={`stac-colordot stac-colordot--${size} ${className}`.trim()} style={{ '--dot-color': color } as React.CSSProperties} aria-hidden />
}
