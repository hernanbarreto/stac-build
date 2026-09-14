/**
 * Legend — colour map with range and unit (§6): vertical in the viewport,
 * horizontal in reports. `ramp` is a list of token names (`--dev-0`…) or
 * runtime colours; `categories` renders a discrete legend (point status,
 * edge kinds) with a swatch per entry.
 */
import type { ReactNode } from 'react'

export interface LegendCategory { color: string; label: ReactNode; value?: ReactNode }

interface LegendProps {
  title?: ReactNode
  ramp?: string[]
  min?: string
  max?: string
  unit?: string
  categories?: LegendCategory[]
  orientation?: 'vertical' | 'horizontal'
  glass?: boolean
  className?: string
}

export function Legend({ title, ramp, min, max, unit, categories, orientation = 'vertical', glass = false, className = '' }: LegendProps) {
  const gradient = ramp?.map(c => (c.startsWith('--') ? `var(${c})` : c)).join(', ')
  const dir = orientation === 'vertical' ? 'to top' : 'to right'
  return (
    <div className={`stac-legend stac-legend--${orientation} ${glass ? 'stac-legend--glass' : ''} ${className}`.trim()} role="img" aria-label={typeof title === 'string' ? title : undefined}>
      {title && <div className="stac-legend__title">{title}</div>}
      {gradient && (
        <div className="stac-legend__ramp-wrap">
          <span className="stac-legend__end stac-legend__end--max">{max}{unit ? ` ${unit}` : ''}</span>
          <div className="stac-legend__ramp" style={{ '--legend-gradient': `linear-gradient(${dir}, ${gradient})` } as React.CSSProperties} />
          <span className="stac-legend__end stac-legend__end--min">{min}{unit ? ` ${unit}` : ''}</span>
        </div>
      )}
      {categories && (
        <ul className="stac-legend__cats">
          {categories.map((c, i) => (
            <li key={i} className="stac-legend__cat">
              <span className="stac-legend__swatch" style={{ '--swatch': c.color.startsWith('--') ? `var(${c.color})` : c.color } as React.CSSProperties} aria-hidden />
              <span className="stac-legend__cat-label">{c.label}</span>
              {c.value != null && <span className="stac-legend__cat-value">{c.value}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
