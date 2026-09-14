/**
 * Panel / Section — every panel has a title (--fs-3), actions on the right,
 * a body with its own scroll, an empty state with an instruction and a
 * skeleton while loading (§5). Section is the sentence-case sub-heading
 * inside a panel body (--fs-1, medium, --text-2).
 */
import type { ReactNode } from 'react'
import { Skeleton } from './Skeleton'

interface PanelProps {
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  toolbar?: ReactNode
  footer?: ReactNode
  loading?: boolean
  empty?: ReactNode
  isEmpty?: boolean
  padded?: boolean
  className?: string
  bodyClassName?: string
  children?: ReactNode
}

export function Panel({ title, subtitle, actions, toolbar, footer, loading = false, empty, isEmpty = false, padded = false, className = '', bodyClassName = '', children }: PanelProps) {
  return (
    <section className={`stac-panel ${className}`.trim()}>
      <header className="stac-panel__header">
        <div className="stac-panel__titles">
          <h2 className="stac-panel__title">{title}</h2>
          {subtitle && <div className="stac-panel__subtitle">{subtitle}</div>}
        </div>
        {actions && <div className="stac-panel__actions">{actions}</div>}
      </header>
      {toolbar && <div className="stac-panel__toolbar">{toolbar}</div>}
      <div className={`stac-panel__body ${padded ? 'stac-panel__body--padded' : ''} ${bodyClassName}`.trim()}>
        {loading ? <div className="stac-panel__skeleton"><Skeleton lines={6} /></div> : isEmpty && empty ? empty : children}
      </div>
      {footer && <footer className="stac-panel__footer">{footer}</footer>}
    </section>
  )
}

interface SectionProps {
  title: ReactNode
  actions?: ReactNode
  hint?: ReactNode
  className?: string
  children?: ReactNode
  flush?: boolean
}

export function Section({ title, actions, hint, className = '', flush = false, children }: SectionProps) {
  return (
    <div className={`stac-section ${flush ? 'stac-section--flush' : ''} ${className}`.trim()}>
      <div className="stac-section__header">
        <h3 className="stac-section__title">{title}</h3>
        {actions && <div className="stac-section__actions">{actions}</div>}
      </div>
      {hint && <p className="stac-section__hint">{hint}</p>}
      <div className="stac-section__body">{children}</div>
    </div>
  )
}

/** Key-value row for property lists. */
export function KeyValue({ label, children, mono = false }: { label: ReactNode; children: ReactNode; mono?: boolean }) {
  return (
    <div className="stac-kv">
      <span className="stac-kv__label">{label}</span>
      <span className={`stac-kv__value ${mono ? 'stac-mono' : ''}`.trim()}>{children}</span>
    </div>
  )
}

/** Horizontal group of controls (gap --sp-2). */
export function Row({ children, className = '', wrap = false, justify }: { children: ReactNode; className?: string; wrap?: boolean; justify?: 'start' | 'end' | 'between' | 'center' }) {
  return <div className={`stac-row ${wrap ? 'stac-row--wrap' : ''} ${justify ? `stac-row--${justify}` : ''} ${className}`.trim()}>{children}</div>
}

/** Vertical stack (gap --sp-2). */
export function Stack({ children, className = '', gap = 2 }: { children: ReactNode; className?: string; gap?: 1 | 2 | 3 | 4 | 6 }) {
  return <div className={`stac-stack stac-stack--${gap} ${className}`.trim()}>{children}</div>
}

/** Thin vertical rule used as a separator between inline metadata (replaces the middle-dot). */
export function Sep() {
  return <span className="stac-sep" aria-hidden />
}
