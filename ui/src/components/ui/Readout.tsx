/**
 * Readout — the product's numeric primitive (§6): value in mono --fs-4,
 * unit in --fs-0 --text-2, ± σ in --fs-0, provenance as a tag
 * (tool_measured / vlm_proposed / human_directed / human_validated).
 * `compact` is the viewport overlay variant (one line, glass background).
 */
import type { ReactNode } from 'react'
import { Badge, type BadgeTone } from './Badge'
import { useFmt, useT } from '../../i18n'

export type Provenance = 'tool_measured' | 'vlm_proposed' | 'human_directed' | 'human_validated'

const PROVENANCE_TONE: Record<Provenance, BadgeTone> = {
  tool_measured: 'measure',
  vlm_proposed: 'warn',
  human_directed: 'brand',
  human_validated: 'ok',
}

export interface ReadoutProps {
  value: number | string
  unit?: string
  digits?: number
  sigma?: number
  label?: ReactNode
  provenance?: Provenance
  compact?: boolean
  tone?: 'measure' | 'default' | 'warn' | 'err' | 'ok'
  className?: string
}

export function Readout({ value, unit, digits = 3, sigma, label, provenance, compact = false, tone = 'measure', className = '' }: ReadoutProps) {
  const fmt = useFmt()
  const t = useT()
  const text = typeof value === 'number' ? fmt.number(value, digits) : value
  return (
    <div className={`stac-readout ${compact ? 'stac-readout--compact' : ''} stac-readout--${tone} ${className}`.trim()}>
      {label && <span className="stac-readout__label">{label}</span>}
      <span className="stac-readout__value">{text}</span>
      {unit && <span className="stac-readout__unit">{unit}</span>}
      {sigma != null && <span className="stac-readout__sigma">± {fmt.number(sigma, digits)}{unit ? ` ${unit}` : ''}</span>}
      {provenance && !compact && <Badge tone={PROVENANCE_TONE[provenance]} size="sm">{t(`provenance.${provenance}`)}</Badge>}
    </div>
  )
}

/** Readout for a length in metres: picks mm / cm / m automatically. */
export function LengthReadout({ metres, sigmaMetres, ...rest }: Omit<ReadoutProps, 'value' | 'unit' | 'sigma'> & { metres: number; sigmaMetres?: number }) {
  const fmt = useFmt()
  const l = fmt.length(metres)
  const sigma = sigmaMetres != null ? fmt.length(sigmaMetres) : null
  return (
    <div className={`stac-readout ${rest.compact ? 'stac-readout--compact' : ''} stac-readout--${rest.tone ?? 'measure'} ${rest.className ?? ''}`.trim()}>
      {rest.label && <span className="stac-readout__label">{rest.label}</span>}
      <span className="stac-readout__value">{l.value}</span>
      <span className="stac-readout__unit">{l.unit}</span>
      {sigma && <span className="stac-readout__sigma">± {sigma.value} {sigma.unit}</span>}
    </div>
  )
}
