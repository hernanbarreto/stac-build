/**
 * Progress — a bar with an optional list of named stages (pipeline:
 * reconstruction -> segmentation -> …) and the percentage in mono (§6).
 * Stage status drives the colour: running = brand, done = ok, failed = err.
 */
import { Check, Loader2, X } from 'lucide-react'
import { useFmt } from '../../i18n'

export interface ProgressStage {
  id: string
  label: string
  status: 'pending' | 'running' | 'done' | 'failed' | 'skipped' | string
  pct: number
  detail?: string
}

interface ProgressProps {
  value?: number
  tone?: 'brand' | 'measure' | 'ok' | 'warn' | 'err'
  size?: 'sm' | 'md'
  showValue?: boolean
  label?: string
  stages?: ProgressStage[]
  className?: string
}

export function Progress({ value, tone = 'brand', size = 'sm', showValue = true, label, stages, className = '' }: ProgressProps) {
  const fmt = useFmt()
  const pct = Math.max(0, Math.min(100, value ?? (stages ? avg(stages) : 0)))
  return (
    <div className={`stac-progress stac-progress--${tone} stac-progress--${size} ${className}`.trim()} role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label={label}>
      {(label || showValue) && (
        <div className="stac-progress__head">
          {label && <span className="stac-progress__label">{label}</span>}
          {showValue && <span className="stac-progress__value">{fmt.percent(pct / 100)}</span>}
        </div>
      )}
      <div className="stac-progress__track">
        <div className="stac-progress__fill" style={{ '--pct': `${pct}%` } as React.CSSProperties} />
      </div>
      {stages && stages.length > 0 && (
        <ol className="stac-progress__stages">
          {stages.map(s => (
            <li key={s.id} className={`stac-progress__stage stac-progress__stage--${s.status}`}>
              <span className="stac-progress__stage-icon" aria-hidden>
                {s.status === 'done' ? <Check /> : s.status === 'failed' ? <X /> : s.status === 'running' ? <Loader2 className="stac-spin" /> : <span className="stac-progress__stage-dot" />}
              </span>
              <span className="stac-progress__stage-label" title={s.detail}>{s.label}</span>
              <span className="stac-progress__stage-track"><span className="stac-progress__stage-fill" style={{ '--pct': `${Math.max(0, Math.min(100, s.pct))}%` } as React.CSSProperties} /></span>
              <span className="stac-progress__stage-pct">{fmt.percent(Math.max(0, Math.min(100, s.pct)) / 100)}</span>
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

function avg(stages: ProgressStage[]): number {
  const enabled = stages.filter(s => s.status !== 'skipped')
  if (!enabled.length) return 0
  return enabled.reduce((a, s) => a + (s.status === 'done' ? 100 : s.pct), 0) / enabled.length
}
