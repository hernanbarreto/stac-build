/**
 * GateList — the measured gates of a correction epoch as the certification
 * kit shows them: pass / fail, or an advisory warning (measured, applied
 * anyway — the user's eye on the epochs is the verdict).
 */
import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import { useT } from '../i18n'

export interface GateRow { name: string; group?: string; passed: boolean; advisory?: boolean; detail?: string }

export function GateList({ gates, compact = false }: { gates: GateRow[]; compact?: boolean }) {
  const t = useT()
  return (
    <ul className={`stac-gates ${compact ? 'stac-gates--compact' : ''}`.trim()}>
      {gates.map(g => (
        <li key={g.name + String(g.group ?? '')} className="stac-gates__row">
          <span className={`stac-gates__icon stac-gates__icon--${g.advisory && !g.passed ? 'warn' : g.passed ? 'ok' : 'err'}`}>
            {g.advisory && !g.passed ? <AlertTriangle aria-hidden /> : g.passed ? <CheckCircle2 aria-hidden /> : <XCircle aria-hidden />}
          </span>
          <span className="stac-gates__name">{g.name}</span>
          <span className="stac-gates__detail">{g.detail}{g.advisory && !g.passed ? ` ${t('certify.advisoryNote')}` : ''}</span>
        </li>
      ))}
    </ul>
  )
}
