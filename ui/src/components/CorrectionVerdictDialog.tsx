import { useState } from 'react'
import { AlertTriangle, CheckCircle2, ChevronUp, Wrench } from 'lucide-react'
import { Dialog } from './ui/Dialog'
import { Button } from './ui/Button'
import { Table, type Column } from './ui/Table'
import { Stack } from './ui/Panel'
import { GateList } from '../features/CorrectionDialog'
import { useFmt, useT } from '../i18n'

/** Pending-correction verdict (USER 2026-09-06: centred, like the confirm
 *  dialog, but WITHOUT a blocking backdrop — the user must orbit the cloud
 *  to judge the correction). A collapse button shrinks it to a pill at the
 *  top so the centre of the viewport is free; click reopens.
 *
 *  USER 2026-09-16: there is no approving and no undoing — every epoch of the
 *  session stays on disk and this only picks which one is on screen ("todas
 *  viven, solo se seleccionan y la que se selecciona se muestra"). */
export default function CorrectionVerdictDialog({ state, session, otherSession, busy, epochs, onSelectEpoch }: {
  state: any
  session: string | null
  otherSession: string | null
  busy: boolean
  epochs?: { epoch: number; live: boolean; potree: boolean }[]
  onSelectEpoch: (epoch: number) => void
}) {
  const t = useT()
  const fmt = useFmt()
  const [collapsed, setCollapsed] = useState(false)
  const report = state?.report
  const solutions: any[] = report?.solutions || []
  const gates: any[] = report?.gates || []
  const kind = state?.kind || report?.kind || t('correction.kindFallback')
  const epoch = report?.epoch_to ?? state?.epoch
  const title = otherSession ? t('verdict.pendingIn', { session: otherSession }) : t('verdict.appliedOn', { session: session ?? '' })

  if (collapsed) {
    return (
      <button type="button" className="stac-verdict-pill" onClick={() => setCollapsed(false)} title={t('verdict.showPending')}>
        <AlertTriangle aria-hidden />
        {otherSession ? t('verdict.pendingIn', { session: otherSession }) : t('verdict.pendingPill')}
      </button>
    )
  }

  const columns: Column<any>[] = [
    { id: 'visit', header: t('verdict.visit'), mono: true, cell: c => c.kf_span ? `${c.kf_span[0]}..${c.kf_span[1]}` : c.later_kfs ? `${c.earlier_kfs[0]}..${c.earlier_kfs[1]} / ${c.later_kfs[0]}..${c.later_kfs[1]}` : c.anchors ? t('verdict.anchors', { n: c.anchors.length }) : t('status.dash') },
    { id: 'rigid', header: t('verdict.rigid'), mono: true, cell: c => c.rot_deg != null ? `${fmt.degrees(c.rot_deg)} / ${fmt.number(c.t_norm_m ?? c.t_m ?? 0, 3)} m` : t('status.dash') },
    { id: 'k', header: t('verdict.depthK'), mono: true, align: 'right', cell: c => c.k && c.k !== 1 ? fmt.number(c.k, 4) : t('status.dash') },
    { id: 'copies', header: t('verdict.copies'), mono: true, cell: c => c.residual_cm ? `${fmt.number(c.residual_cm.before, 1)} -> ${fmt.number(c.residual_cm.after, 1)} cm` : c.per_block ? c.per_block.map((b: any) => `R${b.region} ${b.before_cm}->${b.after_cm}`).join(', ') : t('status.dash') },
    { id: 'dof', header: t('verdict.dof'), mono: true, cell: c => c.dof ? c.dof.join(',') : c.n_blocks != null ? t('verdict.blocks', { improved: c.blocks_improved, n: c.n_blocks }) : t('status.dash') },
  ]

  return (
    <Dialog open nonBlocking size="lg" tone="warn" icon={<Wrench aria-hidden />} title={title} busy={busy}
      subtitle={`${kind}${epoch != null ? ` — ${t('status.epoch')} ${epoch}` : ''}${report?.points_moved ? ` — ${t('verdict.pointsMoved', { n: fmt.millions(report.points_moved) })}` : ''}${report?.operator ? ` — ${report.operator}` : ''}`}
      footer={
        <>
          <Button variant="ghost" size="sm" icon={<ChevronUp aria-hidden />} onClick={() => setCollapsed(true)} title={t('verdict.collapseHint')}>{t('verdict.collapse')}</Button>
          {(epochs || []).map(e => (
            <Button
              key={e.epoch}
              variant={e.live ? 'primary' : undefined}
              icon={e.live ? <CheckCircle2 aria-hidden /> : undefined}
              disabled={busy || e.live}
              onClick={() => onSelectEpoch(e.epoch)}
              data-autofocus={e.live || undefined}
            >
              {e.epoch === 0 ? t('correction.epochOriginal') : t('correction.epochN', { epoch: e.epoch })}
            </Button>
          ))}
        </>
      }>
      <Stack gap={3}>
        {solutions.length > 0 && <Table columns={columns} rows={solutions} rowKey={(_, i) => i} maxHeight={false} caption={t('verdict.solutions')} />}
        {gates.length > 0 && <GateList compact gates={gates} />}
        {report?.distribution && (
          <p className="stac-dialog__message">{t('verdict.distribution', { kf: report.distribution.keyframes_warped, from: report.distribution.identity_until_kf, mm: report.distribution.max_step_between_keyframes_mm, deg: report.distribution.max_step_between_keyframes_deg })}</p>
        )}
        <p className="stac-dialog__message">{t('verdict.instruction')}</p>
      </Stack>
    </Dialog>
  )
}
