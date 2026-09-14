/**
 * ScansPanel — the scans of the loaded session (left "Scans" tab) and, in
 * `timeline` mode, the same data laid out by date for the bottom dock.
 * Reference scan marked; double-click / Open activates the scan as a tab;
 * the Fuse button needs two non-fused scans.
 */
import { Layers, Star } from 'lucide-react'
import type { ScanRow } from './SessionsPanel'
import { Panel, Row } from '../components/ui/Panel'
import { Table, type Column } from '../components/ui/Table'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

interface ScansPanelProps {
  sessionId: string | null
  scans: ScanRow[]
  activeScanKey: string | null
  onOpenScan: (scan: ScanRow) => void
  onSetReference: (scan: ScanRow) => void
  onFuse: () => void
  variant?: 'panel' | 'timeline'
}

export function ScansPanel({ sessionId, scans, activeScanKey, onOpenScan, onSetReference, onFuse, variant = 'panel' }: ScansPanelProps) {
  const t = useT()
  const fmt = useFmt()
  const canFuse = scans.filter(s => s.kind !== 'fused').length >= 2

  if (variant === 'timeline') {
    const ordered = [...scans].sort((a, b) => a.date.localeCompare(b.date))
    return (
      <div className="stac-timeline">
        {!sessionId && <EmptyState compact title={t('scans.noSession')} />}
        {sessionId && ordered.length === 0 && <EmptyState compact title={t('scans.emptyTitle')} />}
        <ol className="stac-timeline__track">
          {ordered.map(sc => (
            <li key={sc.key} className={`stac-timeline__item ${sc.key === activeScanKey ? 'stac-timeline__item--active' : ''} ${sc.kind === 'fused' ? 'stac-timeline__item--fused' : ''}`.trim()}>
              <button type="button" className="stac-timeline__node" onClick={() => onOpenScan(sc)} title={t('scans.openHint')}>
                <span className="stac-timeline__date stac-mono">{sc.kind === 'fused' ? t('header.fused') : sc.date}</span>
                <span className="stac-timeline__label">{sc.label}</span>
                <span className="stac-timeline__meta stac-mono">{sc.points ? t('status.points', { n: fmt.millions(sc.points) }) : sc.recon_state ? t(`reconState.${sc.recon_state}`) : ''}</span>
                {sc.is_reference && <Badge tone="brand" size="sm">{t('header.reference')}</Badge>}
              </button>
            </li>
          ))}
        </ol>
      </div>
    )
  }

  const columns: Column<ScanRow>[] = [
    { id: 'date', header: t('scans.date'), mono: true, sortValue: r => r.date, cell: r => r.kind === 'fused' ? <Row><Layers aria-hidden className="stac-scans__icon" />{t('header.fused')}</Row> : r.date },
    { id: 'label', header: t('scans.label'), sortValue: r => r.label, cell: r => r.label },
    { id: 'points', header: t('scans.points'), align: 'right', mono: true, sortValue: r => r.points ?? -1, cell: r => r.points ? fmt.millions(r.points) : r.recon_state ? t(`reconState.${r.recon_state}`) : t('status.dash') },
    { id: 'ref', header: t('scans.reference'), align: 'center', width: 'var(--sp-12)', cell: r => r.is_reference ? <Star className="stac-sessions__star" aria-hidden /> : r.kind !== 'fused' ? <IconButton size="sm" label={t('scans.setReference')} icon={<Star aria-hidden />} onClick={() => onSetReference(r)} /> : null },
  ]

  return (
    <Panel title={t('scans.title')} subtitle={sessionId ?? t('scans.noSession')}
      isEmpty={!sessionId || scans.length === 0}
      empty={<EmptyState icon={<Layers aria-hidden />} title={sessionId ? t('scans.emptyTitle') : t('scans.noSession')} description={sessionId ? t('scans.emptyDesc') : t('welcome.pickSession')} />}
      footer={<Button block icon={<Layers aria-hidden />} disabled={!canFuse} onClick={onFuse} title={canFuse ? t('instances.fuseHint') : t('instances.fuseNeedsTwo')}>{t('instances.fuse')}</Button>}>
      <Table columns={columns} rows={scans} rowKey={r => r.key} selectedKey={activeScanKey} onRowClick={onOpenScan} caption={t('scans.title')} initialSort={{ id: 'date', dir: 'asc' }} />
    </Panel>
  )
}
