/**
 * PipelineDialog — "Reconstruct": pick the scans to rebuild, see partial
 * (resumable) reconstructions, decide whether existing outputs are
 * replaced. Same endpoint and payload as before (run_pipeline over the WS).
 */
import { Play } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Checkbox } from '../components/ui/Field'
import { Banner } from '../components/ui/Banner'
import { Badge } from '../components/ui/Badge'
import { Stack, Row } from '../components/ui/Panel'
import { useFmt, useT } from '../i18n'

export interface PipelineScan { date: string; source: string; key: string; frame_count: number; has_output: boolean; recon_state?: string; cached_chunks?: number }

interface PipelineDialogProps {
  open: boolean
  sessionId: string | null
  scans: PipelineScan[]
  selected: string[]
  onToggleScan: (key: string, on: boolean) => void
  replace: boolean
  onReplace: (v: boolean) => void
  rebuildingKeys: string[]
  onCancel: () => void
  onRun: () => void
}

export function PipelineDialog({ open, sessionId, scans, selected, onToggleScan, replace, onReplace, rebuildingKeys, onCancel, onRun }: PipelineDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const partial = scans.filter(s => selected.includes(s.key) && s.recon_state === 'partial')
  const cached = partial.reduce((n, s) => n + (s.cached_chunks || 0), 0)
  return (
    <Dialog open={open} title={t('pipeline.title')} subtitle={sessionId ?? undefined} onClose={onCancel}
      footer={
        <>
          <Button onClick={onCancel}>{t('common.cancel')}</Button>
          <Button variant="primary" icon={<Play aria-hidden />} disabled={selected.length === 0} onClick={onRun} data-autofocus>{t('pipeline.run')}</Button>
        </>
      }>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t('pipeline.description')}</p>
        {scans.length > 0 && (
          <Stack gap={1}>
            <div className="stac-section__title">{scans.length === 1 ? t('pipeline.scanToRebuild') : t('pipeline.selectScans', { n: selected.length, total: scans.length })}</div>
            {scans.map(scan => {
              const rebuilding = rebuildingKeys.includes(scan.key)
              return (
                <div key={scan.key} className={`stac-pipeline__scan ${rebuilding ? 'stac-pipeline__scan--busy' : ''}`.trim()}>
                  <Checkbox checked={selected.includes(scan.key)} disabled={scans.length === 1 || rebuilding} onChange={v => onToggleScan(scan.key, v)}
                    label={<Row><span className="stac-mono stac-pipeline__date">{scan.date}</span>{scan.source !== 'default' && <Badge size="sm">{scan.source}</Badge>}</Row>} />
                  <span className="stac-pipeline__frames stac-mono">{t.plural('sessions.frames', scan.frame_count)}</span>
                  {scan.has_output && <Badge tone="ok" size="sm">{t('pipeline.hasOutput')}</Badge>}
                  {rebuilding && <Badge tone="brand" size="sm" dot>{t('sessions.rebuilding')}</Badge>}
                </div>
              )
            })}
          </Stack>
        )}
        {partial.length > 0 && (
          <Banner tone="ok" compact title={t('pipeline.partialTitle', { n: fmt.integer(cached) })}>{t('pipeline.partialDesc')}</Banner>
        )}
        <Checkbox checked={replace} onChange={onReplace} label={t('pipeline.replaceOutputs')} description={t('pipeline.replaceHint')} />
        {replace && partial.length > 0 && <Banner tone="warn" compact>{t('pipeline.replaceWarning')}</Banner>}
      </Stack>
    </Dialog>
  )
}
