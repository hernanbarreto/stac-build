/**
 * MeshingDialog — per-object meshing: pick instances, then Object
 * (generative MeshFlow asset) or Mesh (RANSAC + Poisson from the object's
 * own cloud). Live phase per instance; existing meshes are overwritten.
 */
import { Box, Puzzle } from 'lucide-react'
import type { SegmentInstance } from '../components/Viewport'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Checkbox } from '../components/ui/Field'
import { Badge, ColorDot, type BadgeTone } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { Row, Stack } from '../components/ui/Panel'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface InstanceProgress { id: number; phase: string; elapsed?: number; error?: string; mesh?: string }
export interface OverallProgress { phase: string; total?: number; done?: number }

interface MeshingDialogProps {
  open: boolean
  onClose: () => void
  segments: SegmentInstance[]
  selected: Set<number>
  onToggle: (id: number) => void
  tsdfStatus: Record<number, { has_mesh: boolean }>
  tsdfProgress: Record<number, InstanceProgress>
  tsdfOverall: OverallProgress
  tsdfRunning: boolean
  shapeRunning: boolean
  shapeOverall: OverallProgress
  onObject: () => void
  onMesh: () => void
}

const PHASE_TONE: Record<string, BadgeTone> = { starting: 'info', integrating: 'brand', extracting: 'info', surface_fit: 'brand', done: 'ok', error: 'err' }

export function MeshingDialog(p: MeshingDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const list = p.segments.filter(s => s.label !== 'Unsegmented')
  const busy = p.tsdfRunning || p.shapeRunning
  const overallTone: BadgeTone = p.tsdfOverall.phase === 'error' ? 'err' : p.tsdfOverall.phase === 'done' ? 'ok' : 'brand'
  return (
    <Dialog open={p.open} title={t('meshing.title')} onClose={p.onClose} icon={<Puzzle aria-hidden />} size="lg"
      footer={
        <>
          <Button variant="secondary" icon={<Box aria-hidden />} loading={p.shapeRunning} disabled={busy || p.selected.size === 0} onClick={p.onObject} title={t('meshing.objectHint')}>
            {p.shapeRunning ? t('meshing.objectRunning', { done: p.shapeOverall.done || 0, total: p.shapeOverall.total || 0 }) : t('meshing.object', { n: p.selected.size })}
          </Button>
          <Button variant="primary" icon={<Puzzle aria-hidden />} loading={p.tsdfRunning} disabled={busy || p.selected.size === 0} onClick={p.onMesh} title={t('meshing.meshHint')}>
            {p.tsdfRunning ? t('meshing.meshRunning', { done: p.tsdfOverall.done || 0, total: p.tsdfOverall.total || 0 }) : t('meshing.mesh', { n: p.selected.size })}
          </Button>
        </>
      }>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t('meshing.description')}</p>
        {list.length === 0 && <EmptyState compact icon={<Puzzle aria-hidden />} title={t('meshing.noSegments')} description={t('meshing.noSegmentsDesc')} />}
        <ul className="stac-picklist">
          {list.map(seg => {
            const st = p.tsdfStatus[seg.id]
            const prog = p.tsdfProgress[seg.id]
            const phase = prog?.phase && prog.phase !== 'pending' ? prog.phase : st?.has_mesh ? 'done' : null
            const on = p.selected.has(seg.id)
            return (
              <li key={seg.key} className={`stac-picklist__item ${on ? 'stac-picklist__item--on' : ''}`.trim()}>
                <Checkbox checked={on} disabled={p.tsdfRunning} onChange={() => p.onToggle(seg.id)} label={<Row><ColorDot color={seg.color} /><span>{seg.label}</span></Row>} />
                <span className="stac-mono stac-picklist__meta">{t('status.points', { n: fmt.integer(seg.totalPoints) })}</span>
                {phase && <Badge tone={PHASE_TONE[phase] ?? 'neutral'} size="sm">{t(`meshPhase.${phase}`)}</Badge>}
                {prog?.elapsed != null && (phase === 'done' || phase === 'integrating' || phase === 'extracting') && <span className="stac-picklist__meta stac-mono">{fmt.duration(prog.elapsed)}</span>}
                {prog?.error && <span className="stac-picklist__error">{prog.error.slice(0, 140)}</span>}
                {on && !p.tsdfRunning && st?.has_mesh && <span className="stac-picklist__warn">{t('meshing.overwriteWarning')}</span>}
              </li>
            )
          })}
        </ul>
        {p.tsdfOverall.phase !== 'idle' && (p.tsdfRunning || p.tsdfOverall.phase === 'done' || p.tsdfOverall.phase === 'error') && (
          <Banner tone={overallTone === 'brand' ? 'info' : overallTone} compact>
            {p.tsdfOverall.phase === 'done' ? t('meshing.allDone', { done: p.tsdfOverall.done || 0, total: p.tsdfOverall.total || 0 })
              : p.tsdfOverall.phase === 'error' ? t('meshing.pipelineError')
              : t('meshing.integrating', { done: p.tsdfOverall.done || 0, total: p.tsdfOverall.total || 0 })}
          </Banner>
        )}
      </Stack>
    </Dialog>
  )
}
