/**
 * MeshingDialog — per-object meshing: pick instances, then Object
 * (generative ShapeR asset from the object's cloud + views + description —
 * USER 2026-10-01) or Mesh (RANSAC + Poisson from the object's own cloud).
 * Live phase per instance for BOTH jobs; existing meshes are overwritten.
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

export interface InstanceProgress { id: number; phase: string; elapsed?: number; error?: string; mesh?: string; reason?: string; textured?: boolean }
export interface OverallProgress { phase: string; total?: number; done?: number; error?: string }

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
  /** per-instance phases of the shape (ShapeR) job — /api/segmentation/shape/progress */
  shapeProgress: Record<number, InstanceProgress>
  shapeOverall: OverallProgress
  onObject: () => void
  onMesh: () => void
}

const PHASE_TONE: Record<string, BadgeTone> = { starting: 'info', integrating: 'brand', extracting: 'info', surface_fit: 'brand', done: 'ok', error: 'err' }
// The backend's per-instance phases (server/main.py `_shape_progress`): the
// export (exporting_ply → exporting_pkl → captioning → ply_ready | skipped), the
// batch (reconstructing → icp_ok | icp_skip → fit_applied | fit_noop | fit_error
// → done), the texture (texturing → done), error.
const SHAPE_PHASE_TONE: Record<string, BadgeTone> = {
  exporting_ply: 'info', exporting_pkl: 'info', captioning: 'info', ply_ready: 'info', skipped: 'neutral',
  reconstructing: 'brand', icp_ok: 'brand', icp_skip: 'warn', fit_applied: 'brand', fit_noop: 'brand', fit_error: 'warn',
  warn: 'warn', texturing: 'brand', done: 'ok', error: 'err',
}

export function MeshingDialog(p: MeshingDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const list = p.segments.filter(s => s.label !== 'Unsegmented')
  const busy = p.tsdfRunning || p.shapeRunning
  const overallTone: BadgeTone = p.tsdfOverall.phase === 'error' ? 'err' : p.tsdfOverall.phase === 'done' ? 'ok' : 'brand'
  const shapeTone: 'err' | 'ok' | 'info' = p.shapeOverall.phase === 'error' ? 'err' : p.shapeOverall.phase === 'done' ? 'ok' : 'info'
  const shapeCounts = { done: p.shapeOverall.done || 0, total: p.shapeOverall.total || 0 }
  return (
    <Dialog open={p.open} title={t('meshing.title')} onClose={p.onClose} icon={<Puzzle aria-hidden />} size="lg"
      footer={
        <>
          <Button variant="secondary" icon={<Box aria-hidden />} loading={p.shapeRunning} disabled={busy || p.selected.size === 0} onClick={p.onObject} title={t('meshing.objectHint')}>
            {p.shapeRunning ? t('meshing.objectRunning', shapeCounts) : t('meshing.object', { n: p.selected.size })}
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
            const sp = p.shapeProgress[seg.id]
            const shapePhase = sp?.phase && sp.phase !== 'pending' ? sp.phase : null
            const on = p.selected.has(seg.id)
            return (
              <li key={seg.key} className={`stac-picklist__item ${on ? 'stac-picklist__item--on' : ''}`.trim()}>
                <Checkbox checked={on} disabled={busy} onChange={() => p.onToggle(seg.id)} label={<Row><ColorDot color={seg.color} /><span>{seg.label}</span></Row>} />
                <span className="stac-mono stac-picklist__meta">{t('status.points', { n: fmt.integer(seg.totalPoints) })}</span>
                {phase && <Badge tone={PHASE_TONE[phase] ?? 'neutral'} size="sm">{t(`meshPhase.${phase}`)}</Badge>}
                {prog?.elapsed != null && (phase === 'done' || phase === 'integrating' || phase === 'extracting') && <span className="stac-picklist__meta stac-mono">{fmt.duration(prog.elapsed)}</span>}
                {prog?.error && <span className="stac-picklist__error">{prog.error.slice(0, 140)}</span>}
                {shapePhase && <Badge tone={SHAPE_PHASE_TONE[shapePhase] ?? 'neutral'} size="sm" title={t('meshing.objectHint')}>{t(`shapePhase.${shapePhase}`, shapeCounts)}</Badge>}
                {sp?.elapsed != null && (shapePhase === 'done' || shapePhase === 'texturing') && <span className="stac-picklist__meta stac-mono">{fmt.duration(sp.elapsed)}</span>}
                {sp?.error && <span className="stac-picklist__error">{sp.error.slice(0, 140)}</span>}
                {sp?.reason && shapePhase === 'skipped' && <span className="stac-picklist__warn">{sp.reason.slice(0, 140)}</span>}
                {on && !busy && st?.has_mesh && <span className="stac-picklist__warn">{t('meshing.overwriteWarning')}</span>}
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
        {p.shapeOverall.phase !== 'idle' && (p.shapeRunning || p.shapeOverall.phase === 'done' || p.shapeOverall.phase === 'error') && (
          <Banner tone={shapeTone} compact>
            {t(`shapePhase.${p.shapeOverall.phase}`, shapeCounts)}
            {p.shapeOverall.phase === 'error' && p.shapeOverall.error ? ` — ${p.shapeOverall.error.slice(0, 160)}` : ''}
          </Banner>
        )}
      </Stack>
    </Dialog>
  )
}
