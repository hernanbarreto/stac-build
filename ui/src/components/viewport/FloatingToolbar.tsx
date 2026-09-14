/**
 * FloatingToolbar — glass toolbar centred at the top of the viewport (§7).
 * Groups: navigate (orbit, reset view) · select (brush: sphere / cube / box)
 * · measure (distance, angle, clear) · edit (align cloud, add object,
 * section box, reset section) · view (display settings, colour by, camera
 * poses, grid, axes, fullscreen) · BIM comparison when a model and segments
 * exist. One active mode at a time, highlighted with --brand-orange-soft.
 *
 * Sub-panels (brush, display settings, placed-object alignment) open under
 * the toolbar as glass popovers. Every action here already existed in the
 * previous toolbar / Tools menu; nothing new is invented.
 */
import { useState, type ReactNode } from 'react'
import {
  Axis3D, Box, Brush, Camera, Check, Circle, Eraser, Grid3X3, Home, Maximize, Move, Package, Palette,
  RotateCcw, Ruler, Scale, Scissors, SlidersHorizontal, Square, Thermometer, Trash2, TriangleRight, Undo2, Unlock, X, ArrowDownToLine, AlignVerticalJustifyEnd, ArrowUpToLine, AlignHorizontalJustifyCenter, AlignVerticalJustifyCenter, Loader2,
} from 'lucide-react'
import { IconButton } from '../ui/IconButton'
import { Button } from '../ui/Button'
import { Slider, Select, SegmentedControl } from '../ui/Field'
import { Badge } from '../ui/Badge'
import { Sep } from '../ui/Panel'
import { useFmt, useT } from '../../i18n'

export type Tool = 'navigate' | 'measure-distance' | 'measure-angle' | 'section-box' | 'align' | 'erase'
export type ColorMode = 'rgb' | 'status' | 'mv_votes' | 'deviation'
export type EraseShape = 'sphere' | 'cube' | 'box'

export interface BrushState {
  shape: EraseShape
  onShape: (s: EraseShape) => void
  radiusM: number
  onRadiusM: (r: number) => void
  yawDeg: number
  onYawDeg: (d: number) => void
  hasConfidence: boolean
  confThr: number
  confArmed: boolean
  onConfThr: (v: number) => void
  onApplyConf: () => void
  onCancelConf: () => void
  boxSelected: boolean
  onBoxMode: (m: 'translate' | 'rotate' | 'scale') => void
  onRemoveBox: () => void
  marks: number
  onErase: () => void
  targets: Array<{ id: number; label: string }>
  target: string
  onTarget: (v: string) => void
  onAssign: () => void
  onClearMarks: () => void
  onUndo: () => void
}

export interface DisplayState {
  pointSize: number
  onPointSize: (v: number) => void
  budgetM: number
  onBudgetM: (v: number) => void
  hasConfidence: boolean
  confidence: number
  onConfidence: (v: number) => void
}

export interface ViewState {
  hasCameraPoses: boolean
  showCameraPoses: boolean
  onToggleCameraPoses: () => void
  showGrid: boolean
  onToggleGrid: () => void
  showAxes: boolean
  onToggleAxes: () => void
  onFullscreen: () => void
}

export interface ColorState {
  mode: ColorMode
  onMode: (m: ColorMode) => void
  mvThreshold: number
  onMvThreshold: (v: number) => void
  hasWitness: boolean
  hasDeviation: boolean
}

export interface CompareState {
  available: boolean
  loading: boolean
  hasResult: boolean
  showing: boolean
  onRun: () => void
  onToggle: () => void
  chip?: string
}

export interface PlacedObjectState {
  name: string
  onMode: (m: 'translate' | 'rotate' | 'scale') => void
  onAlign: (op: 'floor' | 'same_base' | 'on_top' | 'center_xz' | 'center_y', target?: string) => void
  targets: Array<{ key: string; label: string }>
  target: string
  onTarget: (v: string) => void
  onRemove: () => void
}

interface FloatingToolbarProps {
  activeTool: Tool
  onTool: (t: Tool) => void
  onResetCamera: () => void
  onClearMeasurements: () => void
  onAddObject: () => void
  onResetSection: () => void
  brush: BrushState
  display: DisplayState
  view: ViewState
  color: ColorState
  compare: CompareState
  placed: PlacedObjectState | null
  sabanaVisible: boolean
}

type Popover = 'brush' | 'display' | 'color' | null

export function FloatingToolbar({ activeTool, onTool, onResetCamera, onClearMeasurements, onAddObject, onResetSection, brush, display, view, color, compare, placed, sabanaVisible }: FloatingToolbarProps) {
  const t = useT()
  const fmt = useFmt()
  const [popover, setPopover] = useState<Popover>(null)
  const toggle = (p: Popover) => setPopover(cur => (cur === p ? null : p))
  const brushOn = activeTool === 'erase'

  return (
    <div className="stac-vptoolbar-wrap">
      <div className="stac-vptoolbar" role="toolbar" aria-label={t('toolbar.label')}>
        {!sabanaVisible && (
          <>
            <Group>
              <IconButton size="lg" label={t('toolbar.navigate')} shortcut="V" icon={<RotateCcw aria-hidden />} active={activeTool === 'navigate'} onClick={() => onTool('navigate')} />
              <IconButton size="lg" label={t('toolbar.resetView')} shortcut="Home" icon={<Home aria-hidden />} onClick={onResetCamera} />
            </Group>
            <Sep />
            <Group>
              <IconButton size="lg" label={t('toolbar.brush')} icon={<Brush aria-hidden />} active={brushOn}
                onClick={() => { onTool(brushOn ? 'navigate' : 'erase'); setPopover(brushOn ? null : 'brush') }} />
              {brushOn && <IconButton size="lg" label={t('toolbar.brushOptions')} icon={<SlidersHorizontal aria-hidden />} active={popover === 'brush'} onClick={() => toggle('brush')} badge={brush.marks || undefined} />}
            </Group>
            <Sep />
            <Group>
              <IconButton size="lg" tone="measure" label={t('toolbar.measureDistance')} shortcut="M" icon={<Ruler aria-hidden />} active={activeTool === 'measure-distance'} onClick={() => onTool('measure-distance')} />
              <IconButton size="lg" tone="measure" label={t('toolbar.measureAngle')} shortcut="A" icon={<TriangleRight aria-hidden />} active={activeTool === 'measure-angle'} onClick={() => onTool('measure-angle')} />
              <IconButton size="lg" label={t('toolbar.clearMeasurements')} icon={<Trash2 aria-hidden />} onClick={onClearMeasurements} />
            </Group>
            <Sep />
            <Group>
              <IconButton size="lg" label={t('toolbar.alignCloud')} shortcut="G" icon={<Move aria-hidden />} active={activeTool === 'align'} onClick={() => onTool(activeTool === 'align' ? 'navigate' : 'align')} />
              <IconButton size="lg" label={t('toolbar.addObject')} icon={<Package aria-hidden />} onClick={onAddObject} />
              <IconButton size="lg" label={t('toolbar.sectionBox')} shortcut="X" icon={<Scissors aria-hidden />} active={activeTool === 'section-box'} onClick={() => onTool('section-box')} />
              <IconButton size="lg" label={t('toolbar.resetSection')} icon={<Unlock aria-hidden />} onClick={() => { onResetSection(); onTool('navigate') }} />
            </Group>
            <Sep />
            <Group>
              <IconButton size="lg" label={t('toolbar.display')} icon={<SlidersHorizontal aria-hidden />} active={popover === 'display'} onClick={() => toggle('display')} />
              <IconButton size="lg" label={t('toolbar.colorBy')} icon={<Palette aria-hidden />} active={popover === 'color' || color.mode !== 'rgb'} onClick={() => toggle('color')} />
            </Group>
            <Sep />
          </>
        )}
        <Group>
          {!sabanaVisible && view.hasCameraPoses && (
            <IconButton size="lg" label={view.showCameraPoses ? t('toolbar.hideCameraPoses') : t('toolbar.showCameraPoses')} icon={<Camera aria-hidden />} active={view.showCameraPoses} onClick={view.onToggleCameraPoses} />
          )}
          <IconButton size="lg" label={view.showGrid ? t('toolbar.hideGrid') : t('toolbar.showGrid')} icon={<Grid3X3 aria-hidden />} active={view.showGrid} onClick={view.onToggleGrid} />
          <IconButton size="lg" label={view.showAxes ? t('toolbar.hideAxes') : t('toolbar.showAxes')} icon={<Axis3D aria-hidden />} active={view.showAxes} onClick={view.onToggleAxes} />
          <IconButton size="lg" label={t('toolbar.fullscreen')} shortcut="F11" icon={<Maximize aria-hidden />} onClick={view.onFullscreen} />
        </Group>
        {compare.available && (
          <>
            <Sep />
            <Group>
              {!compare.showing && (
                <IconButton size="lg" tone="measure" label={t('toolbar.runComparison')} icon={compare.loading ? <Loader2 className="stac-spin" aria-hidden /> : <Scale aria-hidden />} disabled={compare.loading} onClick={compare.onRun} />
              )}
              {!compare.loading && compare.hasResult && (
                <IconButton size="lg" tone="measure" label={compare.showing ? t('toolbar.hideDeviation') : t('toolbar.showDeviation')} icon={<Thermometer aria-hidden />} active={compare.showing} onClick={compare.onToggle} />
              )}
              {compare.showing && compare.chip && <Badge tone="measure" mono>{compare.chip}</Badge>}
            </Group>
          </>
        )}
      </div>

      {popover === 'brush' && brushOn && (
        <div className="stac-vppopover stac-vppopover--brush" role="group" aria-label={t('toolbar.brushOptions')}>
          <div className="stac-vppopover__row">
            <SegmentedControl<EraseShape> size="sm" ariaLabel={t('brush.shape')} value={brush.shape} onChange={brush.onShape} options={[
              { value: 'sphere', icon: <Circle aria-hidden />, title: t('brush.sphere') },
              { value: 'cube', icon: <Square aria-hidden />, title: t('brush.cube') },
              { value: 'box', icon: <Box aria-hidden />, title: t('brush.box') },
            ]} />
            <Slider tone="err" min={3} max={150} step={1} value={Math.round(brush.radiusM * 100)} onChange={v => brush.onRadiusM(v / 100)} unit="cm" ariaLabel={t('brush.size')} />
          </div>
          {brush.shape === 'cube' && (
            <div className="stac-vppopover__row">
              <Slider min={0} max={90} step={1} value={brush.yawDeg} onChange={brush.onYawDeg} label={t('brush.rotation')} format={v => fmt.degrees(v, 0)} />
            </div>
          )}
          {brush.hasConfidence && (
            <div className="stac-vppopover__row">
              <Slider tone="err" min={0} max={100} step={1} value={Math.round(brush.confThr * 100)} onChange={v => brush.onConfThr(v / 100)} label={t('brush.confidence')} format={v => fmt.percent(v / 100)} />
              <Button size="sm" variant={brush.confArmed ? 'danger' : 'secondary'} disabled={!brush.confArmed} onClick={brush.onApplyConf} title={t('brush.applyConfidenceHint')}>{t('common.apply')}</Button>
              {brush.confArmed && <IconButton size="sm" label={t('brush.cancelPreview')} icon={<X aria-hidden />} onClick={brush.onCancelConf} />}
            </div>
          )}
          {brush.boxSelected && (
            <div className="stac-vppopover__row">
              <Button size="sm" onClick={() => brush.onBoxMode('translate')}>{t('gizmo.move')}</Button>
              <Button size="sm" onClick={() => brush.onBoxMode('rotate')}>{t('gizmo.rotate')}</Button>
              <Button size="sm" onClick={() => brush.onBoxMode('scale')}>{t('gizmo.stretch')}</Button>
              <Button size="sm" variant="danger" onClick={brush.onRemoveBox}>{t('gizmo.remove')}</Button>
            </div>
          )}
          <Button block variant={brush.marks > 0 ? 'danger' : 'secondary'} disabled={brush.marks === 0} icon={<Eraser aria-hidden />} onClick={brush.onErase} title={t('brush.eraseHint')}>
            {t('brush.erase', { n: brush.marks })}
          </Button>
          <div className="stac-vppopover__row">
            <Select<string> size="sm" value={brush.target} onChange={brush.onTarget} placeholder={t('brush.reassignTo')}
              options={[...brush.targets.map(s => ({ value: String(s.id), label: `${s.label} ${s.id}` })), { value: 'new', label: t('brush.newSegment') }]} />
            <Button size="sm" icon={<Check aria-hidden />} disabled={brush.marks === 0 || brush.target === ''} onClick={brush.onAssign} title={t('brush.assignHint')}>{t('brush.assign')}</Button>
          </div>
          <div className="stac-vppopover__row">
            <Button size="sm" block icon={<X aria-hidden />} disabled={brush.marks === 0} onClick={brush.onClearMarks}>{t('brush.clearMarks')}</Button>
            <Button size="sm" block icon={<Undo2 aria-hidden />} onClick={brush.onUndo}>{t('brush.undo')}</Button>
          </div>
        </div>
      )}

      {popover === 'display' && (
        <div className="stac-vppopover" role="group" aria-label={t('toolbar.display')}>
          <Slider min={0.01} max={5} step={0.01} value={display.pointSize} onChange={display.onPointSize} label={t('display.pointSize')} digits={2} />
          <Slider min={2} max={40} step={1} value={display.budgetM} onChange={display.onBudgetM} label={t('display.detail')} format={v => t('display.millionPoints', { n: fmt.integer(v) })} />
          {display.hasConfidence && (
            <Slider min={0} max={1} step={0.01} value={display.confidence} onChange={display.onConfidence} label={t('display.confidence')} format={v => fmt.percent(v)} />
          )}
        </div>
      )}

      {popover === 'color' && (
        <div className="stac-vppopover" role="group" aria-label={t('toolbar.colorBy')}>
          <SegmentedControl<ColorMode> size="sm" ariaLabel={t('toolbar.colorBy')} value={color.mode} onChange={color.onMode} options={[
            { value: 'rgb', label: t('colorMode.rgb') },
            { value: 'status', label: t('colorMode.status'), disabled: !color.hasWitness },
            { value: 'mv_votes', label: t('colorMode.votes'), disabled: !color.hasWitness },
            { value: 'deviation', label: t('colorMode.deviation'), disabled: !color.hasDeviation },
          ]} />
          {color.mode === 'mv_votes' && (
            <Slider min={0} max={8} step={1} value={color.mvThreshold} onChange={color.onMvThreshold} label={t('colorMode.threshold')} format={v => t('colorMode.views', { n: fmt.integer(v) })} />
          )}
          {!color.hasWitness && <p className="stac-vppopover__hint">{t('colorMode.noWitness')}</p>}
        </div>
      )}

      {placed && (
        <div className="stac-vppopover stac-vppopover--placed" role="toolbar" aria-label={t('placed.title')}>
          <span className="stac-vppopover__title">{placed.name}</span>
          <Button size="sm" onClick={() => placed.onMode('translate')}>{t('gizmo.move')}</Button>
          <Button size="sm" onClick={() => placed.onMode('rotate')}>{t('gizmo.rotate')}</Button>
          <Button size="sm" onClick={() => placed.onMode('scale')}>{t('gizmo.scale')}</Button>
          <Sep />
          <Button size="sm" icon={<ArrowDownToLine aria-hidden />} onClick={() => placed.onAlign('floor')} title={t('placed.floorHint')}>{t('placed.floor')}</Button>
          <Select<string> size="sm" value={placed.target} onChange={placed.onTarget} placeholder={t('placed.alignTo')} options={placed.targets.map(x => ({ value: x.key, label: x.label }))} className="stac-vppopover__select" />
          <IconButton size="sm" label={t('placed.sameBase')} icon={<AlignVerticalJustifyEnd aria-hidden />} disabled={!placed.target} onClick={() => placed.onAlign('same_base', placed.target)} />
          <IconButton size="sm" label={t('placed.onTop')} icon={<ArrowUpToLine aria-hidden />} disabled={!placed.target} onClick={() => placed.onAlign('on_top', placed.target)} />
          <IconButton size="sm" label={t('placed.centerHorizontal')} icon={<AlignHorizontalJustifyCenter aria-hidden />} disabled={!placed.target} onClick={() => placed.onAlign('center_xz', placed.target)} />
          <IconButton size="sm" label={t('placed.centerVertical')} icon={<AlignVerticalJustifyCenter aria-hidden />} disabled={!placed.target} onClick={() => placed.onAlign('center_y', placed.target)} />
          <Sep />
          <Button size="sm" variant="danger" icon={<Trash2 aria-hidden />} onClick={placed.onRemove} title={t('placed.removeHint')}>{t('placed.remove')}</Button>
        </div>
      )}
    </div>
  )
}

function Group({ children }: { children: ReactNode }) {
  return <div className="stac-vptoolbar__group">{children}</div>
}
