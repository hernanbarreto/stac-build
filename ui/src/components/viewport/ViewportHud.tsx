/**
 * ViewportHud (§7): navigation cube (top-right), gravity indicator + scale
 * bar (bottom-left), vertical legend (right) when the colour mode needs
 * one, compact Readouts anchored to the measured points with a 1 px cyan
 * guide line, and the work banner (pipeline running / correction pending)
 * as a banner inside the viewport, never a modal.
 *
 * All positions arrive as CSS variables; the rules live in viewport.css.
 * The Viewport projects the measurement anchors to screen space each frame
 * (that is the only Three.js work involved); this component only draws.
 */
import type { ReactNode } from 'react'
import { ArrowDown } from 'lucide-react'
import { Legend } from '../ui/Legend'
import { Tooltip } from '../ui/Tooltip'
import { useFmt, useT } from '../../i18n'
import { EDGE_TOKENS, STATUS_TOKENS, DEV_RAMP } from './palette'
import type { ColorMode } from './FloatingToolbar'

export type ViewPreset = 'top' | 'front' | 'right' | 'left' | 'back' | 'bottom' | 'iso'

export interface ReadoutAnchor {
  id: string
  x: number
  y: number
  /** guide-line origin (the measured point) in screen px */
  ax: number
  ay: number
  kind: 'distance' | 'angle'
  metres?: number
  degrees?: number
  visible: boolean
}

interface ViewportHudProps {
  onView: (preset: ViewPreset) => void
  metersPerPixel: number | null
  colorMode: ColorMode
  mvThreshold: number
  statusFractions?: Record<string, number> | null
  deviationRange?: { maxMm: number; toleranceMm: number } | null
  showEdges?: boolean
  readouts: ReadoutAnchor[]
  banner?: ReactNode
}

const SCALE_STEPS_M = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

export function ViewportHud({ onView, metersPerPixel, colorMode, mvThreshold, statusFractions, deviationRange, showEdges, readouts, banner }: ViewportHudProps) {
  const t = useT()
  const fmt = useFmt()

  let scale: { px: number; label: string } | null = null
  if (metersPerPixel && metersPerPixel > 0) {
    const maxM = metersPerPixel * 120
    const step = [...SCALE_STEPS_M].reverse().find(s => s <= maxM) ?? SCALE_STEPS_M[0]
    scale = { px: step / metersPerPixel, label: fmt.lengthText(step) }
  }

  const cube: Array<{ id: ViewPreset; label: string }> = [
    { id: 'top', label: t('view.top') }, { id: 'front', label: t('view.front') }, { id: 'right', label: t('view.right') },
    { id: 'left', label: t('view.left') }, { id: 'back', label: t('view.back') }, { id: 'iso', label: t('view.iso') },
  ]

  return (
    <>
      {banner && <div className="stac-hud__banner">{banner}</div>}

      <div className="stac-hud__cube" role="group" aria-label={t('view.cube')}>
        {cube.map(v => (
          <Tooltip key={v.id} label={v.label} side="left">
            <button type="button" className={`stac-hud__face stac-hud__face--${v.id}`} onClick={() => onView(v.id)} aria-label={v.label}>
              {v.label.slice(0, 1)}
            </button>
          </Tooltip>
        ))}
      </div>

      <div className="stac-hud__bottomleft">
        <Tooltip label={t('view.gravity')} side="right">
          <div className="stac-hud__gravity" aria-label={t('view.gravity')} role="img">
            <ArrowDown aria-hidden />
            <span className="stac-mono">{t('view.upAxis')}</span>
          </div>
        </Tooltip>
        {scale && (
          <div className="stac-hud__scale" role="img" aria-label={t('view.scaleBar', { length: scale.label })}>
            <div className="stac-hud__scale-bar" style={{ '--scale-px': `${Math.round(scale.px)}px` } as React.CSSProperties} />
            <span className="stac-hud__scale-label stac-mono">{scale.label}</span>
          </div>
        )}
      </div>

      {(colorMode !== 'rgb' || showEdges) && (
        <div className="stac-hud__legend">
          {colorMode === 'status' && (
            <Legend glass title={t('colorMode.status')} categories={Object.entries(STATUS_TOKENS).map(([k, tok]) => ({
              color: tok, label: t(`witness.${k}`), value: statusFractions?.[k] != null ? fmt.percent(statusFractions[k], 1) : undefined,
            }))} />
          )}
          {colorMode === 'mv_votes' && (
            <Legend glass title={t('colorMode.votes')} categories={[
              { color: '--ok', label: t('colorMode.votesAtLeast', { n: fmt.integer(mvThreshold) }) },
              { color: '--err', label: t('colorMode.votesFewer', { n: fmt.integer(mvThreshold) }) },
            ]} />
          )}
          {colorMode === 'deviation' && (
            <Legend glass title={t('colorMode.deviation')} ramp={DEV_RAMP} min={fmt.number(0, 0)} max={deviationRange ? fmt.number(deviationRange.maxMm, 0) : undefined} unit="mm"
              categories={deviationRange ? [{ color: '--measure', label: t('deviation.tolerance'), value: fmt.mm(deviationRange.toleranceMm, 0) }] : undefined} />
          )}
          {showEdges && (
            <Legend glass title={t('edges.title')} categories={Object.entries(EDGE_TOKENS).map(([k, tok]) => ({ color: tok, label: t(`edgeKind.${k}`) }))} />
          )}
        </div>
      )}

      <svg className="stac-hud__guides" aria-hidden>
        {readouts.filter(r => r.visible).map(r => <line key={r.id} x1={r.ax} y1={r.ay} x2={r.x} y2={r.y} className="stac-hud__guide" />)}
      </svg>
      {readouts.filter(r => r.visible).map(r => {
        const len = r.metres != null ? fmt.length(r.metres) : null
        return (
          <div key={r.id} className="stac-hud__readout" style={{ '--rx': `${r.x}px`, '--ry': `${r.y}px` } as React.CSSProperties}>
            <span className="stac-readout stac-readout--compact stac-readout--measure">
              {r.kind === 'distance' && len && <><span className="stac-readout__value">{len.value}</span><span className="stac-readout__unit">{len.unit}</span></>}
              {r.kind === 'angle' && r.degrees != null && <><span className="stac-readout__value">{fmt.number(r.degrees, 1)}</span><span className="stac-readout__unit">°</span></>}
            </span>
          </div>
        )
      })}
    </>
  )
}
