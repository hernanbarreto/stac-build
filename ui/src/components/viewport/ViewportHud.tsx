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
import { ArrowDown, Pin } from 'lucide-react'
import type * as THREE from 'three'
import { Legend } from '../ui/Legend'
import { Tooltip } from '../ui/Tooltip'
import { ViewCube } from './ViewCube'
import { useFmt, useT } from '../../i18n'
import { STATUS_TOKENS, DEV_RAMP } from './palette'
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
  /** a short label in front of the value (H / W / D of a selected segment) */
  tag?: string
  metres?: number
  degrees?: number
  visible: boolean
  /** a distance the user measured on the model: it can be marked as a calibration measurement */
  calibratable?: boolean
  /** a stored calibration measurement of the scan (its role) */
  cal?: 'gauge' | 'test' | 'plane'
}

interface ViewportHudProps {
  onView: (preset: ViewPreset) => void
  /** the ViewCube's hooks into the main camera (USER 2026-10-04) */
  getQuaternion: () => THREE.Quaternion | null
  onOrient: (dir: [number, number, number]) => void
  onOrbit: (dAz: number, dEl: number) => void
  onHome: () => void
  metersPerPixel: number | null
  /** What the grid's cells and major lines measure (USER 2026-11-09); null when the grid is hidden. */
  gridLabel?: string | null
  colorMode: ColorMode
  mvThreshold: number
  statusFractions?: Record<string, number> | null
  deviationRange?: { maxMm: number; toleranceMm: number } | null
  showEdges?: boolean
  readouts: ReadoutAnchor[]
  /** the pin of a measured distance was pressed: mark it as a calibration measurement
   *  (USER 2026-11-13: "se debe poder seleccionar que una medicion es de calibracion") */
  onCalibrate?: (readoutId: string) => void
  /** a stored calibration measurement's readout was pressed (open it in the calibration window) */
  onOpenCalibration?: (calibrationId: string) => void
  /** THE CLOUD'S CALIBRATION, shown from the moment it opens (USER 2026-11-13: "debe indicarse si la nube esta
   *  calibrada o no, eso lo debe indicar al abrir la nube"): null while it is not known */
  calibration?: { state: 'uncalibrated' | 'calibrated' | 'verified'; factor: number | null; worstDifferenceM: number | null
                  /** the per-axis factors when the calibration in use is per axis */
                  axes?: { x: number; y: number; z: number } | null } | null
  banner?: ReactNode
}

const SCALE_STEPS_M = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

export function ViewportHud({ onView, getQuaternion, onOrient, onOrbit, onHome, metersPerPixel, gridLabel, colorMode, mvThreshold, statusFractions, deviationRange, readouts, onCalibrate, onOpenCalibration, calibration, banner }: ViewportHudProps) {
  const t = useT()
  const fmt = useFmt()

  let scale: { px: number; label: string } | null = null
  if (metersPerPixel && metersPerPixel > 0) {
    const maxM = metersPerPixel * 120
    const step = [...SCALE_STEPS_M].reverse().find(s => s <= maxM) ?? SCALE_STEPS_M[0]
    scale = { px: step / metersPerPixel, label: fmt.lengthText(step) }
  }

  void onView   // the preset buttons gave way to the ViewCube; the handle stays for shortcuts

  return (
    <>
      {banner && <div className="stac-hud__banner">{banner}</div>}

      <div className="stac-hud__cube">
        <ViewCube getQuaternion={getQuaternion} onOrient={onOrient} onOrbit={onOrbit} onHome={onHome} />
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
        {gridLabel && <span className="stac-hud__grid stac-mono">{gridLabel}</span>}
        {calibration && (
          <button type="button" className={`stac-hud__calstate stac-hud__calstate--${calibration.state}`} title={t('calibration.open')}
            onClick={() => onOpenCalibration?.('')}>
            <Pin aria-hidden />
            <span>{t(`calibration.state_${calibration.state}`)}</span>
            {calibration.axes
              ? <span className="stac-mono">{`X × ${fmt.number(calibration.axes.x, 4)} Y × ${fmt.number(calibration.axes.y, 4)} Z × ${fmt.number(calibration.axes.z, 4)}`}</span>
              : calibration.factor != null && <span className="stac-mono">{`× ${fmt.number(calibration.factor, 4)}`}</span>}
            {calibration.state === 'verified' && calibration.worstDifferenceM != null && (
              <span className="stac-mono">{`${calibration.worstDifferenceM >= 0 ? '+' : ''}${fmt.mm(calibration.worstDifferenceM * 1000, 1)}`}</span>
            )}
          </button>
        )}
      </div>

      {colorMode !== 'rgb' && (
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
        </div>
      )}

      <svg className="stac-hud__guides" aria-hidden>
        {readouts.filter(r => r.visible).map(r => <line key={r.id} x1={r.ax} y1={r.ay} x2={r.x} y2={r.y} className="stac-hud__guide" />)}
      </svg>
      {readouts.filter(r => r.visible).map(r => {
        const len = r.metres != null ? fmt.length(r.metres) : null
        return (
          <div key={r.id} className={`stac-hud__readout ${r.cal ? `stac-hud__readout--cal stac-hud__readout--cal-${r.cal}` : ''}`.trim()} style={{ '--rx': `${r.x}px`, '--ry': `${r.y}px` } as React.CSSProperties}>
            <span className="stac-readout stac-readout--compact stac-readout--measure">
              {r.tag && <span className="stac-readout__tag">{r.tag}</span>}
              {r.kind === 'distance' && len && <><span className="stac-readout__value">{len.value}</span><span className="stac-readout__unit">{len.unit}</span></>}
              {r.kind === 'angle' && r.degrees != null && <><span className="stac-readout__value">{fmt.number(r.degrees, 1)}</span><span className="stac-readout__unit">°</span></>}
              {r.calibratable && onCalibrate && (
                <button type="button" className="stac-hud__calibrate" title={t('calibration.markHint')} aria-label={t('calibration.markHint')}
                  onClick={e => { e.stopPropagation(); onCalibrate(r.id) }}><Pin aria-hidden /></button>
              )}
              {r.cal && onOpenCalibration && (
                <button type="button" className="stac-hud__calibrate" title={t('calibration.open')} aria-label={t('calibration.open')}
                  onClick={e => { e.stopPropagation(); onOpenCalibration(r.id.replace(/^cal-/, '')) }}><Pin aria-hidden /></button>
              )}
            </span>
          </div>
        )
      })}
    </>
  )
}
