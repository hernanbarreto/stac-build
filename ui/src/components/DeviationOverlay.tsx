/**
 * STAC Build — Deviation analysis (BIM vs scan): tolerance, comparison run,
 * aggregate readouts, histogram, legend and per-element pass rates. Lives in
 * the bottom dock's Report tab. Histogram bars use the deviation ramp tokens.
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { Play } from 'lucide-react'
import type { ViewportHandle } from './Viewport'
import { Button } from './ui/Button'
import { Slider } from './ui/Field'
import { Readout } from './ui/Readout'
import { Legend } from './ui/Legend'
import { Badge } from './ui/Badge'
import { Banner } from './ui/Banner'
import { Progress } from './ui/Progress'
import { Table, type Column } from './ui/Table'
import { Row, Stack } from './ui/Panel'
import { EmptyState } from './ui/EmptyState'
import { DEV_RAMP, deviationColor, tokenColor } from './viewport/palette'
import { useFmt, useT } from '../i18n'

export interface DeviationMatch { segment_label: string; element_key: string; ifc_type?: string; ifc_name?: string }

export interface DeviationElementResult {
  element_key: string
  label: string
  status?: string
  ifc_type?: string
  distances_mm?: number[]
  point_indices?: number[]
  total_points?: number
  sabana_positions?: number[]
  sabana_colors?: number[]
  sabana_n_points?: number
  n_faces?: number
  stats?: { min_mm: number; max_mm: number; mean_mm: number; std_mm: number; median_mm: number; p95_mm: number; within_tolerance: number; total_points: number; pass_rate: number; tolerance_mm: number }
  histogram?: { counts: number[]; bin_edges_mm: number[] }
  error?: string
}

export interface DeviationResult { ok: boolean; date?: string; tolerance_mm: number; transform?: number[][]; results: DeviationElementResult[] }

interface Props {
  sessionId: string
  viewportRef?: React.RefObject<ViewportHandle | null>
  onClose: () => void
  onHeatmapData?: (data: DeviationResult | null) => void
}

export default function DeviationOverlay({ sessionId, viewportRef, onHeatmapData }: Props) {
  const t = useT()
  const fmt = useFmt()
  const [matches, setMatches] = useState<DeviationMatch[]>([])
  const [result, setResult] = useState<DeviationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [tolerance, setTolerance] = useState(15) // mm
  const [error, setError] = useState<string | null>(null)
  const histCanvasRef = useRef<HTMLCanvasElement>(null)

  // Auto-match on mount
  useEffect(() => {
    fetch(`/api/bim/auto_match/${sessionId}`)
      .then(r => r.json())
      .then(data => { if (data.matches) setMatches(data.matches); if (data.error) setError(data.error) })
      .catch(() => setError(t('deviation.autoMatchFailed')))
  }, [sessionId, t])

  const runComparison = useCallback(async () => {
    if (!matches.length) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/bim/compare', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, matches: matches.map(m => ({ segment_label: m.segment_label, element_key: m.element_key, ifc_type: m.ifc_type })), tolerance_mm: tolerance }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data: DeviationResult = await res.json()
      setResult(data)
      onHeatmapData?.(data)
      if (viewportRef?.current && data.results) {
        if (data.transform) viewportRef.current.applyRegistrationTransform(data.transform)
        const sabanaData: Record<string, { positions: number[], colors: number[] }> = {}
        const unmatchedKeys: string[] = []
        for (const r of data.results) {
          if (r.sabana_positions && r.sabana_positions.length > 0) sabanaData[r.element_key] = { positions: r.sabana_positions, colors: r.sabana_colors || [] }
          if (r.status === 'unmatched') unmatchedKeys.push(r.element_key)
        }
        viewportRef.current.applyDeviationSurface(sabanaData, unmatchedKeys)
      }
    } catch (e: any) {
      setError(e.message || t('deviation.failed', { detail: '' }))
    } finally {
      setLoading(false)
    }
  }, [sessionId, matches, tolerance, onHeatmapData, viewportRef, t])

  // Histogram: bars coloured by the deviation ramp, tolerance line in cyan
  useEffect(() => {
    if (!result || !histCanvasRef.current) return
    const canvas = histCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const allResults = result.results.filter(r => r.histogram)
    if (!allResults.length) return
    const hist = allResults[0].histogram!
    const maxCount = Math.max(...hist.counts, 1)
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    const w = canvas.offsetWidth, h = canvas.offsetHeight
    ctx.clearRect(0, 0, w, h)
    const barW = w / hist.counts.length
    for (let i = 0; i < hist.counts.length; i++) {
      const barH = (hist.counts[i] / maxCount) * (h - 4)
      ctx.fillStyle = deviationColor(i / Math.max(1, hist.counts.length - 1))
      ctx.fillRect(i * barW, h - barH, barW - 1, barH)
    }
    if (allResults[0].stats) {
      const maxMm = hist.bin_edges_mm[hist.bin_edges_mm.length - 1]
      const tolX = (tolerance / maxMm) * w
      if (tolX > 0 && tolX < w) {
        ctx.strokeStyle = tokenColor('--measure')
        ctx.lineWidth = 1.5
        ctx.setLineDash([4, 3])
        ctx.beginPath(); ctx.moveTo(tolX, 0); ctx.lineTo(tolX, h); ctx.stroke()
        ctx.setLineDash([])
      }
    }
  }, [result, tolerance])

  const aggStats = result?.results.reduce((acc, r) => {
    if (!r.stats) return acc
    return { totalPoints: acc.totalPoints + r.stats.total_points, withinTol: acc.withinTol + r.stats.within_tolerance, maxDev: Math.max(acc.maxDev, r.stats.max_mm), meanDevSum: acc.meanDevSum + r.stats.mean_mm * r.stats.total_points }
  }, { totalPoints: 0, withinTol: 0, maxDev: 0, meanDevSum: 0 })
  const passRate = aggStats && aggStats.totalPoints > 0 ? aggStats.withinTol / aggStats.totalPoints : 0
  const meanDev = aggStats && aggStats.totalPoints > 0 ? aggStats.meanDevSum / aggStats.totalPoints : 0
  const passTone = passRate >= 0.9 ? 'ok' : passRate >= 0.7 ? 'warn' : 'err'
  const maxMm = result?.results[0]?.histogram ? result.results[0].histogram.bin_edges_mm.slice(-1)[0] : tolerance * 3

  const columns: Column<DeviationElementResult>[] = [
    { id: 'label', header: t('report.element'), sortValue: r => r.label, cell: r => r.label },
    { id: 'pass', header: t('deviation.passRate'), align: 'right', mono: true, sortValue: r => r.stats?.pass_rate ?? -1,
      cell: r => r.error ? <Badge tone="err" size="sm">{t('deviation.error')}</Badge> : r.stats ? <Badge size="sm" tone={r.stats.pass_rate >= 90 ? 'ok' : r.stats.pass_rate >= 70 ? 'warn' : 'err'}>{fmt.percent(r.stats.pass_rate / 100)}</Badge> : t('status.dash') },
    { id: 'mean', header: t('deviation.mean'), align: 'right', mono: true, sortValue: r => r.stats?.mean_mm ?? -1, cell: r => r.stats ? fmt.mm(r.stats.mean_mm) : t('status.dash') },
    { id: 'p95', header: t('deviation.p95'), align: 'right', mono: true, sortValue: r => r.stats?.p95_mm ?? -1, cell: r => r.stats ? fmt.mm(r.stats.p95_mm) : t('status.dash') },
    { id: 'max', header: t('deviation.max'), align: 'right', mono: true, sortValue: r => r.stats?.max_mm ?? -1, cell: r => r.stats ? fmt.mm(r.stats.max_mm) : t('status.dash') },
  ]

  return (
    <div className="stac-devrep">
      <div className="stac-devrep__side">
        <Stack gap={3}>
          <Slider tone="measure" min={1} max={50} step={1} value={tolerance} onChange={setTolerance} label={t('deviation.tolerance')} format={v => fmt.mm(v, 0)} />
          <Button variant="primary" icon={<Play aria-hidden />} loading={loading} disabled={matches.length === 0} onClick={runComparison}>{t('deviation.runComparison')}</Button>
          {!result && <p className="stac-section__hint">{matches.length > 0 ? t.plural('deviation.matched', matches.length) : t('deviation.noMatchesHint')}</p>}
          {error && <Banner tone="err" compact>{error}</Banner>}
          {result && aggStats && aggStats.totalPoints > 0 && (
            <>
              <Row wrap>
                <Readout label={t('deviation.mean')} value={meanDev} digits={1} unit="mm" provenance="tool_measured" />
                <Readout label={t('deviation.max')} value={aggStats.maxDev} digits={1} unit="mm" tone="warn" />
                <Readout label={t('deviation.passRate')} value={fmt.number(passRate * 100, 0)} unit="%" tone={passTone} />
              </Row>
              <Progress value={passRate * 100} tone={passTone} label={t('deviation.withinTolerance', { within: fmt.integer(aggStats.withinTol), total: fmt.integer(aggStats.totalPoints), tol: fmt.mm(tolerance, 0) })} />
            </>
          )}
        </Stack>
      </div>
      <div className="stac-devrep__main">
        {!result && <EmptyState compact title={t('deviation.emptyTitle')} description={t('deviation.emptyDesc')} />}
        {result && aggStats && aggStats.totalPoints > 0 && (
          <>
            <div className="stac-devrep__hist">
              <div className="stac-section__title">{t('deviation.histogram')}</div>
              <canvas ref={histCanvasRef} className="stac-devrep__canvas" />
              <Legend orientation="horizontal" ramp={DEV_RAMP} min={fmt.number(0, 0)} max={fmt.number(maxMm, 0)} unit="mm" />
            </div>
            <Table columns={columns} rows={result.results} rowKey={r => r.element_key} caption={t('report.elements', { n: result.results.length })} initialSort={{ id: 'pass', dir: 'asc' }} />
          </>
        )}
      </div>
    </div>
  )
}
