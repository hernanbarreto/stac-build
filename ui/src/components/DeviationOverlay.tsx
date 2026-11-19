/**
 * STAC Build — Deviation analysis (BIM vs scan): the Report tab of the bottom dock. The tolerance,
 * the launch and THE STORED COMPARISON's numbers — aggregate readouts, histogram, legend and
 * per-element pass rates. It owns no comparison of its own (USER 2026-11-19: "sigue viendose el
 * not comparison yet a pesar que ya esta la comparacion" / "deja la tolerancia por defecto en 50mm
 * pero toma la del slider"): the launch is the application's — the one the toolbar's button
 * starts —, the tolerance is the application's one value, and what is shown is the comparison
 * stored in the project. (Until then it launched its own, with a pairing that carried no object
 * ids, and waited for the numbers in the launch's answer.)
 */
import { useState, useEffect, useMemo, useRef } from 'react'
import { registerCommands, type Executors, type Outcome, type ReportCommandId } from '../commands/registry'
import { Play } from 'lucide-react'
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
  /** the scan's objects compared with this element (their instance ids) */
  segment_ids?: number[]
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

export interface DeviationResult {
  ok: boolean; date?: string; tolerance_mm: number; transform?: number[][]; results: DeviationElementResult[]
  /** every compared point together: its histogram over 0 … 2 × tolerance and how many lie beyond */
  overall?: { histogram?: { counts: number[]; bin_edges_mm: number[]; beyond?: number } }
}

/** The distribution of ALL the compared points over 0 … 2 × tolerance — the sheet's colour scale: green
 *  at 0, the tolerance in the middle, red at twice it — and how many lie beyond. The server's own when
 *  the stored report carries it; else every element's histogram spread over those bins. (Until
 *  2026-11-19 the chart was the FIRST element's histogram alone.) */
function wholeHistogram(result: DeviationResult, tolMm: number, totalPoints: number): { counts: number[]; beyond: number; maxMm: number } | null {
  const o = result.overall?.histogram
  if (o && o.counts.length) return { counts: o.counts, beyond: o.beyond ?? 0, maxMm: o.bin_edges_mm[o.bin_edges_mm.length - 1] }
  const hs = result.results.map(r => r.histogram).filter((h): h is { counts: number[]; bin_edges_mm: number[] } => !!h)
  if (!hs.length || !(tolMm > 0)) return null
  const n = 40, maxMm = 2 * tolMm, w = maxMm / n
  const counts = new Array<number>(n).fill(0)
  for (const h of hs) for (let i = 0; i < h.counts.length; i++) {
    const a = h.bin_edges_mm[i], b = h.bin_edges_mm[i + 1], c = h.counts[i]
    if (!c || !(b > a)) continue
    for (let k = Math.floor(a / w); k < n && k * w < b; k++) {
      const lo = Math.max(a, k * w), hi = Math.min(b, (k + 1) * w)
      if (hi > lo) counts[k] += c * (hi - lo) / (b - a)
    }
  }
  return { counts, beyond: Math.max(0, totalPoints - counts.reduce((x, y) => x + y, 0)), maxMm }
}

interface Props {
  sessionId: string
  /** the scan on screen (the one a launch compares) */
  scanKey: string | null
  /** the tolerance of the NEXT comparison, in mm: the slider's — the application's one value */
  tolerance: number
  onTolerance: (mm: number) => void
  /** launch the application's comparison (at this tolerance when given) */
  onRun: (toleranceMm?: number) => void
  running: boolean
  /** the comparison stored in the project, of the scan on screen — null: there is none */
  report: DeviationResult | null
  /** the object selected in the viewer (its instance id): its row is the selected one */
  selectedInstanceId?: number | null
  /** a row was touched: select its object in the viewer (USER 2026-11-19: "en la lista de elementos, cuando
   *  toco alguno debe mostrarmelo como seleccionado en la sabana") */
  onSelectInstance?: (instanceId: number) => void
  /** in a narrow panel (the BIM panel): one column, the element list with its three main columns */
  compact?: boolean
  onClose?: () => void
}

/** The scan's objects of a row: the ids the comparison recorded, else the element's own key when it
 *  is a number (an IFC that STAC wrote from the scan names each element by its object's id). */
function instancesOf(r: DeviationElementResult): number[] {
  if (r.segment_ids?.length) return r.segment_ids
  return /^\d+$/.test(r.element_key) ? [Number(r.element_key)] : []
}

export default function DeviationOverlay({ sessionId, scanKey, tolerance, onTolerance, onRun, running, report, selectedInstanceId, onSelectInstance, compact = false }: Props) {
  const t = useT()
  const fmt = useFmt()
  const [matched, setMatched] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const histCanvasRef = useRef<HTMLCanvasElement>(null)
  const result = report
  // the tolerance the STORED comparison was run with (the slider's is the next one's)
  const shownTolerance = result?.tolerance_mm ?? tolerance
  // STAC's command of this tab (reports.run_deviation): on a spoken confirmation
  const rcmd = useRef<{ matches: number; loading: boolean; run: (tol: number) => unknown }>({ matches: 0, loading: false, run: () => undefined })
  useEffect(() => {
    const ready = ({ tolerance_mm }: { tolerance_mm: number }): Outcome =>
      rcmd.current.matches === 0 || rcmd.current.loading ? { ok: false, reason: 'nothing_to_do' } : { ok: true, detail: `${sessionId}, ${tolerance_mm} mm` }
    const executors: Executors<ReportCommandId> = {
      'reports.run_deviation': { prepare: ready, run: a => { const o = ready(a); if (o.ok) void rcmd.current.run(a.tolerance_mm); return o } },
    }
    return registerCommands(executors)
  }, [sessionId])
  rcmd.current = { matches: matched ?? 0, loading: running, run: tol => { onTolerance(tol); onRun(tol) } }

  // how many objects of the scan on screen are paired with an element of the IFC
  useEffect(() => {
    let live = true
    setError(null)
    fetch(`/api/bim/auto_match/${sessionId}${scanKey ? `?scan=${encodeURIComponent(scanKey)}` : ''}`)
      .then(r => r.json())
      .then(data => { if (!live) return; setMatched(Array.isArray(data.matches) ? data.matches.length : 0); if (data.error) setError(data.error) })
      .catch(() => { if (live) setError(t('deviation.autoMatchFailed')) })
    return () => { live = false }
  }, [sessionId, scanKey, t])

  const aggStats = result?.results.reduce((acc, r) => {
    if (!r.stats) return acc
    return { totalPoints: acc.totalPoints + r.stats.total_points, withinTol: acc.withinTol + r.stats.within_tolerance, maxDev: Math.max(acc.maxDev, r.stats.max_mm), meanDevSum: acc.meanDevSum + r.stats.mean_mm * r.stats.total_points }
  }, { totalPoints: 0, withinTol: 0, maxDev: 0, meanDevSum: 0 })
  const passRate = aggStats && aggStats.totalPoints > 0 ? aggStats.withinTol / aggStats.totalPoints : 0
  const meanDev = aggStats && aggStats.totalPoints > 0 ? aggStats.meanDevSum / aggStats.totalPoints : 0
  const passTone = passRate >= 0.9 ? 'ok' : passRate >= 0.7 ? 'warn' : 'err'
  const hist = useMemo(() => (result ? wholeHistogram(result, shownTolerance, aggStats?.totalPoints ?? 0) : null),
    [result, shownTolerance, aggStats?.totalPoints])

  // Histogram: every compared point, one bar a slice of deviation from 0 to twice the tolerance (each bar in
  // the colour its points have on the model) and a last bar for the points beyond; the tolerance dashed
  useEffect(() => {
    if (!hist || !histCanvasRef.current) return
    const canvas = histCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const bars = hist.beyond > 0 ? [...hist.counts, hist.beyond] : hist.counts
    const maxCount = Math.max(...bars, 1)
    canvas.width = canvas.offsetWidth * 2
    canvas.height = canvas.offsetHeight * 2
    ctx.scale(2, 2)
    const w = canvas.offsetWidth, h = canvas.offsetHeight
    ctx.clearRect(0, 0, w, h)
    const barW = w / bars.length
    for (let i = 0; i < bars.length; i++) {
      const barH = (bars[i] / maxCount) * (h - 4)
      ctx.fillStyle = deviationColor(Math.min(1, i / Math.max(1, hist.counts.length - 1)))
      ctx.fillRect(i * barW, h - barH, barW - 1, barH)
    }
    const tolX = (shownTolerance / hist.maxMm) * barW * hist.counts.length
    if (tolX > 0 && tolX < w) {
      ctx.strokeStyle = tokenColor('--measure')
      ctx.lineWidth = 1.5
      ctx.setLineDash([4, 3])
      ctx.beginPath(); ctx.moveTo(tolX, 0); ctx.lineTo(tolX, h); ctx.stroke()
      ctx.setLineDash([])
    }
  }, [hist, shownTolerance])


  const columns: Column<DeviationElementResult>[] = [
    { id: 'label', header: t('report.element'), sortValue: r => r.label, cell: r => r.label },
    { id: 'pass', header: t('deviation.passRate'), align: 'right', mono: true, sortValue: r => r.stats?.pass_rate ?? -1,
      cell: r => r.error ? <Badge tone="err" size="sm">{t('deviation.error')}</Badge> : r.stats ? <Badge size="sm" tone={r.stats.pass_rate >= 90 ? 'ok' : r.stats.pass_rate >= 70 ? 'warn' : 'err'}>{fmt.percent(r.stats.pass_rate / 100)}</Badge> : t('status.dash') },
    { id: 'mean', header: t('deviation.mean'), align: 'right', mono: true, sortValue: r => r.stats?.mean_mm ?? -1, cell: r => r.stats ? fmt.mm(r.stats.mean_mm) : t('status.dash') },
    { id: 'p95', header: t('deviation.p95'), align: 'right', mono: true, sortValue: r => r.stats?.p95_mm ?? -1, cell: r => r.stats ? fmt.mm(r.stats.p95_mm) : t('status.dash') },
    { id: 'max', header: t('deviation.max'), align: 'right', mono: true, sortValue: r => r.stats?.max_mm ?? -1, cell: r => r.stats ? fmt.mm(r.stats.max_mm) : t('status.dash') },
  ]

  return (
    <div className={`stac-devrep ${compact ? 'stac-devrep--stacked' : ''}`.trim()}>
      <div className="stac-devrep__side">
        <Stack gap={3}>
          <Slider tone="measure" min={1} max={50} step={1} value={tolerance} onChange={onTolerance} label={t('deviation.tolerance')} format={v => fmt.mm(v, 0)} />
          <Button variant="primary" icon={<Play aria-hidden />} loading={running} disabled={!matched} onClick={() => onRun()}>{t('deviation.runComparison')}</Button>
          {!result && matched !== null && <p className="stac-section__hint">{matched > 0 ? t.plural('deviation.matched', matched) : t('deviation.noMatchesHint')}</p>}
          {error && <Banner tone="err" compact>{error}</Banner>}
          {result && aggStats && aggStats.totalPoints > 0 && (
            <>
              <Row wrap>
                <Readout label={t('deviation.mean')} value={meanDev} digits={1} unit="mm" provenance="tool_measured" />
                <Readout label={t('deviation.max')} value={aggStats.maxDev} digits={1} unit="mm" tone="warn" />
                <Readout label={t('deviation.passRate')} value={fmt.number(passRate * 100, 0)} unit="%" tone={passTone} />
              </Row>
              <Progress value={passRate * 100} tone={passTone} label={t('deviation.withinTolerance', { within: fmt.integer(aggStats.withinTol), total: fmt.integer(aggStats.totalPoints), tol: fmt.mm(shownTolerance, 0) })} />
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
              <Legend orientation="horizontal" ramp={DEV_RAMP} min={fmt.number(0, 0)} max={hist ? `${fmt.number(hist.maxMm, 0)}${hist.beyond > 0 ? '+' : ''}` : undefined} unit="mm" />
            </div>
            <Table columns={compact ? columns.filter(c => c.id === 'label' || c.id === 'pass' || c.id === 'mean') : columns} rows={result.results} rowKey={r => r.element_key} caption={t('report.elements', { n: result.results.length })} initialSort={{ id: 'pass', dir: 'asc' }}
              selectedKey={selectedInstanceId != null ? result.results.find(r => instancesOf(r).includes(selectedInstanceId))?.element_key : undefined}
              onRowClick={r => { const id = instancesOf(r)[0]; if (id !== undefined) onSelectInstance?.(id) }} />
          </>
        )}
      </div>
    </div>
  )
}
