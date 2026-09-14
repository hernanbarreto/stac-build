import { useCallback, useEffect, useState } from 'react'
import { CheckCircle2, Crosshair, Undo2, AlertTriangle } from 'lucide-react'
import { Section, Row, Stack, KeyValue } from './ui/Panel'
import { Button } from './ui/Button'
import { Checkbox, SegmentedControl, Slider } from './ui/Field'
import { Badge } from './ui/Badge'
import { Tabs } from './ui/Tabs'
import { Table, type Column } from './ui/Table'
import { Legend } from './ui/Legend'
import { EmptyState } from './ui/EmptyState'
import { GateList } from '../features/CorrectionDialog'
import { EDGE_TOKENS, STATUS_TOKENS } from './viewport/palette'
import { useFmt, useT } from '../i18n'
import type { ColorMode } from './viewport/FloatingToolbar'

/** Visual validation kit (claude_stac.txt §11) — what the user needs to look
 *  where it matters. Nothing here decides: the certification loop measured,
 *  the acta says what and why; the user judges with his eyes and the
 *  Approve/Undo buttons (same flow as the correction module — a certification
 *  leaves a CHAIN of pending epochs: Undo pops the last one, Approve accepts
 *  them all).
 *
 *  Lives in the Inspector's "Acta / Quality" tab (prompt_ui.txt §5). The
 *  colour mode (rgb / status / votes) is owned by App and shared with the
 *  viewport toolbar, so the two controls never fight over the viewer. */

export interface KitViewport {
  setWitnessColorMode: (mode: 'rgb' | 'status' | 'mv_votes', mvThreshold: number) => void
  setEpochLayer: (url: string | null) => void
  setEpochLayerVisible: (visible: boolean) => void
  setCloudObjectVisible: (visible: boolean) => void
  setTrajectoryEdges: (data: any | null, show: { odometry: boolean; loops: boolean }) => void
  flyToPoint: (p: number[], radius: number) => void
}

type KitTab = 'attention' | 'edges' | 'duplicates' | 'acta'

export default function CertifyKitPanel({ session, token, viewport, onStatus, refreshKey = 0, colorMode, onColorMode, mvThreshold, onMvThreshold, onStateLoaded }: {
  session: string
  token: string | null
  viewport: KitViewport | null
  onStatus: (msg: string) => void
  /** bumped by the app when the pipeline's automatic certification finishes
   *  (the cloud reloads) — the kit re-reads the acta/epochs/edges */
  refreshKey?: number
  colorMode: ColorMode
  onColorMode: (m: ColorMode) => void
  mvThreshold: number
  onMvThreshold: (v: number) => void
  onStateLoaded?: (state: any) => void
}) {
  const t = useT()
  const fmt = useFmt()
  const [state, setState] = useState<any>(null)
  const [epochs, setEpochs] = useState<any>(null)
  const [edges, setEdges] = useState<any>(null)
  const [attention, setAttention] = useState<any>(null)
  const [report, setReport] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [showOdometry, setShowOdometry] = useState(true)
  const [showLoops, setShowLoops] = useState(true)
  const [beforeOn, setBeforeOn] = useState(false)
  const [afterOn, setAfterOn] = useState(true)
  const [tab, setTab] = useState<KitTab>('attention')

  const headers = useCallback((): HeadersInit => {
    const h: HeadersInit = { 'Content-Type': 'application/json' }
    if (token) (h as any)['Authorization'] = `Bearer ${token}`
    return h
  }, [token])

  const load = useCallback(async () => {
    const get = async (url: string) => { const r = await fetch(url); return r.ok ? r.json() : null }
    const [st, ep, ed, at, rp] = await Promise.all([
      get(`/api/certify/state/${session}`), get(`/api/certify/epochs/${session}`),
      get(`/api/certify/edges/${session}`), get(`/api/certify/attention/${session}`),
      get(`/api/certify/report/${session}`),
    ])
    setState(st); setEpochs(ep); setEdges(ed); setAttention(at); setReport(rp)
    onStateLoaded?.(st ? { ...st, report_metrics: rp?.metrics } : null)
  }, [session, onStateLoaded])

  useEffect(() => { load() }, [load, refreshKey])

  // the trajectory + edges live in the viewer while the panel is open
  useEffect(() => {
    if (!viewport) return
    viewport.setTrajectoryEdges(edges, { odometry: showOdometry, loops: showLoops })
    return () => viewport.setTrajectoryEdges(null, { odometry: false, loops: false })
  }, [viewport, edges, showOdometry, showLoops])

  // before/after: the previous pending epoch's octree as a tinted layer
  useEffect(() => {
    if (!viewport) return
    const prev = epochs?.previous?.find((p: any) => p.potree)
    if (beforeOn && prev) viewport.setEpochLayer(`/potree_epoch/${session}/${prev.epoch}/`)
    else viewport.setEpochLayer(null)
    viewport.setCloudObjectVisible(afterOn)
    return () => { viewport.setEpochLayer(null); viewport.setCloudObjectVisible(true) }
  }, [viewport, epochs, beforeOn, afterOn, session])

  const verdict = async (kind: 'approve' | 'undo') => {
    setBusy(true)
    onStatus(kind === 'approve' ? t('certify.approving') : t('certify.undoing'))
    try {
      const r = await fetch(`/api/certify/${kind}`, { method: 'POST', headers: headers(), body: JSON.stringify({ session_id: session }) })
      const d = await r.json().catch(() => ({}))
      onStatus(r.ok ? t(kind === 'approve' ? 'certify.approved' : 'certify.undone', { epoch: d.epoch ?? '' })
        : t('certify.verdictFailed', { detail: typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail || d) }))
    } catch { onStatus(t('certify.verdictFailed', { detail: '' })) }
    setBusy(false)
    load()
  }

  const fly = (p: number[] | null | undefined, radius = 3) => { if (p && viewport) viewport.flyToPoint(p, radius) }
  const cm = (m: number | null | undefined, digits = 1) => (m != null ? `${fmt.number(m * 100, digits)} cm` : t('status.dash'))
  const acta = state?.acta
  const metrics = report?.metrics
  const prevWithPotree = epochs?.previous?.find((p: any) => p.potree)
  const pending = state?.pending_epochs ?? 0
  const iterations: any[] = acta?.iterations || []
  const dupList: any[] = edges?.duplicates || []
  const loopList: any[] = edges?.loops || []
  const kitMode = colorMode === 'status' || colorMode === 'mv_votes' ? colorMode : 'rgb'

  const edgeColumns: Column<any>[] = [
    { id: 'kind', header: t('edges.kind'), cell: e => <Badge size="sm" tone={e.kind === 'accepted' ? 'ok' : e.kind === 'scale_break' ? 'warn' : e.kind === 'rejected' || e.kind === 'vetoed' ? 'err' : 'info'}>{t(`edgeKind.${e.kind}`)}</Badge> },
    { id: 'pair', header: t('edges.pair'), mono: true, cell: e => `${e.i} / ${e.j}`, sortValue: e => e.i },
    { id: 'source', header: t('edges.source'), cell: e => e.source },
    { id: 'residual', header: t('edges.residual'), align: 'right', mono: true, sortValue: e => e.residual_m ?? -1, cell: e => cm(e.residual_m) },
    { id: 'reason', header: t('edges.reason'), cell: e => e.reason || e.observability || '' },
    { id: 'fly', header: '', align: 'right', cell: e => (
      <Row>
        {edges?.positions?.[e.i] && <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} onClick={() => fly(edges.positions[e.i])}>{t('edges.flyI')}</Button>}
        {edges?.positions?.[e.j] && <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} onClick={() => fly(edges.positions[e.j])}>{t('edges.flyJ')}</Button>}
      </Row>
    ) },
  ]
  const dupColumns: Column<any>[] = [
    { id: 'label', header: t('duplicates.instance'), cell: d => `${d.label}${d.instance_id != null ? ` ${d.instance_id}` : ''}` },
    { id: 'pair', header: t('edges.pair'), mono: true, cell: d => `${d.i} / ${d.j}` },
    { id: 'sep', header: t('duplicates.separation'), align: 'right', mono: true, sortValue: d => d.separation_m ?? -1, cell: d => cm(d.separation_m) },
    { id: 'after', header: t('duplicates.after'), align: 'right', mono: true, cell: d => cm(d.closure_after_m) },
    { id: 'verdict', header: t('duplicates.verdict'), cell: d => d.verdict },
    { id: 'fly', header: '', align: 'right', cell: d => (
      <Row>
        {d.i_pos && <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} onClick={() => fly(d.i_pos)}>{t('edges.flyI')}</Button>}
        {d.j_pos && <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} onClick={() => fly(d.j_pos)}>{t('edges.flyJ')}</Button>}
      </Row>
    ) },
  ]

  return (
    <div className="stac-kit">
      <Section title={t('certify.epochTitle', { epoch: epochs?.epoch ?? state?.epoch ?? t('status.dash') })}
        actions={state?.running_task ? <Badge tone="brand" dot size="sm">{t('certify.running')}</Badge> : undefined}>
        <p className="stac-section__hint">
          {pending > 0 ? t.plural('certify.pendingEpochs', pending) : t('certify.nothingPending')} {state?.witness_fields ? t('certify.witnessesPresent') : t('certify.noWitnesses')}
        </p>
        <Row wrap>
          <Checkbox checked={beforeOn} disabled={!prevWithPotree} onChange={setBeforeOn} label={t('certify.before', { epoch: prevWithPotree?.epoch ?? t('status.dash') })}
            description={prevWithPotree ? undefined : t('certify.noPreviousOctree')} />
          <Checkbox checked={afterOn} onChange={setAfterOn} label={t('certify.after')} />
        </Row>
        <Row>
          <Button block variant="primary" icon={<CheckCircle2 aria-hidden />} disabled={busy || pending === 0} loading={busy} onClick={() => verdict('approve')}>{t('certify.approveChain')}</Button>
          <Button block icon={<Undo2 aria-hidden />} disabled={busy || pending === 0} onClick={() => verdict('undo')}>{t('certify.undoEpoch')}</Button>
        </Row>
        {!acta && !state?.running_task && <p className="stac-section__hint">{t('certify.noActa')}</p>}
        {acta && (
          <KeyValue label={t('certify.stopped')} mono>{acta.stop_reason} ({fmt.duration(acta.elapsed_s || 0)})</KeyValue>
        )}
      </Section>

      <Section title={t('toolbar.colorBy')}>
        <SegmentedControl<'rgb' | 'status' | 'mv_votes'> size="sm" ariaLabel={t('toolbar.colorBy')} value={kitMode} onChange={m => onColorMode(m)} options={[
          { value: 'rgb', label: t('colorMode.rgb') }, { value: 'status', label: t('colorMode.status') }, { value: 'mv_votes', label: t('colorMode.votes') },
        ]} />
        {kitMode === 'mv_votes' && <Slider min={0} max={8} step={1} value={mvThreshold} onChange={onMvThreshold} label={t('colorMode.threshold')} format={v => t('colorMode.views', { n: fmt.integer(v) })} />}
        {kitMode === 'status' && (
          <Legend orientation="horizontal" categories={Object.entries(STATUS_TOKENS).map(([k, tok]) => ({
            color: tok, label: t(`witness.${k}`), value: metrics?.witnesses?.status_fraction?.[k] != null ? fmt.percent(metrics.witnesses.status_fraction[k], 1) : undefined,
          }))} />
        )}
        {kitMode === 'mv_votes' && <p className="stac-section__hint">{t('colorMode.votesHint', { n: fmt.integer(mvThreshold) })}</p>}
      </Section>

      <Section title={t('edges.trajectory')}>
        <Row wrap>
          <Checkbox checked={showOdometry} onChange={setShowOdometry} label={t('edgeKind.odometry')} />
          <Checkbox checked={showLoops} onChange={setShowLoops} label={t('edges.loopEdges')} />
        </Row>
        <Legend orientation="horizontal" categories={Object.entries(EDGE_TOKENS).filter(([k]) => k !== 'odometry').map(([k, tok]) => ({ color: tok, label: t(`edgeKind.${k}`) }))} />
      </Section>

      <Tabs<KitTab> size="sm" ariaLabel={t('certify.sections')} value={tab} onChange={setTab} items={[
        { id: 'attention', label: t('certify.attention'), badge: attention?.n ?? 0 },
        { id: 'edges', label: t('edges.title'), badge: loopList.length },
        { id: 'duplicates', label: t('duplicates.title'), badge: dupList.length },
        { id: 'acta', label: t('inspector.acta') },
      ]} />

      <div className="stac-kit__body">
        {tab === 'attention' && (
          <Stack gap={2}>
            <p className="stac-section__hint">{t('certify.attentionHint')}</p>
            {(attention?.items || []).length === 0 && <EmptyState compact title={t('certify.nothingFlagged')} />}
            {(attention?.items || []).map((it: any, i: number) => (
              <div key={i} className="stac-attention">
                <span className={`stac-attention__sev stac-attention__sev--${it.severity >= 3 ? 'err' : it.severity === 2 ? 'warn' : 'info'}`} aria-hidden><AlertTriangle /></span>
                <span className="stac-attention__text"><strong>{it.kind}</strong> {it.text}</span>
                {it.anchor?.position && <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} onClick={() => fly(it.anchor.position)}>{t('certify.flyTo')}</Button>}
                {it.anchor_b?.position && <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} onClick={() => fly(it.anchor_b.position)}>{t('certify.otherSide')}</Button>}
              </div>
            ))}
          </Stack>
        )}
        {tab === 'edges' && <Table columns={edgeColumns} rows={loopList} rowKey={(_, i) => i} emptyTitle={t('edges.none')} caption={t('edges.title')} />}
        {tab === 'duplicates' && (
          <Stack gap={2}>
            <p className="stac-section__hint">{t('duplicates.target')}</p>
            <Table columns={dupColumns} rows={dupList} rowKey={(_, i) => i} emptyTitle={t('duplicates.none')} caption={t('duplicates.title')} />
          </Stack>
        )}
        {tab === 'acta' && (
          <Stack gap={3}>
            {!report && <EmptyState compact title={t('certify.noReport')} description={t('certify.noActa')} />}
            {report && (
              <>
                <KeyValue label={t('certify.reportEpoch')} mono>{report.epoch}{report.available_epochs ? ` (${report.available_epochs.join(', ')})` : ''}</KeyValue>
                <KeyValue label={t('acta.objective')} mono>{fmtNum(metrics?.objective, fmt)}{report.comparison_vs_previous?.objective != null ? ` (Δ ${fmtNum(report.comparison_vs_previous.objective, fmt)})` : ''}</KeyValue>
                <KeyValue label={t('acta.loopDemand')} mono>{cm(metrics?.loop_residual?.median_demand_m)} — {metrics?.loop_residual?.n_accepted ?? 0} / {metrics?.loop_residual?.n_edges ?? 0}</KeyValue>
                <KeyValue label={t('acta.seams')} mono>{`${cm(metrics?.seam_residual?.median_m, 2)} (p95 ${cm(metrics?.seam_residual?.p95_m, 2)})`}</KeyValue>
                <KeyValue label={t('acta.closure')} mono>{`${cm(metrics?.closure?.median_m)} (max ${cm(metrics?.closure?.max_m)})`}</KeyValue>
                <KeyValue label={t('duplicates.title')} mono>{metrics?.duplicates?.n ?? t('status.dash')} ({t('duplicates.targetShort')} {metrics?.duplicates?.target ?? 0})</KeyValue>
                <KeyValue label={t('acta.scale')} mono>{metrics?.scale?.applied ? (metrics.scale.r_per_chunk || []).map((x: number) => fmt.number(x, 4)).join(' ') : t('acta.identity')}{metrics?.scale?.scale_break ? ` ${t('edgeKind.scale_break')}` : ''}</KeyValue>
                <KeyValue label={t('acta.witnesses')} mono>{metrics?.witnesses ? Object.entries(metrics.witnesses.status_fraction || {}).map(([k, v]: any) => `${t(`witness.${k}`)} ${fmt.percent(v, 1)}`).join(', ') : t('status.dash')}</KeyValue>
                <KeyValue label={t('acta.depthDisagreement')} mono>{metrics?.depth_disagreement?.holdout_rel_after != null ? fmt.percent(metrics.depth_disagreement.holdout_rel_after, 2) : fmtNum(metrics?.depth_disagreement?.pair_rel_median_before, fmt)} ({metrics?.depth_disagreement?.applied ? t('acta.corrected') : t('acta.identity')})</KeyValue>
                <KeyValue label={t('acta.knownDimensions')} mono>{metrics?.known_dimensions ? `p50 ${fmt.lengthText(metrics.known_dimensions.p50_m)} / p95 ${fmt.lengthText(metrics.known_dimensions.p95_m)}` : t('acta.noneRegistered')}</KeyValue>
                <KeyValue label={t('acta.loopCoverage')} mono>{metrics?.loop_coverage ? fmt.percent(metrics.loop_coverage.coverage) : t('status.dash')}{metrics?.authority?.pose_graph ? ` — ${t('acta.authority')} ${fmt.percent(metrics.authority.pose_graph.fraction_used)}${metrics.authority.pose_graph.saturated ? ` (${t('acta.saturated')})` : ''}` : ''}</KeyValue>
                {iterations.map((it: any) => (
                  <Section key={it.iteration} flush title={t('acta.iteration', { n: it.iteration, verdict: it.verdict })} hint={`${it.reason || ''}${it.gate_mode ? ` — ${t('acta.gates')} ${it.gate_mode}` : ''}`}>
                    <GateList compact gates={(it.gates || []).map((g: any) => ({ name: g.name, passed: g.passed, advisory: g.advisory, detail: `${fmtNum(g.value, fmt)} vs ${fmtNum(g.threshold, fmt)}${g.detail ? ` (${g.detail})` : ''}` }))} />
                  </Section>
                ))}
              </>
            )}
          </Stack>
        )}
      </div>
    </div>
  )
}

function fmtNum(v: any, fmt: ReturnType<typeof useFmt>): string {
  if (v == null || Number.isNaN(v)) return '—'
  if (typeof v === 'number') return Math.abs(v) < 1 && v !== 0 ? fmt.number(v, 3) : fmt.number(v, 2)
  return String(v)
}
