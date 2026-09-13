import { useCallback, useEffect, useState } from 'react'

/** Visual validation kit (claude_stac.txt §11) — what the user needs to look
 *  where it matters. Nothing here decides: the certification loop measured,
 *  the acta says what and why; the user judges with his eyes and the
 *  Approve/Undo buttons (same flow as the correction module — a certification
 *  leaves a CHAIN of pending epochs: Undo pops the last one, Approve accepts
 *  them all).
 *
 *  Sections: epochs (before/after toggle, Approve/Undo, run), colour by
 *  status / mv_votes with the threshold slider, trajectory with edges
 *  (odometry grey, accepted green, scale_break orange, rejected/vetoed red
 *  with the reason), duplicates with fly-to, attention list (opens first),
 *  acta (§10 metrics per epoch, gates with value / threshold / verdict). */

type ColorMode = 'rgb' | 'status' | 'mv_votes'

export interface KitViewport {
  setWitnessColorMode: (mode: ColorMode, mvThreshold: number) => void
  setEpochLayer: (url: string | null) => void
  setEpochLayerVisible: (visible: boolean) => void
  setCloudObjectVisible: (visible: boolean) => void
  setTrajectoryEdges: (data: any | null, show: { odometry: boolean; loops: boolean }) => void
  flyToPoint: (p: number[], radius: number) => void
}

const KIND_COLOR: Record<string, string> = {
  accepted: '#3ddc84', scale_break: '#ff9f1c', vetoed: '#ff4d4d', rejected: '#ff4d4d', ambiguous: '#ffd166', odometry: '#9a9a9a',
}
const STATUS_COLOR: Record<string, string> = {
  verified: '#3ddc84', single_witness: '#ffd166', mask_conflict: '#ff4d4d', dynamic: '#d66cff', unobserved: '#8a8a8a',
}

function fmt(v: any, digits = 3): string {
  if (v == null || Number.isNaN(v)) return '—'
  if (typeof v === 'number') return Math.abs(v) < 1 && v !== 0 ? v.toFixed(digits) : v.toFixed(2)
  return String(v)
}

export default function CertifyKitPanel({ session, token, viewport, onStatus, onClose }: {
  session: string
  token: string | null
  viewport: KitViewport | null
  onStatus: (msg: string) => void
  onClose: () => void
}) {
  const [state, setState] = useState<any>(null)
  const [epochs, setEpochs] = useState<any>(null)
  const [edges, setEdges] = useState<any>(null)
  const [attention, setAttention] = useState<any>(null)
  const [report, setReport] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const [colorMode, setColorMode] = useState<ColorMode>('rgb')
  const [mvThreshold, setMvThreshold] = useState(2)
  const [showOdometry, setShowOdometry] = useState(true)
  const [showLoops, setShowLoops] = useState(true)
  const [beforeOn, setBeforeOn] = useState(false)
  const [afterOn, setAfterOn] = useState(true)
  const [tab, setTab] = useState<'attention' | 'edges' | 'duplicates' | 'acta'>('attention')

  const headers = useCallback((): HeadersInit => {
    const h: HeadersInit = { 'Content-Type': 'application/json' }
    if (token) (h as any)['Authorization'] = `Bearer ${token}`
    return h
  }, [token])

  const load = useCallback(async () => {
    const get = async (url: string) => {
      const r = await fetch(url)
      return r.ok ? r.json() : null
    }
    const [st, ep, ed, at, rp] = await Promise.all([
      get(`/api/certify/state/${session}`), get(`/api/certify/epochs/${session}`),
      get(`/api/certify/edges/${session}`), get(`/api/certify/attention/${session}`),
      get(`/api/certify/report/${session}`),
    ])
    setState(st); setEpochs(ep); setEdges(ed); setAttention(at); setReport(rp)
  }, [session])

  useEffect(() => { load() }, [load])

  // the trajectory + edges live in the viewer while the panel is open
  useEffect(() => {
    if (!viewport) return
    viewport.setTrajectoryEdges(edges, { odometry: showOdometry, loops: showLoops })
    return () => viewport.setTrajectoryEdges(null, { odometry: false, loops: false })
  }, [viewport, edges, showOdometry, showLoops])

  useEffect(() => {
    viewport?.setWitnessColorMode(colorMode, mvThreshold)
    return () => viewport?.setWitnessColorMode('rgb', 0)
  }, [viewport, colorMode, mvThreshold])

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
    onStatus(kind === 'approve' ? '🧪 approving the epoch chain…' : '🧪 undoing the last epoch (restores the previous one exactly)…')
    try {
      const r = await fetch(`/api/certify/${kind}`, { method: 'POST', headers: headers(), body: JSON.stringify({ session_id: session }) })
      const d = await r.json().catch(() => ({}))
      onStatus(r.ok ? `🧪 ${kind === 'approve' ? 'approved' : 'undone'} — epoch ${d.epoch ?? '?'}` : `🧪 ${kind} failed: ${typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail || d)}`)
    } catch (e) { onStatus(`🧪 ${kind} failed`) }
    setBusy(false)
    load()
  }

  const run = async () => {
    setBusy(true)
    onStatus('🧪 certification loop running (scale → poses → depth → witnesses per iteration)…')
    try {
      const r = await fetch('/api/certify/run', { method: 'POST', headers: headers(), body: JSON.stringify({ session_id: session }) })
      const d = await r.json().catch(() => ({}))
      onStatus(r.ok ? `🧪 certification: ${d.acta?.stop_reason || 'done'} (epoch ${d.acta?.epoch_final})` : `🧪 certification failed: ${typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail || d)}`)
    } catch (e) { onStatus('🧪 certification failed') }
    setBusy(false)
    load()
  }

  const fly = (p: number[] | null | undefined, radius = 3) => { if (p && viewport) viewport.flyToPoint(p, radius) }
  const acta = state?.acta
  const metrics = report?.metrics
  const prevWithPotree = epochs?.previous?.find((p: any) => p.potree)
  const pending = state?.pending_epochs ?? 0
  const iterations: any[] = acta?.iterations || []
  const dupList: any[] = edges?.duplicates || []
  const loopList: any[] = edges?.loops || []

  return (
    <div className="admin-overlay" style={{ zIndex: 2000, pointerEvents: 'none' }}>
      <div className="admin-panel" style={{ maxWidth: 520, maxHeight: '86vh', overflow: 'auto', pointerEvents: 'auto', position: 'absolute', right: 16, top: 60 }}>
        <div className="admin-header">
          <h2>🧪 Certification kit</h2>
          <button className="admin-close" onClick={onClose}>✕</button>
        </div>
        <div style={{ padding: 14, fontSize: 12, color: 'var(--text-secondary)' }}>
          {/* ── epochs: state, before/after, verdict ── */}
          <div style={{ marginBottom: 12, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <div style={{ color: 'var(--text-primary)', fontSize: 13, marginBottom: 6 }}>
              <b>Epoch {epochs?.epoch ?? state?.epoch ?? '—'}</b>
              {pending > 0 ? ` · ${pending} pending epoch${pending > 1 ? 's' : ''} (Undo pops the last, Approve accepts all)` : ' · nothing pending'}
              {state?.witness_fields ? ' · witnesses in the cloud' : ' · no witness fields yet'}
              {state?.running_task ? ' · ⏳ running' : ''}
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
              <label style={{ display: 'flex', gap: 4, alignItems: 'center' }} title={prevWithPotree ? `epoch ${prevWithPotree.epoch} octree, tinted orange` : 'the previous epoch has no octree (apply.potree_rebuild off) — nothing to overlay'}>
                <input type="checkbox" disabled={!prevWithPotree} checked={beforeOn} onChange={e => setBeforeOn(e.target.checked)} /> before (epoch {prevWithPotree?.epoch ?? '—'})
              </label>
              <label style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                <input type="checkbox" checked={afterOn} onChange={e => setAfterOn(e.target.checked)} /> after (current)
              </label>
              <span style={{ flex: 1 }} />
              <button className="bim-action-btn upload" disabled={busy || pending === 0} onClick={() => verdict('approve')}>✅ Approve</button>
              <button className="bim-action-btn upload" disabled={busy || pending === 0} onClick={() => verdict('undo')}>↩ Undo epoch</button>
              <button className="bim-action-btn upload" disabled={busy || !!state?.running_task} onClick={run}>▶ Run certification</button>
            </div>
            {acta && (
              <div style={{ marginTop: 6 }}>
                stopped: <b>{acta.stop_reason}</b> · {acta.elapsed_s}s ·
                {' '}{iterations.map((it: any) => `it${it.iteration}:${it.verdict}${it.objective != null ? ` (${fmt(it.objective_prev)}→${fmt(it.objective)})` : ''}`).join(' · ')}
              </div>
            )}
          </div>

          {/* ── colour mode ── */}
          <div style={{ marginBottom: 12, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <div style={{ color: 'var(--text-primary)', fontSize: 13, marginBottom: 6 }}><b>Colour</b></div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
              {(['rgb', 'status', 'mv_votes'] as ColorMode[]).map(m => (
                <label key={m} style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                  <input type="radio" name="kit-color" checked={colorMode === m} onChange={() => setColorMode(m)} /> {m}
                </label>
              ))}
              {colorMode === 'mv_votes' && (
                <label style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  threshold <input type="range" min={0} max={8} step={1} value={mvThreshold} onChange={e => setMvThreshold(parseInt(e.target.value))} /> {mvThreshold} views
                </label>
              )}
            </div>
            {colorMode === 'status' && (
              <div style={{ display: 'flex', gap: 10, marginTop: 6, flexWrap: 'wrap' }}>
                {Object.entries(STATUS_COLOR).map(([k, c]) => (
                  <span key={k}><span style={{ display: 'inline-block', width: 10, height: 10, background: c, borderRadius: 2, marginRight: 4 }} />{k}
                    {metrics?.witnesses?.status_counts ? ` ${(100 * (metrics.witnesses.status_fraction?.[k] ?? 0)).toFixed(1)}%` : ''}</span>
                ))}
              </div>
            )}
            {colorMode === 'mv_votes' && <div style={{ marginTop: 4 }}>red = fewer than {mvThreshold} agreeing views, green = more</div>}
          </div>

          {/* ── trajectory ── */}
          <div style={{ marginBottom: 12, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <div style={{ color: 'var(--text-primary)', fontSize: 13, marginBottom: 6 }}><b>Trajectory</b></div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <label><input type="checkbox" checked={showOdometry} onChange={e => setShowOdometry(e.target.checked)} /> odometry</label>
              <label><input type="checkbox" checked={showLoops} onChange={e => setShowLoops(e.target.checked)} /> loop edges</label>
              {Object.entries(KIND_COLOR).filter(([k]) => k !== 'odometry').map(([k, c]) => (
                <span key={k}><span style={{ display: 'inline-block', width: 10, height: 10, background: c, borderRadius: 2, marginRight: 4 }} />{k}</span>
              ))}
            </div>
          </div>

          {/* ── tabs ── */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
            {(['attention', 'edges', 'duplicates', 'acta'] as const).map(t => (
              <button key={t} className="bim-action-btn upload" style={{ opacity: tab === t ? 1 : 0.6 }} onClick={() => setTab(t)}>
                {t === 'attention' ? `👁 attention (${attention?.n ?? 0})` : t === 'edges' ? `↔ edges (${loopList.length})` : t === 'duplicates' ? `⧉ duplicates (${dupList.length})` : '📋 acta'}
              </button>
            ))}
          </div>

          {tab === 'attention' && (
            <div>
              <div style={{ marginBottom: 6 }}>Where the system is least sure — look here first, not where the cloud looks nice.</div>
              {(attention?.items || []).length === 0 && <div>nothing flagged</div>}
              {(attention?.items || []).map((it: any, i: number) => (
                <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'flex-start', padding: '4px 0', borderBottom: '1px solid var(--border-color)' }}>
                  <span style={{ color: it.severity >= 3 ? '#ff4d4d' : it.severity === 2 ? '#ff9f1c' : '#ffd166' }}>●</span>
                  <span style={{ flex: 1 }}><b>{it.kind}</b> — {it.text}</span>
                  {it.anchor?.position && <button className="bim-action-btn upload" onClick={() => fly(it.anchor.position)}>fly to</button>}
                  {it.anchor_b?.position && <button className="bim-action-btn upload" onClick={() => fly(it.anchor_b.position)}>other side</button>}
                </div>
              ))}
            </div>
          )}

          {tab === 'edges' && (
            <div>
              {loopList.map((e: any, i: number) => {
                const pi = edges?.positions?.[e.i], pj = edges?.positions?.[e.j]
                return (
                  <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '3px 0', borderBottom: '1px solid var(--border-color)' }}>
                    <span style={{ display: 'inline-block', width: 10, height: 10, background: KIND_COLOR[e.kind] || '#fff', borderRadius: 2 }} />
                    <span style={{ flex: 1 }}>kf {e.i} ↔ {e.j} · {e.kind} · {e.source}{e.residual_m != null ? ` · ${(e.residual_m * 100).toFixed(1)} cm` : ''}{e.observability ? ` · ${e.observability}` : ''}
                      {e.reason ? <span style={{ color: '#ff9f1c' }}> — {e.reason}</span> : null}</span>
                    {pi && <button className="bim-action-btn upload" onClick={() => fly(pi)}>i</button>}
                    {pj && <button className="bim-action-btn upload" onClick={() => fly(pj)}>j</button>}
                  </div>
                )
              })}
            </div>
          )}

          {tab === 'duplicates' && (
            <div>
              <div style={{ marginBottom: 6 }}>Target: an empty list.</div>
              {dupList.length === 0 && <div style={{ color: '#3ddc84' }}>no duplicates</div>}
              {dupList.map((d: any, i: number) => (
                <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'center', padding: '3px 0', borderBottom: '1px solid var(--border-color)' }}>
                  <span style={{ flex: 1 }}>{d.label}{d.instance_id != null ? `#${d.instance_id}` : ''} · kf {d.i} ↔ {d.j}
                    {d.separation_m != null ? ` · ${(d.separation_m * 100).toFixed(1)} cm` : ''}
                    {d.closure_after_m != null ? ` → ${(d.closure_after_m * 100).toFixed(1)} cm` : ''} · {d.verdict}</span>
                  {d.i_pos && <button className="bim-action-btn upload" onClick={() => fly(d.i_pos)}>fly to</button>}
                  {d.j_pos && <button className="bim-action-btn upload" onClick={() => fly(d.j_pos)}>copy</button>}
                </div>
              ))}
            </div>
          )}

          {tab === 'acta' && (
            <div>
              {!report && <div>no quality report yet — run the certification</div>}
              {report && (
                <>
                  <div style={{ color: 'var(--text-primary)', marginBottom: 4 }}><b>Report epoch {report.epoch}</b>{report.available_epochs ? ` (epochs ${report.available_epochs.join(', ')})` : ''}</div>
                  <table className="cvd-table" style={{ width: '100%' }}>
                    <tbody>
                      <tr><td>objective</td><td>{fmt(metrics?.objective)}</td><td>{report.comparison_vs_previous?.objective != null ? `Δ ${fmt(report.comparison_vs_previous.objective)}` : ''}</td></tr>
                      <tr><td>loop demand (median)</td><td>{metrics?.loop_residual?.median_demand_m != null ? `${(metrics.loop_residual.median_demand_m * 100).toFixed(1)} cm` : '—'}</td><td>{metrics?.loop_residual?.n_accepted} accepted / {metrics?.loop_residual?.n_edges}</td></tr>
                      <tr><td>seams (held-out)</td><td>{metrics?.seam_residual?.median_m != null ? `${(metrics.seam_residual.median_m * 100).toFixed(2)} cm` : '—'}</td><td>p95 {metrics?.seam_residual?.p95_m != null ? `${(metrics.seam_residual.p95_m * 100).toFixed(2)} cm` : '—'}</td></tr>
                      <tr><td>closure</td><td>{metrics?.closure?.median_m != null ? `${(metrics.closure.median_m * 100).toFixed(1)} cm` : '—'}</td><td>max {metrics?.closure?.max_m != null ? `${(metrics.closure.max_m * 100).toFixed(1)} cm` : '—'}</td></tr>
                      <tr><td>duplicates</td><td>{metrics?.duplicates?.n}</td><td>target {metrics?.duplicates?.target}</td></tr>
                      <tr><td>scale</td><td>{metrics?.scale?.applied ? `r ${(metrics.scale.r_per_chunk || []).map((x: number) => x.toFixed(4)).join(' ')}` : 'identity'}</td><td>{metrics?.scale?.scale_break ? 'scale_break' : ''}</td></tr>
                      <tr><td>witnesses</td><td>{metrics?.witnesses ? Object.entries(metrics.witnesses.status_fraction || {}).map(([k, v]: any) => `${k} ${(100 * v).toFixed(1)}%`).join(' · ') : '—'}</td><td>mv_votes mean {fmt(metrics?.witnesses?.mv_votes_mean)} · conflicts {metrics?.witnesses?.mask_conflicts_total}</td></tr>
                      <tr><td>depth disagreement</td><td>{metrics?.depth_disagreement?.holdout_rel_after != null ? `${(100 * metrics.depth_disagreement.holdout_rel_after).toFixed(2)}%` : fmt(metrics?.depth_disagreement?.pair_rel_median_before)}</td><td>{metrics?.depth_disagreement?.applied ? 'corrected' : 'identity'}</td></tr>
                      <tr><td>known dimensions</td><td>{metrics?.known_dimensions ? `p50 ${fmt(metrics.known_dimensions.p50_m)} m` : 'none registered'}</td><td>{metrics?.known_dimensions ? `p95 ${fmt(metrics.known_dimensions.p95_m)} m` : ''}</td></tr>
                      <tr><td>loop coverage</td><td>{metrics?.loop_coverage ? `${(100 * metrics.loop_coverage.coverage).toFixed(0)}%` : '—'}</td><td>{metrics?.authority?.pose_graph ? `authority ${(100 * metrics.authority.pose_graph.fraction_used).toFixed(0)}%${metrics.authority.pose_graph.saturated ? ' SATURATED' : ''}` : ''}</td></tr>
                    </tbody>
                  </table>
                  {iterations.map((it: any) => (
                    <div key={it.iteration} style={{ marginTop: 8 }}>
                      <div style={{ color: 'var(--text-primary)' }}>iteration {it.iteration}: <b>{it.verdict}</b>{it.reason ? ` — ${it.reason}` : ''}</div>
                      {(it.gates || []).map((g: any) => (
                        <div key={g.name}>{g.passed ? '✅' : '❌'} {g.name} — {fmt(g.value)} vs {fmt(g.threshold)}{g.detail ? ` (${g.detail})` : ''}</div>
                      ))}
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
