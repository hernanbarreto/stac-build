import { useState } from 'react'
import './ConfirmDialog.css'

/** Pending-correction verdict (USER 2026-09-06: "ponelo lindo en el medio,
 *  como uno que ya usamos de aceptar o rechazar"). Same look as the
 *  ConfirmDialog, centred, but WITHOUT a blocking backdrop — the user must
 *  orbit the cloud to judge the correction. A ⌃ collapses it to a small
 *  pill at the top so the centre of the viewport is free; click reopens.
 *  Approve makes the corrected cloud THE cloud; Undo restores the
 *  previous one (one level). */
export default function CorrectionVerdictDialog({ state, session, otherSession, busy, onApprove, onUndo }: {
  state: any
  session: string | null
  otherSession: string | null
  busy: boolean
  onApprove: () => void
  onUndo: () => void
}) {
  const [collapsed, setCollapsed] = useState(false)
  const report = state?.report
  const solutions: any[] = report?.solutions || []
  const gates: any[] = report?.gates || []
  const kind = state?.kind || report?.kind || 'correction'
  const epoch = report?.epoch_to ?? state?.epoch
  const title = otherSession ? `Correction pending in ${otherSession}` : `Correction applied${session ? ` on ${session}` : ''} — your verdict`

  if (collapsed) {
    return (
      <button className="cvd-pill" onClick={() => setCollapsed(false)} title="show the pending correction">
        ⚠ Correction pending{otherSession ? ` in ${otherSession}` : ''} — click to decide
      </button>
    )
  }
  return (
    <div className="cvd-wrap">
      <div className="cd-dialog cvd-dialog">
        <div className="cd-header">
          <img src="/logo.png" alt="STAC" className="cd-logo" />
          <span className="cd-app-name">STAC Build</span>
          <span style={{ flex: 1 }} />
          <button className="cvd-collapse" onClick={() => setCollapsed(true)} title="collapse — inspect the cloud">⌃</button>
        </div>
        <div className="cd-body">
          <span className="cd-icon">🔧</span>
          <div className="cd-content">
            <div className="cd-title">{title}</div>
            <div className="cd-message">
              <div>{kind}{epoch != null ? ` · epoch ${epoch}` : ''}{report?.points_moved ? ` · ${(report.points_moved / 1e6).toFixed(2)}M points moved` : ''}{report?.operator ? ` · by ${report.operator}` : ''}</div>
              {solutions.length > 0 && (
                <table className="cvd-table">
                  <thead><tr><th>visit (kf)</th><th>rigid</th><th>k</th><th>copies before → after</th><th>DOF</th></tr></thead>
                  <tbody>
                    {solutions.map((c, i) => (
                      <tr key={i}>
                        <td>{c.kf_span ? `${c.kf_span[0]}..${c.kf_span[1]}` : c.later_kfs ? `${c.earlier_kfs[0]}..${c.earlier_kfs[1]} ↔ ${c.later_kfs[0]}..${c.later_kfs[1]}` : (c.anchors ? `${c.anchors.length} anchors` : '—')}</td>
                        <td>{c.rot_deg != null ? `${c.rot_deg}° / ${c.t_norm_m ?? c.t_m} m` : '—'}</td>
                        <td>{c.k && c.k !== 1 ? c.k : '—'}</td>
                        <td>{c.residual_cm ? <>{c.residual_cm.before} → <b>{c.residual_cm.after}</b> cm</>
                          : c.per_block ? c.per_block.map((b: any) => `R${b.region} ${b.before_cm}→${b.after_cm}`).join(' · ') : '—'}</td>
                        <td>{c.dof ? c.dof.join(',') : c.n_blocks != null ? `yaw+t · ${c.blocks_improved}/${c.n_blocks} blocks` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {gates.length > 0 && (
                <div style={{ marginTop: 6, opacity: 0.85 }}>
                  {gates.map((g: any) => (
                    <div key={g.name + String(g.group ?? '')}>{g.advisory ? '⚠' : g.passed ? '✅' : '❌'} {g.name} — {g.detail}{g.advisory ? ' (advisory)' : ''}</div>
                  ))}
                </div>
              )}
              {report?.distribution && (
                <div style={{ marginTop: 6, opacity: 0.85 }}>
                  Spread over {report.distribution.keyframes_warped} keyframes after kf {report.distribution.identity_until_kf}
                  {' '}· max step {report.distribution.max_step_between_keyframes_mm} mm / {report.distribution.max_step_between_keyframes_deg}° (no seam)
                </div>
              )}
              <div style={{ marginTop: 8 }}>
                Orbit the cloud and decide. <b>Approve</b> makes this the cloud; <b>Undo</b> restores the previous one.
              </div>
            </div>
          </div>
        </div>
        <div className="cd-actions">
          <button className="cd-btn cd-btn-cancel" disabled={busy} onClick={onUndo}>↩ Undo</button>
          <button className="cd-btn cd-btn-confirm" disabled={busy} onClick={onApprove} autoFocus>✓ Approve</button>
        </div>
      </div>
    </div>
  )
}
