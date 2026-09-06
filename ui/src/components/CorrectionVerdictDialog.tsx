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
  const chunks: any[] = report?.chunks || []
  const solved = chunks.filter(c => c.object_residual_cm)
  const mode = state?.mode || (report?.instance_ids ? 'correction analysis' : 'correction')
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
              <div>{mode}{state?.chunks?.length ? ` · chunk${state.chunks.length > 1 ? 's' : ''} ${state.chunks.join(', ')}` : ''}{state?.points_moved ? ` · ${(state.points_moved / 1e6).toFixed(2)}M points moved` : ''}</div>
              {solved.length > 0 && (
                <table className="cvd-table">
                  <thead><tr><th>chunk</th><th>diagnosis</th><th>rigid</th><th>copies before → after</th><th>floor</th></tr></thead>
                  <tbody>
                    {solved.map(c => (
                      <tr key={c.chunk}>
                        <td>ch{String(c.chunk).padStart(2, '0')}</td>
                        <td>{c.diagnosis}</td>
                        <td>{c.rigid ? `${c.rigid.rot_deg}° / ${c.rigid.t_m} m` : '—'}</td>
                        <td>{c.object_residual_cm.before} → <b>{c.object_residual_cm.after}</b> cm</td>
                        <td>{c.floor_heldout_cm ? `${c.floor_heldout_cm.before} → ${c.floor_heldout_cm.after} cm` : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {chunks.length > solved.length && (
                <div style={{ marginTop: 6, opacity: 0.8 }}>
                  + {chunks.length - solved.length} chunk(s) inherited
                </div>
              )}
              {report?.distribution && (
                <div style={{ marginTop: 6, opacity: 0.85 }}>
                  Loop closure spread over {report.distribution.keyframes_warped} keyframes after kf {report.distribution.identity_until_kf}
                  {' '}· max step between neighbouring keyframes {report.distribution.max_step_between_keyframes_mm} mm (no seam)
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
