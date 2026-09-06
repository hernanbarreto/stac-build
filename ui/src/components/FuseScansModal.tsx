import { useEffect, useState } from 'react'

/**
 * Fuse scans (USER DESIGN 2026-09-06).
 *
 * 1. Pick the scans to fuse into the project's composition reference.
 * 2. The backend finds, per scan, the segments that exist with the SAME
 *    label in the reference (the user chose invariant objects on purpose);
 *    at least 2 pairs are required per scan. Pairs can be excluded.
 * 3. Register (CloudComPy: RansacSD planes + ICP with scale, symmetric
 *    scale split, guards) → build the merged cloud → it appears in the
 *    scans list as a "fused" entry and opens as a tab.
 */

type Scan = {
  key: string; date: string; label: string; kind?: string
  is_reference?: boolean; points?: number; has_potree?: boolean
}
type Pair = { label: string; ref_points: number; scan_points: number; include: boolean }
type PairsResult = Record<string, { pairs: Pair[]; only_in_scan: string[]; only_in_ref: string[] }>

export default function FuseScansModal({ project, scans, onClose, onStatus, onFused }: {
  project: string
  scans: Scan[]
  onClose: () => void
  onStatus: (m: string) => void
  onFused: () => void
}) {
  const reference = scans.find(s => s.is_reference)
  const candidates = scans.filter(s => !s.is_reference && s.kind !== 'fused' && s.has_potree)
  const [selected, setSelected] = useState<Set<string>>(new Set(candidates.map(s => s.key)))
  const [pairs, setPairs] = useState<PairsResult>({})
  const [loadingPairs, setLoadingPairs] = useState(false)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<{ pct: number; detail: string } | null>(null)
  const [report, setReport] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  // pairs by label for the selected scans
  useEffect(() => {
    if (!reference || selected.size === 0) { setPairs({}); return }
    let cancelled = false
    setLoadingPairs(true)
    fetch(`/api/project/${project}/fuse/pairs`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scans: Array.from(selected) }),
    }).then(r => r.json()).then(d => {
      if (cancelled) return
      const out: PairsResult = {}
      for (const [k, v] of Object.entries<any>(d.scans || {})) {
        out[k] = {
          pairs: (v.pairs || []).map((p: any) => ({ ...p, include: true })),
          only_in_scan: v.only_in_scan || [], only_in_ref: v.only_in_ref || [],
        }
      }
      setPairs(out)
    }).catch(() => { if (!cancelled) setError('could not fetch segment pairs') })
      .finally(() => { if (!cancelled) setLoadingPairs(false) })
    return () => { cancelled = true }
  }, [project, selected, reference])

  // progress of the fuse task
  useEffect(() => {
    if (!running) return
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`/api/tasks/${project}`).then(r => r.json())
        const t = (d.tasks || []).find((x: any) => x.task_type === 'fuse')
        if (t) setProgress({ pct: t.pct || 0, detail: t.detail || '' })
      } catch { /* keep last */ }
    }, 2000)
    return () => clearInterval(iv)
  }, [running, project])

  const canRun = reference && selected.size > 0 && !loadingPairs && !running &&
    Array.from(selected).every(k => (pairs[k]?.pairs || []).filter(p => p.include).length >= 2)

  const run = async () => {
    setRunning(true); setError(null); setReport(null); setProgress({ pct: 0, detail: 'starting…' })
    onStatus('⛶ fusing scans (register + merged cloud) — several minutes…')
    try {
      const exclude: Record<string, string[]> = {}
      for (const k of selected) exclude[k] = (pairs[k]?.pairs || []).filter(p => !p.include).map(p => p.label)
      const r = await fetch(`/api/project/${project}/fuse/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scans: Array.from(selected), exclude }),
      })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(d.detail || 'fuse failed')
      setReport(d.report || d)
      onStatus(d.report?.accepted === false ? '⛶ registration REJECTED — see the report' : '⛶ merged cloud built')
      if (d.report?.accepted !== false) onFused()
    } catch (e: any) {
      setError(e?.message || 'fuse failed')
      onStatus(`⛶ fuse failed: ${e?.message || 'error'}`)
    }
    setRunning(false)
  }

  return (
    <div className="admin-overlay" style={{ zIndex: 2100 }}>
      <div className="admin-panel" style={{ maxWidth: 640, maxHeight: '85vh', overflow: 'auto' }}>
        <div className="admin-header">
          <h2>⛶ Fuse scans</h2>
          <button className="admin-close" onClick={onClose} disabled={running}>✕</button>
        </div>
        <div style={{ padding: 16, fontSize: 13, color: 'var(--text-primary)' }}>
          {!reference ? (
            <p style={{ color: '#e0a632' }}>No composition reference set — mark one scan with ★ in the scans list first.</p>
          ) : (
            <p style={{ color: 'var(--text-secondary)', marginBottom: 10 }}>
              Reference: <strong>{reference.date} {reference.label}</strong>. Each selected scan is
              registered onto it using the segments that carry the <em>same label</em> in both
              (you chose them as invariant objects). At least <strong>2 pairs</strong> per scan.
            </p>
          )}

          {candidates.length === 0 && (
            <p style={{ color: 'var(--text-secondary)' }}>No other scan with a Potree to fuse.</p>
          )}
          {candidates.map(sc => {
            const on = selected.has(sc.key)
            const pr = pairs[sc.key]
            const nInc = (pr?.pairs || []).filter(p => p.include).length
            return (
              <div key={sc.key} style={{ marginBottom: 10, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 8, opacity: on ? 1 : 0.6 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input type="checkbox" checked={on} disabled={running}
                    onChange={() => setSelected(prev => { const n = new Set(prev); n.has(sc.key) ? n.delete(sc.key) : n.add(sc.key); return n })} />
                  <strong>{sc.date} {sc.label}</strong>
                  <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text-secondary)' }}>
                    {sc.points ? `${(sc.points / 1e6).toFixed(1)}M pts` : ''}
                  </span>
                </label>
                {on && (
                  <div style={{ marginTop: 6, paddingLeft: 24 }}>
                    {loadingPairs && !pr && <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>looking for shared segments…</div>}
                    {pr && pr.pairs.length === 0 && (
                      <div style={{ fontSize: 12, color: '#e0a632' }}>no segment shares a label with the reference — segment ≥2 invariant objects with the same names in both scans</div>
                    )}
                    {pr && pr.pairs.map(p => (
                      <label key={p.label} style={{ display: 'flex', gap: 8, fontSize: 12, alignItems: 'center' }}>
                        <input type="checkbox" checked={p.include} disabled={running}
                          onChange={() => setPairs(prev => ({
                            ...prev,
                            [sc.key]: { ...prev[sc.key], pairs: prev[sc.key].pairs.map(q => q.label === p.label ? { ...q, include: !q.include } : q) },
                          }))} />
                        <span style={{ flex: 1 }}>{p.label}</span>
                        <span style={{ color: 'var(--text-secondary)' }}>ref {(p.ref_points / 1e3).toFixed(0)}k · scan {(p.scan_points / 1e3).toFixed(0)}k</span>
                      </label>
                    ))}
                    {pr && pr.pairs.length > 0 && nInc < 2 && (
                      <div style={{ fontSize: 12, color: '#e0a632' }}>at least 2 pairs are required ({nInc} selected)</div>
                    )}
                    {pr && (pr.only_in_scan.length > 0 || pr.only_in_ref.length > 0) && (
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                        unpaired — scan: {pr.only_in_scan.join(', ') || '—'} · reference: {pr.only_in_ref.join(', ') || '—'}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}

          {running && progress && (
            <div style={{ margin: '10px 0' }}>
              <div style={{ fontSize: 12, marginBottom: 4 }}>⏳ {progress.detail} — {progress.pct}%</div>
              <div style={{ height: 8, background: 'var(--bg-tertiary)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${progress.pct}%`, background: '#e0a632', transition: 'width .4s' }} />
              </div>
            </div>
          )}
          {error && <p style={{ color: '#e05a5a', fontSize: 12 }}>{error}</p>}
          {report && (
            <div style={{ marginTop: 10, padding: 10, background: 'var(--bg-tertiary)', borderRadius: 8, fontSize: 12 }}>
              <strong>Report</strong> — {report.accepted === false ? <span style={{ color: '#e05a5a' }}>REJECTED: {report.reason}</span> : <span style={{ color: '#5ac05a' }}>accepted</span>}
              {(report.scans || []).map((s: any) => (
                <div key={s.key} style={{ marginTop: 6 }}>
                  <div><strong>{s.key}</strong> — scale {s.scale?.toFixed?.(4)} (split ±√), rot {s.rot_deg?.toFixed?.(2)}°, |t| {s.t_m?.toFixed?.(3)} m, rms {s.rms_cm?.toFixed?.(1)} cm — {s.verdict}</div>
                  {(s.pairs || []).map((p: any) => (
                    <div key={p.label} style={{ paddingLeft: 12, color: p.suspect ? '#e0a632' : 'var(--text-secondary)' }}>
                      {p.label}: residual {p.residual_cm_before?.toFixed?.(1)} → {p.residual_cm_after?.toFixed?.(1)} cm · size ratio {p.size_ratio?.toFixed?.(3)}{p.suspect ? ' · inconsistent (excluded from scale)' : ''}
                    </div>
                  ))}
                  {s.heldout && <div style={{ paddingLeft: 12, color: 'var(--text-secondary)' }}>held-out (unused surfaces): {s.heldout.before_cm?.toFixed?.(1)} → {s.heldout.after_cm?.toFixed?.(1)} cm</div>}
                </div>
              ))}
            </div>
          )}

          <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
            <button className="bim-action-btn upload" style={{ flex: 1 }} disabled={!canRun} onClick={run}>
              {running ? '⏳ fusing…' : '⛶ Register & build merged cloud'}
            </button>
            <button className="bim-action-btn" style={{ flex: 1 }} onClick={onClose} disabled={running}>Close</button>
          </div>
        </div>
      </div>
    </div>
  )
}
