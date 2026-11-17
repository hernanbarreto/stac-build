import { useEffect, useRef, useState } from 'react'
import { findByName, registerCommands, type Executors, type FuseCommandId, type Outcome } from '../commands/registry'
import { Layers } from 'lucide-react'
import { Dialog } from './ui/Dialog'
import { Button } from './ui/Button'
import { Checkbox } from './ui/Field'
import { Banner } from './ui/Banner'
import { Badge } from './ui/Badge'
import { Progress } from './ui/Progress'
import { Stack, Row } from './ui/Panel'
import { EmptyState } from './ui/EmptyState'
import { useFmt, useT } from '../i18n'

/**
 * Fuse scans (USER DESIGN 2026-09-06).
 *
 * 1. Pick the scans to fuse into the session's composition reference.
 * 2. The backend finds, per scan, the segments that exist with the SAME
 *    label in the reference (the user chose invariant objects on purpose);
 *    at least 2 pairs are required per scan. Pairs can be excluded.
 * 3. Register (CloudComPy: RansacSD planes + ICP with scale, symmetric
 *    scale split, guards) -> build the merged cloud -> it appears in the
 *    scans list as a "fused" entry and opens as a tab.
 */

type Scan = { key: string; date: string; label: string; kind?: string; is_reference?: boolean; points?: number; has_potree?: boolean }
type Pair = { label: string; ref_points: number; scan_points: number; include: boolean }
type PairsResult = Record<string, { pairs: Pair[]; only_in_scan: string[]; only_in_ref: string[] }>

export default function FuseScansModal({ project, scans, onClose, onStatus, onFused }: {
  project: string
  scans: Scan[]
  onClose: () => void
  onStatus: (m: string) => void
  onFused: () => void
}) {
  const t = useT()
  const fmt = useFmt()
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
        out[k] = { pairs: (v.pairs || []).map((p: any) => ({ ...p, include: true })), only_in_scan: v.only_in_scan || [], only_in_ref: v.only_in_ref || [] }
      }
      setPairs(out)
    }).catch(() => { if (!cancelled) setError(t('fuse.pairsFailed')) })
      .finally(() => { if (!cancelled) setLoadingPairs(false) })
    return () => { cancelled = true }
  }, [project, selected, reference, t])

  // progress of the fuse task
  useEffect(() => {
    if (!running) return
    const iv = setInterval(async () => {
      try {
        const d = await fetch(`/api/tasks/${project}`).then(r => r.json())
        const tk = (d.tasks || []).find((x: any) => x.task_type === 'fuse')
        if (tk) setProgress({ pct: tk.pct || 0, detail: tk.detail || '' })
      } catch { /* keep last */ }
    }, 2000)
    return () => clearInterval(iv)
  }, [running, project])

  // STAC's command of this window (scans.fuse, on a spoken confirmation): the scans named are
  // registered on the reference with at least two object pairs each, as the button requires
  const ucmd = useRef<{ running: boolean; run: (order: { keys: string[]; exclude: Record<string, string[]> }) => unknown }>({ running: false, run: () => undefined })
  useEffect(() => {
    const order = async (a: { session?: string; scans: string[]; exclude?: Record<string, unknown> }): Promise<{ keys: string[]; exclude: Record<string, string[]>; detail: string } | Outcome> => {
      if (a.session && !findByName([{ id: project }], a.session, x => [x.id]).item) return { ok: false, reason: 'not_found', detail: a.session }
      if (!reference || !candidates.length) return { ok: false, reason: 'nothing_to_do' }
      if (ucmd.current.running) return { ok: false, reason: 'unavailable' }
      const names = (sc: Scan) => [sc.key, sc.date, sc.label, `${sc.date} ${sc.label}`]
      const chosen: Scan[] = []
      for (const n of a.scans) {
        const found = findByName(candidates, n, names)
        if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.map(sc => `${sc.date} ${sc.label}`).join(', ') }
        if (!found.item) return { ok: false, reason: 'not_found', detail: n }
        if (!chosen.includes(found.item)) chosen.push(found.item)
      }
      if (!chosen.length) return { ok: false, reason: 'nothing_to_do' }
      const exclude: Record<string, string[]> = {}
      for (const sc of chosen) {
        const given = Object.entries(a.exclude ?? {}).find(([k]) => findByName([sc], k, names).item)?.[1]
        exclude[sc.key] = Array.isArray(given) ? given.map(String) : []
      }
      // the pairs the registration would use, as the window reads them
      try {
        const r = await fetch(`/api/project/${project}/fuse/pairs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scans: chosen.map(sc => sc.key) }) })
        const d = await r.json()
        for (const sc of chosen) {
          const left = ((d.scans || {})[sc.key]?.pairs || []).filter((p: Pair) => !exclude[sc.key].includes(p.label)).length
          if (left < 2) return { ok: false, reason: 'failed', detail: `${sc.date} ${sc.label}: ${left} / 2` }
        }
      } catch (e) { return { ok: false, reason: 'failed', detail: e instanceof Error ? e.message : String(e) } }
      return { keys: chosen.map(sc => sc.key), exclude, detail: `${chosen.map(sc => `${sc.date} ${sc.label}`).join(', ')} → ${reference.date} ${reference.label}` }
    }
    const executors: Executors<FuseCommandId> = {
      'scans.fuse': {
        prepare: async a => { const o = await order(a); return 'ok' in o ? o : { ok: true, detail: o.detail } },
        run: async a => {
          const o = await order(a)
          if ('ok' in o) return o
          setSelected(new Set(o.keys))
          void ucmd.current.run({ keys: o.keys, exclude: o.exclude })
          return { ok: true, detail: o.detail }
        },
      },
    }
    return registerCommands(executors)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, scans])

  const canRun = reference && selected.size > 0 && !loadingPairs && !running &&
    Array.from(selected).every(k => (pairs[k]?.pairs || []).filter(p => p.include).length >= 2)

  ucmd.current.running = running
  const run = async (order?: unknown) => {
    // (a click passes its event; STAC's order carries its own scans and exclusions)
    const asked = order && typeof order === 'object' && 'keys' in order ? order as { keys: string[]; exclude: Record<string, string[]> } : null
    setRunning(true); setError(null); setReport(null); setProgress({ pct: 0, detail: t('fuse.starting') })
    onStatus(t('fuse.statusRunning'))
    try {
      const exclude: Record<string, string[]> = asked ? asked.exclude : {}
      if (!asked) for (const k of selected) exclude[k] = (pairs[k]?.pairs || []).filter(p => !p.include).map(p => p.label)
      const r = await fetch(`/api/project/${project}/fuse/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scans: asked ? asked.keys : Array.from(selected), exclude }),
      })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(d.detail || t('fuse.failed'))
      setReport(d.report || d)
      onStatus(d.report?.accepted === false ? t('fuse.rejectedStatus') : t('fuse.built'))
      if (d.report?.accepted !== false) onFused()
    } catch (e: any) {
      setError(e?.message || t('fuse.failed'))
      onStatus(t('fuse.failedDetail', { detail: e?.message || '' }))
    }
    setRunning(false)
  }
  ucmd.current.run = run

  return (
    <Dialog open title={t('fuse.title')} icon={<Layers aria-hidden />} size="lg" onClose={onClose} busy={running}
      footer={
        <>
          <Button onClick={onClose} disabled={running}>{t('common.close')}</Button>
          <Button variant="primary" icon={<Layers aria-hidden />} disabled={!canRun} loading={running} onClick={run}>{t('fuse.run')}</Button>
        </>
      }>
      <Stack gap={3}>
        {!reference ? (
          <Banner tone="warn">{t('fuse.noReference')}</Banner>
        ) : (
          <p className="stac-dialog__message">{t('fuse.description', { reference: `${reference.date} ${reference.label}` })}</p>
        )}
        {candidates.length === 0 && <EmptyState compact icon={<Layers aria-hidden />} title={t('fuse.noCandidates')} />}
        {candidates.map(sc => {
          const on = selected.has(sc.key)
          const pr = pairs[sc.key]
          const nInc = (pr?.pairs || []).filter(p => p.include).length
          return (
            <div key={sc.key} className={`stac-picklist__item stac-fuse__scan ${on ? 'stac-picklist__item--on' : ''}`.trim()}>
              <Row className="stac-fuse__head">
                <Checkbox checked={on} disabled={running} onChange={() => setSelected(prev => { const n = new Set(prev); if (n.has(sc.key)) n.delete(sc.key); else n.add(sc.key); return n })}
                  label={<strong>{sc.date} {sc.label}</strong>} />
                <span className="stac-picklist__meta stac-mono">{sc.points ? t('status.points', { n: fmt.millions(sc.points) }) : ''}</span>
              </Row>
              {on && (
                <Stack gap={1} className="stac-fuse__pairs">
                  {loadingPairs && !pr && <span className="stac-section__hint">{t('fuse.lookingForPairs')}</span>}
                  {pr && pr.pairs.length === 0 && <Banner tone="warn" compact>{t('fuse.noSharedLabels')}</Banner>}
                  {pr && pr.pairs.map(p => (
                    <Row key={p.label} className="stac-fuse__pair">
                      <Checkbox checked={p.include} disabled={running} label={p.label}
                        onChange={() => setPairs(prev => ({ ...prev, [sc.key]: { ...prev[sc.key], pairs: prev[sc.key].pairs.map(q => q.label === p.label ? { ...q, include: !q.include } : q) } }))} />
                      <span className="stac-picklist__meta stac-mono">{t('fuse.pairPoints', { ref: fmt.integer(p.ref_points), scan: fmt.integer(p.scan_points) })}</span>
                    </Row>
                  ))}
                  {pr && pr.pairs.length > 0 && nInc < 2 && <Banner tone="warn" compact>{t('fuse.needTwoPairs', { n: nInc })}</Banner>}
                  {pr && (pr.only_in_scan.length > 0 || pr.only_in_ref.length > 0) && (
                    <span className="stac-section__hint">{t('fuse.unpaired', { scan: pr.only_in_scan.join(', ') || t('status.dash'), ref: pr.only_in_ref.join(', ') || t('status.dash') })}</span>
                  )}
                </Stack>
              )}
            </div>
          )
        })}
        {running && progress && <Progress value={progress.pct} label={progress.detail} size="md" />}
        {error && <Banner tone="err">{error}</Banner>}
        {report && (
          <Banner tone={report.accepted === false ? 'err' : 'ok'} title={report.accepted === false ? t('fuse.reportRejected', { reason: report.reason }) : t('fuse.reportAccepted')}>
            <Stack gap={2}>
              {(report.scans || []).map((s: any) => (
                <div key={s.key}>
                  <div><strong>{s.key}</strong> — {t('fuse.scanReport', { scale: fmt.number(s.scale ?? 1, 4), rot: fmt.degrees(s.rot_deg ?? 0, 2), t: fmt.number(s.t_m ?? 0, 3), rms: fmt.number(s.rms_cm ?? 0, 1) })} <Badge size="sm">{s.verdict}</Badge></div>
                  {(s.pairs || []).map((p: any) => (
                    <div key={p.label} className={`stac-fuse__pairline ${p.dropped ? 'stac-fuse__pairline--dropped' : p.suspect ? 'stac-fuse__pairline--suspect' : ''}`.trim()}>
                      {t('fuse.pairReport', { label: p.label, before: fmt.number(p.residual_cm_before ?? 0, 1), after: fmt.number(p.residual_cm_after ?? 0, 1), ratio: fmt.number(p.size_ratio ?? 1, 3) })}
                      {p.dropped ? ` — ${t('fuse.pairDropped')}` : p.suspect ? ` — ${t('fuse.pairSuspect')}` : ''}
                    </div>
                  ))}
                  {s.heldout && <div className="stac-fuse__pairline">{t('fuse.heldout', { before: fmt.number(s.heldout.before_cm ?? 0, 1), after: fmt.number(s.heldout.after_cm ?? 0, 1) })}</div>}
                </div>
              ))}
            </Stack>
          </Banner>
        )}
      </Stack>
    </Dialog>
  )
}
