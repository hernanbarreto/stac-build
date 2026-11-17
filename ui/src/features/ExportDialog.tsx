/**
 * ExportDialog — a scan's products as files other programs open, in one ZIP the user downloads
 * (USER 2026-11-17: "podriamos exportar la nube de puntos compatible con varios programas, la malla glb con la
 * textura, y el IFC … y que la persona lo pueda descargar en su maquina"). What to export is chosen by ticks; the
 * cloud can be filtered by the viewer's confidence, as GenRecon's. The export runs on the server (its state is
 * polled); the files hold the geometry AS IT IS ON DISK — the README in the ZIP says the epoch and the calibration.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, PackageOpen, Play } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Banner } from '../components/ui/Banner'
import { Checkbox, Field } from '../components/ui/Field'
import { Progress } from '../components/ui/Progress'
import { Stack } from '../components/ui/Panel'
import { useFmt, useI18n, useT } from '../i18n'
import { registerCommands, type Executors, type ExportCommandId, type Outcome } from '../commands/registry'

export const EXPORT_ITEMS = ['e57', 'las', 'ply', 'glb', 'ifc', 'csv'] as const
export type ExportItem = typeof EXPORT_ITEMS[number]
const CLOUD: ExportItem[] = ['e57', 'las', 'ply']

interface PlanItem { available: boolean; bytes?: number; why?: string; elements?: number; objects?: number; meshes?: string[] }
interface ExportRun { status: 'idle' | 'running' | 'done' | 'failed'; progress?: number; message?: string; error?: string; report?: { file: string; bytes: number; points?: number } }
interface ExportPlan {
  points: number
  items: Record<ExportItem, PlanItem>
  state: { geometry_epoch: number; calibration: { state?: string; scale_factor?: number | null; mode?: string } }
  previous: { file: string; bytes: number; modified: string } | null
  run: ExportRun
}

interface ExportDialogProps {
  open: boolean
  sessionId: string | null
  scanKey: string | null
  /** the viewer's confidence slider for this session (0..1) */
  defaultConfidence: number
  onClose: () => void
}

export function ExportDialog({ open, sessionId, scanKey, defaultConfidence, onClose }: ExportDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const { lang } = useI18n()
  const [plan, setPlan] = useState<ExportPlan | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [chosen, setChosen] = useState<Set<ExportItem>>(new Set())
  const [pct, setPct] = useState(0)
  const [run, setRun] = useState<ExportRun>({ status: 'idle' })

  const query = scanKey ? `?scan_key=${encodeURIComponent(scanKey)}` : ''
  const base = sessionId ? `/api/export/${encodeURIComponent(sessionId)}` : ''

  const load = useCallback(async () => {
    if (!base) return
    try {
      const r = await fetch(`${base}/plan${query}`)
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      setPlan(d as ExportPlan); setRun((d as ExportPlan).run); setError(null)
      return d as ExportPlan
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); return null }
  }, [base, query])

  // opened: what this scan can export; everything available starts ticked
  useEffect(() => {
    if (!open) return
    setPlan(null); setError(null); setRun({ status: 'idle' })
    setPct(Math.round((defaultConfidence || 0) * 100))
    void load().then(p => { if (p) setChosen(new Set(EXPORT_ITEMS.filter(k => p.items[k].available))) })
  }, [open, load, defaultConfidence])

  // while it runs on the server, its state is polled; when it ends the plan is read again (the new ZIP)
  useEffect(() => {
    if (!open || run.status !== 'running' || !base) return
    const iv = setInterval(async () => {
      try {
        const d = await (await fetch(`${base}/status${query}`)).json()
        setRun(d as ExportRun)
        if (d.status !== 'running') void load()
      } catch { /* the next poll tells */ }
    }, 1500)
    return () => clearInterval(iv)
  }, [open, run.status, base, query, load])

  const start = useCallback(async (items: ExportItem[], confidence: number): Promise<true | string> => {
    if (!base) return 'no session'
    try {
      const r = await fetch(base, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ scan_key: scanKey ?? undefined, items, confidence, lang }) })
      const d = await r.json()
      if (!r.ok) { const why = String(d?.detail?.error || d?.detail || `HTTP ${r.status}`); setRun({ status: 'failed', error: why }); return why }
      setRun(d as ExportRun)
      return true
    } catch (e) { const why = e instanceof Error ? e.message : String(e); setRun({ status: 'failed', error: why }); return why }
  }, [base, scanKey, lang])

  const items = EXPORT_ITEMS.filter(k => chosen.has(k))
  const running = run.status === 'running'
  const total = plan ? items.reduce((s, k) => s + (plan.items[k].bytes ?? 0), 0) : 0
  const cloudChosen = items.some(k => CLOUD.includes(k))
  const cal = plan?.state.calibration
  const scaleNote = !cal ? '' : cal.mode === 'axes' || cal.scale_factor == null || cal.state === 'uncalibrated' ? t('export.scaleUncalibrated')
    : cal.scale_factor === 1 ? t('export.scaleApplied') : t('export.scalePending', { factor: fmt.number(cal.scale_factor, 5) })
  const download = base ? `${base}/download${query}` : '#'

  // STAC's commands of this window (server/commands/catalog/290_export.yaml), while it is open
  const xcmd = useRef({ plan, running, start })
  xcmd.current = { plan, running, start }
  useEffect(() => {
    if (!open) return
    const executors: Executors<ExportCommandId> = {
      'export.run': async ({ items: asked, confidence }) => {
        const c = xcmd.current
        if (!c.plan) return { ok: false, reason: 'unavailable' }
        if (c.running) return { ok: false, reason: 'unavailable' }
        const want = (asked && asked.length ? EXPORT_ITEMS.filter(k => asked.includes(k)) : EXPORT_ITEMS.filter(k => c.plan!.items[k].available))
        const missing = want.filter(k => !c.plan!.items[k].available)
        if (missing.length) return { ok: false, reason: 'failed', detail: missing.map(k => `${k}: ${c.plan!.items[k].why ?? ''}`).join('; ') } as Outcome
        if (!want.length) return { ok: false, reason: 'nothing_to_do' }
        setChosen(new Set(want)); if (confidence !== undefined) setPct(Math.round(confidence * 100))
        const r = await c.start(want, confidence ?? 0)
        return r === true ? { ok: true, detail: want.join(', ') } : { ok: false, reason: 'failed', detail: r }
      },
    }
    return registerCommands(executors)
  }, [open])

  return (
    <Dialog open={open} title={t('export.title')} subtitle={[sessionId, scanKey].filter(Boolean).join(' · ') || undefined}
      icon={<PackageOpen aria-hidden />} size="md" onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>{t('common.close')}</Button>
          <Button variant="primary" icon={<Play aria-hidden />} disabled={running || !plan || items.length === 0}
            onClick={() => { void start(items, Math.min(99, Math.max(0, pct)) / 100) }} data-autofocus>{t('export.run')}</Button>
        </>
      }>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t('export.description')}</p>
        {error && <Banner tone="err" compact>{error}</Banner>}
        {plan && (
          <>
            <Stack gap={1}>
              {EXPORT_ITEMS.map(k => {
                const it = plan.items[k]
                const meta = !it.available ? (it.why ?? '')
                  : k === 'ifc' ? t('export.elements', { n: fmt.integer(it.elements ?? 0) })
                  : k === 'csv' ? t('export.objects', { n: fmt.integer(it.objects ?? 0) })
                  : t('export.upTo', { size: fmt.bytesMb((it.bytes ?? 0) / 1e6) })
                return (
                  <Checkbox key={k} checked={chosen.has(k)} disabled={!it.available || running}
                    onChange={on => setChosen(prev => { const next = new Set(prev); if (on) next.add(k); else next.delete(k); return next })}
                    label={`${t(`export.item_${k}`)} — ${meta}`} />
                )
              })}
            </Stack>
            {cloudChosen && (
              <Field label={t('export.confidence', { pct })} hint={t('export.confidenceHint', { points: fmt.integer(plan.points) })}>
                <div className="stac-autoseg__row">
                  <input type="range" min={0} max={99} step={1} value={pct} disabled={running} onChange={e => setPct(Number(e.target.value))} />
                  <input className="stac-input__el stac-mono" type="number" min={0} max={99} step={1} value={pct} disabled={running} style={{ width: '5em' }}
                    onChange={e => setPct(Math.min(99, Math.max(0, Math.round(Number(e.target.value) || 0))))} />
                  <span className="stac-muted">%</span>
                </div>
              </Field>
            )}
            <p className="stac-muted">{t('export.asOnDisk', { epoch: plan.state.geometry_epoch })} {scaleNote}</p>
            {items.length > 0 && <p className="stac-muted">{t('export.totalUpTo', { size: fmt.bytesMb(total / 1e6) })}</p>}
          </>
        )}
        {running && <Progress value={run.progress ?? 0} label={run.message ?? ''} />}
        {run.status === 'failed' && <Banner tone="err" compact>{t('export.failed', { detail: run.error ?? '' })}</Banner>}
        {!running && plan?.previous && (
          <Banner tone={run.status === 'done' ? 'ok' : 'info'} compact
            action={<a className="stac-btn stac-btn--primary stac-btn--sm" href={download} download={plan.previous.file}>
              <span className="stac-btn__icon" aria-hidden><Download aria-hidden /></span><span className="stac-btn__label">{t('export.download')}</span></a>}>
            {t(run.status === 'done' ? 'export.ready' : 'export.previous', { file: plan.previous.file, size: fmt.bytesMb(plan.previous.bytes / 1e6), when: plan.previous.modified })}
          </Banner>
        )}
      </Stack>
    </Dialog>
  )
}
