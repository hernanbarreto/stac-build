/**
 * SessionDownloadDialog — ONE window for what leaves a session towards the user's machine, opened by the Download
 * button of the session's row (USER 2026-11-19: "descargar la sesion el boton que esta al lado de la sesion, debe
 * desplegar una opcion con todo lo que tiene la sesion y tildar que si y que no, y lo baja en un zip. export no
 * deberia existir, porque el boton download ya hace el export" / "uno que sea … download session, asi completa como
 * esta, y eventualmente seleccionar cual de los escaneos quiere agregar o excluir, y otro apartado dentro del mismo
 * modal con export, que genera otro zip pero no a nivel sesion, con todos los datos como para volver a subir sino
 * que baja en un zip las nubes que se seleccionen, los mesh, los mesh convertidos a IFC, los IFC que hay cargados").
 * Its two sections:
 *   · DOWNLOAD SESSION — the session as it is, one ZIP that uploads again as a session; its scans are ticked (all
 *     of them at first) and one left out is not in the ZIP. The browser's own download (it can be resumed).
 *   · EXPORT — files other programs open, one ZIP built on the server and downloaded when it is ready: of each scan
 *     the cloud (E57 / LAS / PLY), its meshes (GLB), the mesh as IFC, the objects table (CSV); and the IFCs loaded
 *     in the project. Nothing is ticked at first. The files hold the geometry AS IT IS ON DISK — each scan's README
 *     says its epoch and calibration.
 * There is no other Export in the interface: STAC's export commands open this window.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { Download, Play } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Banner } from '../components/ui/Banner'
import { Checkbox, Field, SegmentedControl } from '../components/ui/Field'
import { Progress } from '../components/ui/Progress'
import { Stack } from '../components/ui/Panel'
import { useFmt, useI18n, useT } from '../i18n'
import { registerCommands, type Executors, type ExportCommandId, type Outcome } from '../commands/registry'

export const EXPORT_ITEMS = ['e57', 'las', 'ply', 'glb', 'ifc', 'csv'] as const
export type ExportItem = typeof EXPORT_ITEMS[number]
export type DownloadSection = 'session' | 'export'
const CLOUD: ExportItem[] = ['e57', 'las', 'ply']

interface Share { files: number; bytes: number }
/** what the session's ZIP holds: each scan's share and the project's own files (always in it) */
interface ArchivePlan { files: number; bytes: number; scans: string[]; by_scan: Record<string, Share>; common: Share }
interface PlanItem { available: boolean; bytes?: number; why?: string; elements?: number; objects?: number; meshes?: string[] }
interface ScanState { geometry_epoch: number; calibration: { state?: string; scale_factor?: number | null; mode?: string } }
interface ScanPlan { points: number; items: Record<ExportItem, PlanItem>; state: ScanState | null }
interface ExportRun { status: 'idle' | 'running' | 'done' | 'failed'; progress?: number; message?: string; error?: string }
interface ExportPlan {
  scans: Record<string, ScanPlan>
  loaded_ifcs: { file: string; bytes: number }[]
  previous: { file: string; bytes: number; modified: string } | null
  run: ExportRun
}

interface SessionDownloadDialogProps {
  open: boolean
  sessionId: string | null
  /** the section shown when it opens */
  initialSection: DownloadSection
  /** the session's download is a manager's or an administrator's; anyone else has the export alone */
  canManage: boolean
  token: string | null
  /** the scan on screen when this session is the open one: the one STAC's "export the scan" means */
  activeScanKey: string | null
  /** the viewer's confidence slider (0..1) */
  defaultConfidence: number
  /** start the browser's download of the session with these scans (none given: all) — true, or why not */
  onDownloadSession: (id: string, scans?: string[]) => Promise<true | string>
  onClose: () => void
}

const reason = (d: any, status: number): string =>
  String(d?.detail?.error || (typeof d?.detail === 'string' ? d.detail : '') || `HTTP ${status}`)
const scanName = (key: string) => key.replace(/\/default$/, '')

export function SessionDownloadDialog({ open, sessionId, initialSection, canManage, token, activeScanKey, defaultConfidence, onDownloadSession, onClose }: SessionDownloadDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const { lang } = useI18n()
  // (the window is mounted again at every opening — its key, in App — so each one starts from here)
  const [section, setSection] = useState<DownloadSection>(canManage ? initialSection : 'export')
  // ── the session ──
  const [archive, setArchive] = useState<ArchivePlan | null>(null)
  const [archiveError, setArchiveError] = useState<string | null>(null)
  const [scans, setScans] = useState<Set<string>>(new Set())
  const [starting, setStarting] = useState(false)
  const [started, setStarted] = useState(false)
  // ── the export ──
  const [plan, setPlan] = useState<ExportPlan | null>(null)
  const [planError, setPlanError] = useState<string | null>(null)
  const [chosen, setChosen] = useState<Set<string>>(new Set())           // "<scan key>|<item>"
  const [withIfcs, setWithIfcs] = useState(false)
  const [pct, setPct] = useState(Math.round((defaultConfidence || 0) * 100))
  const [run, setRun] = useState<ExportRun>({ status: 'idle' })
  // an export launched from THIS window: when it ends its ZIP is downloaded at once (one action, not Export and
  // then Download)
  const launched = useRef(false)

  const size = (bytes: number) => (bytes >= 1e9 ? `${fmt.number(bytes / 1e9, bytes >= 1e10 ? 0 : 1)} GB` : fmt.bytesMb(bytes / 1e6))
  const base = sessionId ? `/api/export/${encodeURIComponent(sessionId)}` : ''

  const loadArchive = useCallback(async () => {
    if (!sessionId) return
    try {
      const r = await fetch(`/api/session_archive/${encodeURIComponent(sessionId)}/plan`, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(reason(d, r.status))
      setArchive(d as ArchivePlan); setScans(new Set((d as ArchivePlan).scans)); setArchiveError(null)
    } catch (e) { setArchiveError(e instanceof Error ? e.message : String(e)) }
  }, [sessionId, token])

  const loadPlan = useCallback(async (): Promise<ExportPlan | null> => {
    if (!base) return null
    try {
      const r = await fetch(`${base}/session_plan`)
      const d = await r.json().catch(() => ({}))
      if (!r.ok) throw new Error(reason(d, r.status))
      setPlan(d as ExportPlan); setRun((d as ExportPlan).run); setPlanError(null)
      return d as ExportPlan
    } catch (e) { setPlanError(e instanceof Error ? e.message : String(e)); return null }
  }, [base])

  // each section's data is read ONCE, when the section is first shown (the session's listing takes seconds on a
  // large one: an export does not wait for it)
  const archiveAsked = useRef(false)
  const planAsked = useRef<Promise<ExportPlan | null> | null>(null)
  const ensurePlan = useCallback(() => {
    if (!planAsked.current) planAsked.current = loadPlan()
    return planAsked.current
  }, [loadPlan])
  useEffect(() => {
    if (!open) return
    if (section === 'session' && !archiveAsked.current) { archiveAsked.current = true; void loadArchive() }
    if (section === 'export') void ensurePlan()
  }, [open, section, loadArchive, ensurePlan])

  // while the export runs on the server its state is polled; when it ends the plan is read again (the new ZIP)
  useEffect(() => {
    if (!open || run.status !== 'running' || !base) return
    const iv = setInterval(async () => {
      try {
        const d = await (await fetch(`${base}/session_status`)).json()
        setRun(d as ExportRun)
        if (d.status !== 'running') {
          const p = await loadPlan()
          if (d.status === 'done' && launched.current && p?.previous) {
            const a = document.createElement('a')
            a.href = `${base}/session_download`
            a.download = p.previous.file
            document.body.appendChild(a); a.click(); a.remove()
          }
          launched.current = false
        }
      } catch { /* the next poll tells */ }
    }, 1500)
    return () => clearInterval(iv)
  }, [open, run.status, base, loadPlan])

  const start = useCallback(async (selection: Record<string, ExportItem[]>, loadedIfcs: boolean, confidence: number): Promise<true | string> => {
    if (!base) return 'no session'
    try {
      const r = await fetch(`${base}/session`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                 body: JSON.stringify({ scans: selection, loaded_ifcs: loadedIfcs, confidence, lang }) })
      const d = await r.json().catch(() => ({}))
      if (!r.ok) { const why = reason(d, r.status); setRun({ status: 'failed', error: why }); return why }
      setRun(d as ExportRun)
      launched.current = true
      return true
    } catch (e) { const why = e instanceof Error ? e.message : String(e); setRun({ status: 'failed', error: why }); return why }
  }, [base, lang])

  // ── the session: what is ticked ──
  const kept = archive ? archive.scans.filter(k => scans.has(k)) : []
  const sessionBytes = archive ? archive.common.bytes + kept.reduce((s, k) => s + archive.by_scan[k].bytes, 0) : 0
  const sessionFiles = archive ? archive.common.files + kept.reduce((s, k) => s + archive.by_scan[k].files, 0) : 0
  const canDownload = !!archive && (archive.scans.length === 0 || kept.length > 0)
  const downloadSession = async () => {
    if (!sessionId || !archive || !canDownload) return
    setStarting(true); setStarted(false); setArchiveError(null)
    const r = await onDownloadSession(sessionId, kept.length === archive.scans.length ? undefined : kept)
    setStarting(false)
    if (r === true) setStarted(true); else setArchiveError(r)
  }

  // ── the export: what is ticked ──
  const running = run.status === 'running'
  const selection: Record<string, ExportItem[]> = {}
  if (plan) {
    for (const key of Object.keys(plan.scans)) {
      const items = EXPORT_ITEMS.filter(k => chosen.has(`${key}|${k}`) && plan.scans[key].items[k].available)
      if (items.length) selection[key] = items
    }
  }
  const loadedOn = withIfcs && !!plan?.loaded_ifcs.length
  const nothing = Object.keys(selection).length === 0 && !loadedOn
  const cloudChosen = Object.values(selection).some(items => items.some(k => CLOUD.includes(k)))
  const loadedBytes = plan ? plan.loaded_ifcs.reduce((s, f) => s + f.bytes, 0) : 0
  const exportBytes = (plan ? Object.entries(selection).reduce((s, [key, items]) => s + items.reduce((x, k) => x + (plan.scans[key].items[k].bytes ?? 0), 0), 0) : 0)
    + (loadedOn ? loadedBytes : 0)
  const scaleNote = (st: ScanState) => {
    const cal = st.calibration
    return cal.mode === 'axes' || cal.scale_factor == null || cal.state === 'uncalibrated' ? t('export.scaleUncalibrated')
      : cal.scale_factor === 1 ? t('export.scaleApplied') : t('export.scalePending', { factor: fmt.number(cal.scale_factor, 5) })
  }
  const tick = (id: string, on: boolean) => setChosen(prev => { const next = new Set(prev); if (on) next.add(id); else next.delete(id); return next })

  // STAC's command of this window (server/commands/catalog/290_export.yaml), while it is open: "export the scan"
  // is the scan on screen (the session's only scan when it has one)
  const xcmd = useRef({ plan, running, start, ensurePlan, activeScanKey })
  xcmd.current = { plan, running, start, ensurePlan, activeScanKey }
  useEffect(() => {
    if (!open) return
    const executors: Executors<ExportCommandId> = {
      'export.run': async ({ items: asked, confidence }) => {
        const c = xcmd.current
        if (c.running) return { ok: false, reason: 'unavailable' }
        setSection('export')
        const p = c.plan ?? await c.ensurePlan()
        if (!p) return { ok: false, reason: 'unavailable' }
        const keys = Object.keys(p.scans)
        const scan = c.activeScanKey && p.scans[c.activeScanKey] ? c.activeScanKey : keys.length === 1 ? keys[0] : null
        if (!scan) return { ok: false, reason: 'unavailable' }
        const its = p.scans[scan].items
        const want = asked && asked.length ? EXPORT_ITEMS.filter(k => asked.includes(k)) : EXPORT_ITEMS.filter(k => its[k].available)
        const missing = want.filter(k => !its[k].available)
        if (missing.length) return { ok: false, reason: 'failed', detail: missing.map(k => `${k}: ${its[k].why ?? ''}`).join('; ') } as Outcome
        if (!want.length) return { ok: false, reason: 'nothing_to_do' }
        setChosen(new Set(want.map(k => `${scan}|${k}`))); setWithIfcs(false)
        if (confidence !== undefined) setPct(Math.round(confidence * 100))
        const r = await c.start({ [scan]: want }, false, confidence ?? 0)
        return r === true ? { ok: true, detail: want.join(', ') } : { ok: false, reason: 'failed', detail: r }
      },
    }
    return registerCommands(executors)
  }, [open])

  return (
    <Dialog open={open} title={t('download.title')} subtitle={sessionId ?? undefined} icon={<Download aria-hidden />} size="lg" onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>{t('common.close')}</Button>
          {section === 'session'
            ? <Button variant="primary" icon={<Download aria-hidden />} loading={starting} disabled={!canDownload || starting}
                onClick={() => { void downloadSession() }}>{t('download.run')}</Button>
            : <Button variant="primary" icon={<Play aria-hidden />} disabled={running || !plan || nothing}
                onClick={() => { void start(selection, loadedOn, Math.min(99, Math.max(0, pct)) / 100) }}>{t('export.run')}</Button>}
        </>
      }>
      <Stack gap={3}>
        {canManage && (
          <SegmentedControl<DownloadSection> value={section} onChange={setSection} ariaLabel={t('download.title')}
            options={[{ value: 'session', label: t('download.sectionSession') }, { value: 'export', label: t('download.sectionExport') }]} />
        )}

        {section === 'session' && (
          <>
            <p className="stac-dialog__message">{t('download.sessionDescription')}</p>
            {archiveError && <Banner tone="err" compact>{archiveError}</Banner>}
            {!archive && !archiveError && <p className="stac-muted">{t('download.listing')}</p>}
            {archive && (
              <>
                <Stack gap={1}>
                  {archive.scans.map(k => (
                    <Checkbox key={k} checked={scans.has(k)} disabled={starting}
                      onChange={on => setScans(prev => { const next = new Set(prev); if (on) next.add(k); else next.delete(k); return next })}
                      label={t('download.scanLine', { scan: scanName(k), size: size(archive.by_scan[k].bytes), files: fmt.integer(archive.by_scan[k].files) })} />
                  ))}
                </Stack>
                <p className="stac-muted">{t('download.common', { size: size(archive.common.bytes), files: fmt.integer(archive.common.files) })}</p>
                {canDownload
                  ? <p className="stac-muted">{t('download.total', { size: size(sessionBytes), files: fmt.integer(sessionFiles) })}</p>
                  : <Banner tone="warn" compact>{t('download.needScan')}</Banner>}
              </>
            )}
            {started && <Banner tone="ok" compact>{t('download.started')}</Banner>}
          </>
        )}

        {section === 'export' && (
          <>
            <p className="stac-dialog__message">{t('export.description')}</p>
            {planError && <Banner tone="err" compact>{planError}</Banner>}
            {!plan && !planError && <p className="stac-muted">{t('download.reading')}</p>}
            {plan && (
              <>
                {Object.entries(plan.scans).map(([key, sp]) => (
                  <div key={key} className="stac-dl__scan">
                    <div className="stac-dl__scanhead">
                      <strong>{scanName(key)}</strong>
                      <span className="stac-muted">{t('export.scanPoints', { points: fmt.integer(sp.points) })}</span>
                    </div>
                    <div className="stac-dl__items">
                      {EXPORT_ITEMS.map(k => {
                        const it = sp.items[k]
                        const meta = !it.available ? (it.why ?? '')
                          : k === 'ifc' ? t('export.elements', { n: fmt.integer(it.elements ?? 0) })
                          : k === 'csv' ? t('export.objects', { n: fmt.integer(it.objects ?? 0) })
                          : k === 'glb' ? size(it.bytes ?? 0)
                          : t('export.upTo', { size: size(it.bytes ?? 0) })
                        return (
                          <Checkbox key={k} checked={it.available && chosen.has(`${key}|${k}`)} disabled={!it.available || running}
                            onChange={on => tick(`${key}|${k}`, on)} label={t(`export.item_${k}`)} description={meta} />
                        )
                      })}
                    </div>
                    {sp.state && <p className="stac-muted">{t('export.asOnDisk', { epoch: sp.state.geometry_epoch })} {scaleNote(sp.state)}</p>}
                  </div>
                ))}
                {plan.loaded_ifcs.length > 0 && (
                  <div className="stac-dl__scan">
                    <Checkbox checked={withIfcs} disabled={running} onChange={setWithIfcs}
                      label={t('export.loadedIfcs', { n: plan.loaded_ifcs.length, size: size(loadedBytes) })}
                      description={plan.loaded_ifcs.map(f => f.file).join(', ')} />
                  </div>
                )}
                {cloudChosen && (
                  <Field label={t('export.confidence', { pct })} hint={t('export.confidenceHint')}>
                    <div className="stac-autoseg__row">
                      <input type="range" min={0} max={99} step={1} value={pct} disabled={running} onChange={e => setPct(Number(e.target.value))} />
                      <input className="stac-input__el stac-mono" type="number" min={0} max={99} step={1} value={pct} disabled={running} style={{ width: '5em' }}
                        onChange={e => setPct(Math.min(99, Math.max(0, Math.round(Number(e.target.value) || 0))))} />
                      <span className="stac-muted">%</span>
                    </div>
                  </Field>
                )}
                {!nothing && exportBytes > 0 && <p className="stac-muted">{t('export.totalUpTo', { size: size(exportBytes) })}</p>}
              </>
            )}
            {running && <Progress value={run.progress ?? 0} label={run.message ?? ''} />}
            {run.status === 'failed' && <Banner tone="err" compact>{t('export.failed', { detail: run.error ?? '' })}</Banner>}
            {!running && plan?.previous && (
              <Banner tone={run.status === 'done' ? 'ok' : 'info'} compact
                action={<a className="stac-btn stac-btn--sm" href={`${base}/session_download`} download={plan.previous.file}>
                  <span className="stac-btn__icon" aria-hidden><Download aria-hidden /></span><span className="stac-btn__label">{t('export.downloadAgain')}</span></a>}>
                {t(run.status === 'done' ? 'export.ready' : 'export.previous', { file: plan.previous.file, size: size(plan.previous.bytes), when: plan.previous.modified })}
              </Banner>
            )}
          </>
        )}
      </Stack>
    </Dialog>
  )
}
