/**
 * CalibrationDialog — the scan's calibration measurements, taken ON THE 3D MODEL
 * (USER 2026-11-13: "para calibrar debe hacerse tomando mediciones sobre el modelo 3d no sobre
 * las imagenes porque es dificil, se debe poder seleccionar que una medicion es de calibracion
 * y deben quedar almacenadas").
 *
 * The user measures a distance on the model with the distance tool and presses the pin of its
 * readout: the measurement arrives here as the DRAFT (its two picked points and the model's
 * length), the user states the real length, its role (gauge: calibrates the scale / test:
 * measures the error left), its instrument — or the other scan of the project the length was
 * taken from, to bring this scan to that scan's scale — and it is stored by the server
 * (`/api/calibration/{session}`; the model's length is computed there from the two points).
 *
 * A PLANE is calibrated the same way (USER 2026-11-13: "la calibracion por plano tambien debe
 * quedar sobre la nube … y debe dibujar el plano sobre la nube 3d"): the plane tool marks three or
 * more points on a flat surface of the model, the draft arrives here, the user names it, and the
 * server stores it with the plane it fits — its inclination, how far the points lie from it, its
 * area. Stored planes are drawn on the cloud.
 *
 * The window lists what is stored with the scale factor the gauge measurements give and the
 * error on the test ones. The factor is REPORTED: nothing is rescaled or levelled from here.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Crosshair, Pentagon, Pin, Ruler, Save, Trash2 } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Badge } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { EmptyState } from '../components/ui/EmptyState'
import { Field, Input, NumberInput, SegmentedControl, Select } from '../components/ui/Field'
import { KeyValue, Row, Section, Stack } from '../components/ui/Panel'
import { Table, type Column } from '../components/ui/Table'
import { useToast } from '../components/ui/Toast'
import { useFmt, useT } from '../i18n'

export type CalRole = 'gauge' | 'test'
export type CalTool = 'tape' | 'laser' | 'total_station' | 'other_scan'

export interface CalMeasurement {
  id: string
  a: number[]
  b: number[]
  measured_m: number
  value_m: number
  sigma_m: number
  role: CalRole
  tool: CalTool
  note: string
  geometry_epoch: number
  created: string
  reference_scan?: string
}

interface CalRow {
  id: string
  role: CalRole
  tool: CalTool
  measured_m: number
  value_m: number
  sigma_m: number
  ratio: number
  geometry_epoch: number
  stale: boolean
  calibrated_m: number | null
  residual_m: number | null
  /** the axis it runs along (its largest component): X and Z horizontal, Y vertical */
  axis: 'X' | 'Y' | 'Z'
  /** the direction it was measured in: 0 horizontal, 90 vertical */
  elevation_deg: number
  /** a verifying measurement against the uncertainty the user stated for it (null: none stated) */
  within_sigma: boolean | null
}

export interface CalSummary {
  geometry_epoch: number
  n: number
  n_gauge: number
  n_test: number
  n_stale: number
  scale_factor: number | null
  weighting: string | null
  rows: CalRow[]
  planes?: CalPlaneRow[]
  gauge_fit_residual_rms_m?: number
  gauge_fit_residual_max_m?: number
  gauge_ratio_min?: number
  gauge_ratio_max?: number
  test_abs_error_m?: { median: number | null; max: number; n: number }
  test_rel_error?: { median: number | null; max: number; n: number }
  /** the type chosen and the one in use: per axis only when chosen AND a gauge runs along each axis */
  mode: 'global' | 'axes'
  mode_used: 'global' | 'axes'
  axes: { covered: string[]; missing: string[]; ready: boolean; factors: { x: number; y: number; z: number } | null; reason: string | null }
  /** uncalibrated: nothing calibrates on this geometry; calibrated: a factor, not checked; verified: checked */
  state: 'uncalibrated' | 'calibrated' | 'verified'
  verification: { n: number; max_abs_m: number; max_rel: number; worst_id: string; worst_difference_m: number
                  n_with_sigma: number; all_within_sigma: boolean | null } | null
}

export interface CalPlane {
  id: string
  points: number[][]
  label: string
  note: string
  inclination_deg: number
  height_m: number
  rms_m: number
  max_m: number
  area_m2: number
  geometry_epoch: number
  created: string
}

interface CalPlaneRow {
  id: string
  label: string
  inclination_deg: number
  height_m: number
  rms_m: number
  max_m: number
  area_m2: number
  n_points: number
  geometry_epoch: number
  stale: boolean
}

/** what was just measured on the model and waits to be stored: a distance or a plane */
export type CalibrationDraft =
  | { kind: 'distance'; readoutId: string; a: number[]; b: number[]; metres: number }
  | { kind: 'plane'; readoutId: string; points: number[][] }

interface CalibrationDialogProps {
  open: boolean
  sessionId: string | null
  /** the active scan ("date/source") or null for the session's default */
  scanKey: string | null
  /** the project's other scans: a length can be taken from one of them */
  otherScans: Array<{ key: string; label: string }>
  /** a distance just measured on the model, waiting to be marked (null: the list only) */
  draft: CalibrationDraft | null
  onClose: () => void
  /** start the distance or the plane tool on the model (the window closes; what is measured comes back here) */
  onMeasure: (kind: 'distance' | 'plane') => void
  /** the stored set changed (or was loaded): the viewer draws its marks and planes */
  onChanged: (measurements: CalMeasurement[], summary: CalSummary, planes: CalPlane[]) => void
  /** the draft was stored: its temporary measurement leaves the scene */
  onStored: (readoutId: string) => void
  /** fly to a stored measurement or plane (its points) */
  onFrame: (points: number[][]) => void
}

const scanQuery = (scanKey: string | null) => (scanKey ? `?scan_key=${encodeURIComponent(scanKey)}` : '')

export function CalibrationDialog({ open, sessionId, scanKey, otherScans, draft, onClose, onMeasure, onChanged, onStored, onFrame }: CalibrationDialogProps) {
  const t = useT()
  const fmt = useFmt()
  const { notify } = useToast()

  const [measurements, setMeasurements] = useState<CalMeasurement[]>([])
  const [planes, setPlanes] = useState<CalPlane[]>([])
  const [planeLabel, setPlaneLabel] = useState('')
  const [summary, setSummary] = useState<CalSummary | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // ── the draft's form ──
  const [value, setValue] = useState<number | ''>('')
  const [sigma, setSigma] = useState<number | ''>('')
  const [role, setRole] = useState<CalRole>('gauge')
  const [tool, setTool] = useState<CalTool>('tape')
  const [refScan, setRefScan] = useState<string>('')
  const [note, setNote] = useState('')

  const take = useCallback((d: { document?: { measurements?: CalMeasurement[]; planes?: CalPlane[] }; summary?: CalSummary }) => {
    const ms = d.document?.measurements ?? []
    const ps = d.document?.planes ?? []
    setMeasurements(ms)
    setPlanes(ps)
    setSummary(d.summary ?? null)
    if (d.summary) onChanged(ms, d.summary, ps)
  }, [onChanged])

  useEffect(() => {
    if (!open || !sessionId) return
    let cancelled = false
    setLoadError(null)
    fetch(`/api/calibration/${encodeURIComponent(sessionId)}${scanQuery(scanKey)}`)
      .then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`); return d })
      .then(d => { if (!cancelled) take(d) })
      .catch(e => { if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e)) })
    return () => { cancelled = true }
  }, [open, sessionId, scanKey, take])

  // a new draft starts a clean form (the instrument and the role chosen last are kept)
  useEffect(() => { if (draft) { setValue(''); setSigma(''); setNote('') } }, [draft])

  const distance = draft?.kind === 'distance' ? draft : null
  const plane = draft?.kind === 'plane' ? draft : null
  const canSave = !!distance && typeof value === 'number' && value > 0 && (tool !== 'other_scan' || !!refScan) && !busy

  const storePlane = useCallback(async () => {
    if (!plane || !sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/planes`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_key: scanKey ?? undefined, points: plane.points, label: planeLabel.trim(), note }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
      onStored(plane.readoutId)
      notify(t('calibration.planeSaved', { id: d.plane?.id ?? '' }), { tone: 'ok' })
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [plane, sessionId, scanKey, planeLabel, note, take, onStored, notify, t])

  const removePlane = useCallback(async (id: string) => {
    if (!sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/planes/${encodeURIComponent(id)}${scanQuery(scanKey)}`, { method: 'DELETE' })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
      notify(t('calibration.deleted', { id }), { tone: 'ok' })
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [sessionId, scanKey, take, notify, t])

  const store = useCallback(async () => {
    const draft = distance
    if (!draft || !sessionId || typeof value !== 'number') return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/measurements`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scan_key: scanKey ?? undefined, a: draft.a, b: draft.b, value_m: value, sigma_m: typeof sigma === 'number' ? sigma : 0,
                               role, tool, note, ...(tool === 'other_scan' ? { reference_scan: refScan } : {}) }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
      onStored(draft.readoutId)
      notify(t('calibration.saved', { id: d.measurement?.id ?? '' }), { tone: 'ok' })
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [distance, sessionId, scanKey, value, sigma, role, tool, note, refScan, take, onStored, notify, t])

  const patch = useCallback(async (id: string, body: Record<string, unknown>) => {
    if (!sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/measurements/${encodeURIComponent(id)}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scan_key: scanKey ?? undefined, ...body }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [sessionId, scanKey, take, notify, t])

  const remove = useCallback(async (id: string) => {
    if (!sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/measurements/${encodeURIComponent(id)}${scanQuery(scanKey)}`, { method: 'DELETE' })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
      notify(t('calibration.deleted', { id }), { tone: 'ok' })
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [sessionId, scanKey, take, notify, t])

  const byId = useMemo(() => new Map(measurements.map(m => [m.id, m])), [measurements])
  const scanLabel = useCallback((key?: string) => otherScans.find(s => s.key === key)?.label ?? key ?? '', [otherScans])

  const columns: Column<CalRow>[] = useMemo(() => [
    { id: 'id', header: t('calibration.colId'), mono: true, cell: r => r.id },
    { id: 'role', header: t('calibration.role'),
      cell: r => <Badge tone={r.stale ? 'neutral' : r.role === 'gauge' ? 'ok' : 'warn'} size="sm">{r.role === 'gauge' ? t('calibration.roleGauge') : t('calibration.roleTest')}</Badge> },
    { id: 'model', header: t('calibration.colModel'), align: 'right', mono: true, cell: r => fmt.lengthText(r.measured_m), sortValue: r => r.measured_m },
    { id: 'real', header: t('calibration.colReal'), align: 'right', mono: true, cell: r => fmt.lengthText(r.value_m), sortValue: r => r.value_m },
    { id: 'ratio', header: t('calibration.colRatio'), align: 'right', mono: true, cell: r => fmt.number(r.ratio, 4), sortValue: r => r.ratio },
    { id: 'axis', header: t('calibration.colAxis'), mono: true, cell: r => r.axis, sortValue: r => r.axis },
    { id: 'cal', header: t('calibration.colCalibrated'), align: 'right', mono: true, cell: r => (r.calibrated_m == null ? '' : fmt.lengthText(r.calibrated_m)) },
    { id: 'residual', header: t('calibration.colResidual'), align: 'right', mono: true,
      cell: r => (r.residual_m == null ? '' : (
        <span>
          {`${r.residual_m >= 0 ? '+' : ''}${fmt.mm(r.residual_m * 1000, 1)}`}
          {r.role === 'test' && r.within_sigma != null && <Badge tone={r.within_sigma ? 'ok' : 'err'} size="sm">{r.within_sigma ? t('calibration.checks') : t('calibration.checksNot')}</Badge>}
        </span>
      )), sortValue: r => (r.residual_m == null ? null : Math.abs(r.residual_m)) },
    { id: 'source', header: t('calibration.tool'),
      cell: r => { const m = byId.get(r.id); return m?.tool === 'other_scan' ? t('calibration.fromScan', { name: scanLabel(m.reference_scan) }) : t(`calibration.tool_${r.tool}`) } },
    { id: 'state', header: '', cell: r => (r.stale ? <Badge tone="neutral" size="sm" title={t('calibration.staleHint')}>{t('calibration.stale', { n: r.geometry_epoch })}</Badge> : byId.get(r.id)?.note ?? '') },
    { id: 'actions', header: '', align: 'right',
      cell: r => (
        <Row justify="end">
          <IconButton size="sm" label={t('calibration.frame')} icon={<Crosshair aria-hidden />} onClick={() => { const m = byId.get(r.id); if (m) onFrame([m.a, m.b]) }} />
          <Button size="sm" variant="ghost" disabled={busy} onClick={() => patch(r.id, { role: r.role === 'gauge' ? 'test' : 'gauge' })}>
            {r.role === 'gauge' ? t('calibration.makeTest') : t('calibration.makeGauge')}
          </Button>
          <IconButton size="sm" variant="danger" label={t('common.delete')} icon={<Trash2 aria-hidden />} disabled={busy} onClick={() => remove(r.id)} />
        </Row>
      ) },
  ], [t, fmt, byId, scanLabel, busy, patch, remove, onFrame])

  const setMode = useCallback(async (mode: 'global' | 'axes') => {
    if (!sessionId) return
    setBusy(true)
    try {
      const r = await fetch(`/api/calibration/${encodeURIComponent(sessionId)}/mode`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scan_key: scanKey ?? undefined, mode }),
      })
      const d = await r.json()
      if (!r.ok) throw new Error(d?.detail?.error || d?.detail || `HTTP ${r.status}`)
      take(d)
    } catch (e) {
      notify(t('calibration.saveFailed', { detail: e instanceof Error ? e.message : String(e) }), { tone: 'err' })
    } finally { setBusy(false) }
  }, [sessionId, scanKey, take, notify, t])

  const planeById = useMemo(() => new Map(planes.map(pl => [pl.id, pl])), [planes])
  const planeColumns: Column<CalPlaneRow>[] = useMemo(() => [
    { id: 'id', header: t('calibration.colId'), mono: true, cell: r => r.id },
    { id: 'label', header: t('calibration.planeLabel'), cell: r => r.label },
    { id: 'incl', header: t('calibration.colInclination'), align: 'right', mono: true, cell: r => `${fmt.number(r.inclination_deg, 2)}°`, sortValue: r => r.inclination_deg },
    { id: 'rms', header: t('calibration.colPlaneRms'), align: 'right', mono: true, cell: r => fmt.mm(r.rms_m * 1000, 1), sortValue: r => r.rms_m },
    { id: 'area', header: t('calibration.colArea'), align: 'right', mono: true, cell: r => `${fmt.number(r.area_m2, 2)} m²`, sortValue: r => r.area_m2 },
    { id: 'n', header: t('calibration.colPoints'), align: 'right', mono: true, cell: r => fmt.integer(r.n_points) },
    { id: 'state', header: '', cell: r => (r.stale ? <Badge tone="neutral" size="sm" title={t('calibration.staleHint')}>{t('calibration.stale', { n: r.geometry_epoch })}</Badge> : planeById.get(r.id)?.note ?? '') },
    { id: 'actions', header: '', align: 'right',
      cell: r => (
        <Row justify="end">
          <IconButton size="sm" label={t('calibration.frame')} icon={<Crosshair aria-hidden />} onClick={() => { const pl = planeById.get(r.id); if (pl) onFrame(pl.points) }} />
          <IconButton size="sm" variant="danger" label={t('common.delete')} icon={<Trash2 aria-hidden />} disabled={busy} onClick={() => removePlane(r.id)} />
        </Row>
      ) },
  ], [t, fmt, planeById, busy, removePlane, onFrame])

  const factor = summary?.scale_factor ?? null

  return (
    <Dialog open={open} title={t('calibration.title')} subtitle={[sessionId, scanKey].filter(Boolean).join(', ') || undefined}
      icon={<Pin aria-hidden />} size="lg" onClose={onClose}
      footer={
        <>
          <Button icon={<Ruler aria-hidden />} disabled={!sessionId} onClick={() => onMeasure('distance')} title={t('calibration.measureHint')}>{t('calibration.measure')}</Button>
          <Button icon={<Pentagon aria-hidden />} disabled={!sessionId} onClick={() => onMeasure('plane')} title={t('calibration.planeHint')}>{t('calibration.measurePlane')}</Button>
          <Button onClick={onClose}>{t('common.close')}</Button>
        </>
      }>
      <Stack gap={3}>
        {loadError && <Banner tone="err">{t('calibration.loadFailed', { detail: loadError })}</Banner>}

        {plane && (
          <Section title={t('calibration.planeDraftTitle')}>
            <Stack gap={2}>
              <KeyValue label={t('calibration.planePoints')} mono>{fmt.integer(plane.points.length)}</KeyValue>
              <Row wrap>
                <Field label={t('calibration.planeLabel')} hint={t('calibration.planeLabelHint')}>
                  <Input value={planeLabel} onChange={e => setPlaneLabel(e.target.value)} data-autofocus aria-label={t('calibration.planeLabel')} />
                </Field>
                <Field label={t('calibration.note')}>
                  <Input value={note} onChange={e => setNote(e.target.value)} aria-label={t('calibration.note')} />
                </Field>
              </Row>
              <Row justify="end">
                <Button variant="primary" icon={<Save aria-hidden />} disabled={busy} onClick={storePlane}>{t('calibration.savePlane')}</Button>
              </Row>
            </Stack>
          </Section>
        )}

        {distance && (
          <Section title={t('calibration.draftTitle')}>
            <Stack gap={2}>
              <KeyValue label={t('calibration.modelLength')} mono>{fmt.lengthText(distance.metres)}</KeyValue>
              <Row wrap>
                <Field label={t('calibration.realLength')}>
                  <NumberInput value={value} onChange={setValue} unit="m" min={0} step={0.001} data-autofocus aria-label={t('calibration.realLength')} />
                </Field>
                <Field label={t('calibration.sigma')} hint={t('calibration.sigmaHint')}>
                  <NumberInput value={sigma} onChange={setSigma} unit="m" min={0} step={0.001} aria-label={t('calibration.sigma')} />
                </Field>
              </Row>
              <Field label={t('calibration.role')} hint={t('calibration.roleHint')}>
                <SegmentedControl<CalRole> value={role} onChange={setRole} ariaLabel={t('calibration.role')}
                  options={[{ value: 'gauge', label: t('calibration.roleGauge') }, { value: 'test', label: t('calibration.roleTest') }]} />
              </Field>
              <Row wrap>
                <Field label={t('calibration.tool')}>
                  <Select<CalTool> value={tool} onChange={setTool} aria-label={t('calibration.tool')}
                    options={[{ value: 'tape', label: t('calibration.tool_tape') }, { value: 'laser', label: t('calibration.tool_laser') },
                              { value: 'total_station', label: t('calibration.tool_total_station') },
                              { value: 'other_scan', label: t('calibration.tool_other_scan'), disabled: otherScans.length === 0 }]} />
                </Field>
                {tool === 'other_scan' && (
                  <Field label={t('calibration.referenceScan')} hint={t('calibration.referenceScanHint')}>
                    <Select<string> value={refScan} onChange={setRefScan} placeholder={t('calibration.referenceScan')} aria-label={t('calibration.referenceScan')}
                      options={otherScans.map(s => ({ value: s.key, label: s.label }))} />
                  </Field>
                )}
              </Row>
              <Field label={t('calibration.note')}>
                <Input value={note} onChange={e => setNote(e.target.value)} aria-label={t('calibration.note')} />
              </Field>
              <Row justify="end">
                <Button variant="primary" icon={<Save aria-hidden />} disabled={!canSave} onClick={store}>{t('calibration.save')}</Button>
              </Row>
            </Stack>
          </Section>
        )}

        <Section title={t('calibration.summaryTitle')}>
          {summary && summary.n > 0 ? (
            <Stack gap={1}>
              <KeyValue label={t('calibration.state')}>
                <Badge tone={summary.state === 'verified' ? 'ok' : summary.state === 'calibrated' ? 'info' : 'warn'}>{t(`calibration.state_${summary.state}`)}</Badge>
              </KeyValue>
              <Field label={t('calibration.mode')} hint={summary.axes.ready ? t('calibration.modeHint') : t('calibration.modeAxesMissing', { axes: summary.axes.missing.join(', ') || 'X, Y, Z' })}>
                <SegmentedControl<'global' | 'axes'> value={summary.mode_used} onChange={setMode} ariaLabel={t('calibration.mode')}
                  options={[{ value: 'global', label: t('calibration.modeGlobal') },
                            { value: 'axes', label: t('calibration.modeAxes'), disabled: !summary.axes.ready || busy }]} />
              </Field>
              {summary.mode_used === 'axes' && summary.axes.factors ? (
                <KeyValue label={t('calibration.factorAxes')} mono>
                  {`X × ${fmt.number(summary.axes.factors.x, 5)}   Y × ${fmt.number(summary.axes.factors.y, 5)}   Z × ${fmt.number(summary.axes.factors.z, 5)}`}
                </KeyValue>
              ) : (
                <KeyValue label={t('calibration.factor')} mono>{factor != null ? `× ${fmt.number(factor, 5)}` : t('calibration.noFactor')}</KeyValue>
              )}
              {summary.verification && (
                <KeyValue label={t('calibration.verification')} mono>
                  {t('calibration.verificationValue', { n: summary.verification.n, id: summary.verification.worst_id,
                                                        diff: `${summary.verification.worst_difference_m >= 0 ? '+' : ''}${fmt.mm(summary.verification.worst_difference_m * 1000, 1)}`,
                                                        pct: fmt.percent(summary.verification.max_rel, 2) })}
                </KeyValue>
              )}
              {summary.state === 'calibrated' && <Banner tone="info" compact>{t('calibration.needsVerification')}</Banner>}
              <KeyValue label={t('calibration.counts')}>{t('calibration.countsValue', { gauge: summary.n_gauge, test: summary.n_test, stale: summary.n_stale })}</KeyValue>
              {summary.gauge_fit_residual_rms_m != null && (
                <KeyValue label={t('calibration.fitResidual')} mono>{`${fmt.mm(summary.gauge_fit_residual_rms_m * 1000, 1)} rms, ${fmt.mm((summary.gauge_fit_residual_max_m ?? 0) * 1000, 1)} max`}</KeyValue>
              )}
              {summary.gauge_ratio_min != null && summary.n_gauge > 1 && (
                <KeyValue label={t('calibration.ratioRange')} mono>{`${fmt.number(summary.gauge_ratio_min, 4)} - ${fmt.number(summary.gauge_ratio_max ?? summary.gauge_ratio_min, 4)}`}</KeyValue>
              )}
              {summary.test_abs_error_m && (
                <KeyValue label={t('calibration.testError')} mono>{`${fmt.mm((summary.test_abs_error_m.median ?? 0) * 1000, 1)} ${t('calibration.median')}, ${fmt.mm(summary.test_abs_error_m.max * 1000, 1)} max (${summary.test_abs_error_m.n})`}</KeyValue>
              )}
              <p className="stac-muted">{t('calibration.notApplied')}</p>
            </Stack>
          ) : (
            <EmptyState compact title={t('calibration.empty')} description={t('calibration.emptyHint')} />
          )}
        </Section>

        {summary && summary.rows.length > 0 && (
          <Table columns={columns} rows={summary.rows} rowKey={r => r.id} caption={t('calibration.rows', { n: summary.rows.length })} />
        )}

        {summary && (summary.planes?.length ?? 0) > 0 && (
          <Section title={t('calibration.planesTitle')} hint={t('calibration.planesHint')}>
            <Table columns={planeColumns} rows={summary.planes ?? []} rowKey={r => r.id} caption={t('calibration.planeRows', { n: summary.planes?.length ?? 0 })} />
          </Section>
        )}
      </Stack>
    </Dialog>
  )
}
