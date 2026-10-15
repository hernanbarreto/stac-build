/**
 * EvaluationPanel — read-only browser of the evaluation battery's runs
 * (bottom dock). Side column: run selector + the run's files grouped
 * run-level first, then scene / variant / phase; main column: the selected
 * file rendered by kind (csv as a table, json as key/value sections, png as
 * an image, log/other as preformatted text). Nothing here edits anything.
 *
 * Endpoints: GET /api/evaluation/runs, GET /api/evaluation/runs/{run_id},
 * GET /api/evaluation/runs/{run_id}/file?path=<relative path>.
 */
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { Braces, File, FileSpreadsheet, FileText, FlaskConical, Folder, Image, RefreshCw } from 'lucide-react'
import { Badge, type BadgeTone } from '../components/ui/Badge'
import { Banner } from '../components/ui/Banner'
import { IconButton } from '../components/ui/IconButton'
import { EmptyState } from '../components/ui/EmptyState'
import { Select, SegmentedControl } from '../components/ui/Field'
import { KeyValue, Row, Stack } from '../components/ui/Panel'
import { Table, type Column } from '../components/ui/Table'
import { Tree, type TreeNodeData } from '../components/ui/Tree'
import { useFmt, useT, type Formatters } from '../i18n'

type RunStatus = 'running' | 'finished' | 'failed'
type FileKind = 'csv' | 'json' | 'png' | 'log' | 'other'

interface EvalRun { run_id: string; created: string; scenes: string[]; variants: string[]; status: RunStatus; n_files: number }
interface EvalFile { path: string; kind: FileKind; scene: string | null; variant: string | null; phase: string | null; size: number }
interface EvalRunDetail { run_id: string; manifest: Record<string, unknown> | null; files: EvalFile[] }

const STATUS_TONE: Record<RunStatus, BadgeTone> = { running: 'info', finished: 'ok', failed: 'err' }
const KIND_ICON: Record<FileKind, ReactNode> = {
  csv: <FileSpreadsheet aria-hidden />, json: <Braces aria-hidden />, png: <Image aria-hidden />, log: <FileText aria-hidden />, other: <File aria-hidden />,
}
const JSON_MAX_DEPTH = 5

const fileUrl = (runId: string, path: string) => `/api/evaluation/runs/${encodeURIComponent(runId)}/file?path=${encodeURIComponent(path)}`
const basename = (p: string) => p.split('/').pop() ?? p

// ── CSV ─────────────────────────────────────────────────────────────────
/** RFC-4180-ish: quoted fields may hold the delimiter, newlines and doubled quotes; CRLF accepted. */
export function parseCsv(text: string): string[][] {
  const head = text.slice(0, text.indexOf('\n') >= 0 ? text.indexOf('\n') : text.length)
  const delim = [',', ';', '\t'].map(d => ({ d, n: head.split(d).length })).sort((a, b) => b.n - a.n)[0].d
  const rows: string[][] = []
  let row: string[] = [], field = '', quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++ } else quoted = false
      } else field += c
      continue
    }
    if (c === '"') quoted = true
    else if (c === delim) { row.push(field); field = '' }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = '' }
    else if (c !== '\r') field += c
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row) }
  return rows.filter(r => !(r.length === 1 && r[0].trim() === ''))
}

const isNumeric = (s: string) => s.trim() !== '' && Number.isFinite(Number(s))
const decimalsOf = (s: string) => { const m = /\.(\d+)/.exec(s.trim()); return m ? Math.min(6, m[1].length) : 0 }
const fmtNumberText = (fmt: Formatters, s: string) => { const n = Number(s); return Number.isInteger(n) && decimalsOf(s) === 0 ? fmt.integer(n) : fmt.number(n, decimalsOf(s)) }

function CsvView({ text }: { text: string }) {
  const t = useT()
  const fmt = useFmt()
  const { columns, rows } = useMemo(() => {
    const parsed = parseCsv(text)
    if (!parsed.length) return { columns: [] as Column<string[]>[], rows: [] as string[][] }
    const header = parsed[0]
    const body = parsed.slice(1)
    const columns: Column<string[]>[] = header.map((name, i) => {
      const cells = body.map(r => r[i] ?? '').filter(c => c.trim() !== '')
      const numeric = cells.length > 0 && cells.every(isNumeric)
      return {
        id: `c${i}`,
        header: name || t('evaluation.column', { n: i + 1 }),
        align: numeric ? 'right' : 'left',
        mono: numeric,
        cell: r => { const c = r[i] ?? ''; return numeric && isNumeric(c) ? fmtNumberText(fmt, c) : c },
        sortValue: numeric ? r => (isNumeric(r[i] ?? '') ? Number(r[i]) : null) : r => r[i] ?? '',
      }
    })
    return { columns, rows: body }
  }, [text, t, fmt])
  if (!columns.length) return <EmptyState compact title={t('evaluation.empty')} />
  return <Table columns={columns} rows={rows} rowKey={(_, i) => i} caption={t('evaluation.rows', { n: rows.length })} />
}

// ── JSON ────────────────────────────────────────────────────────────────
const isScalar = (v: unknown) => v == null || typeof v !== 'object'
const isPlainObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v)

function scalarText(v: unknown, fmt: Formatters): string {
  if (v == null) return 'null'
  if (typeof v === 'number') return Number.isInteger(v) ? fmt.integer(v) : fmt.number(v, Math.min(6, decimalsOf(String(v))) || 4)
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  return String(v)
}

function uniformKeys(list: unknown[]): string[] | null {
  if (!list.length || !list.every(isPlainObject)) return null
  const keys = Object.keys(list[0] as Record<string, unknown>)
  const sig = [...keys].sort().join('\u0000')
  return list.every(o => Object.keys(o as Record<string, unknown>).sort().join('\u0000') === sig) ? keys : null
}

function JsonView({ value, depth }: { value: unknown; depth: number }) {
  const t = useT()
  const fmt = useFmt()
  if (depth > JSON_MAX_DEPTH) return <pre className="stac-eval__pre stac-mono">{JSON.stringify(value, null, 2)}</pre>
  if (isScalar(value)) return <span className="stac-mono">{scalarText(value, fmt)}</span>
  if (Array.isArray(value)) {
    if (!value.length) return <span className="stac-muted">{t('evaluation.empty')}</span>
    if (value.every(isScalar)) return <span className="stac-mono">{value.map(v => scalarText(v, fmt)).join(', ')}</span>
    const keys = uniformKeys(value)
    if (keys) {
      const rows = value as Record<string, unknown>[]
      const columns: Column<Record<string, unknown>>[] = keys.map(k => {
        const numeric = rows.every(r => typeof r[k] === 'number' || r[k] == null) && rows.some(r => typeof r[k] === 'number')
        return {
          id: k, header: k, align: numeric ? 'right' : 'left', mono: numeric,
          cell: r => (isScalar(r[k]) ? scalarText(r[k], fmt) : JSON.stringify(r[k])),
          sortValue: r => (isScalar(r[k]) ? (r[k] as string | number | null) : JSON.stringify(r[k])),
        }
      })
      return <Table columns={columns} rows={rows} rowKey={(_, i) => i} caption={t('evaluation.rows', { n: rows.length })} maxHeight={false} />
    }
    return (
      <div className="stac-eval__json">
        {value.map((v, i) => (
          <div key={i} className="stac-eval__json-section">
            <div className="stac-eval__json-key stac-mono">{`[${i}]`}</div>
            <JsonView value={v} depth={depth + 1} />
          </div>
        ))}
      </div>
    )
  }
  const entries = Object.entries(value as Record<string, unknown>)
  if (!entries.length) return <span className="stac-muted">{t('evaluation.empty')}</span>
  return (
    <div className="stac-eval__json">
      {entries.map(([k, v]) => (
        isScalar(v) || (Array.isArray(v) && v.every(isScalar))
          ? <KeyValue key={k} label={k} mono><JsonView value={v} depth={depth + 1} /></KeyValue>
          : (
            <div key={k} className="stac-eval__json-section">
              <div className="stac-eval__json-key">{k}</div>
              <JsonView value={v} depth={depth + 1} />
            </div>
          )
      ))}
    </div>
  )
}

// ── panel ───────────────────────────────────────────────────────────────
export function EvaluationPanel() {
  const t = useT()
  const fmt = useFmt()
  const [runs, setRuns] = useState<EvalRun[]>([])
  const [runsError, setRunsError] = useState<string | null>(null)
  const [runsLoading, setRunsLoading] = useState(false)
  const [runId, setRunId] = useState<string>('')
  const [detail, setDetail] = useState<EvalRunDetail | null>(null)
  const [detailError, setDetailError] = useState<string | null>(null)
  const [selected, setSelected] = useState<EvalFile | null>(null)
  const [content, setContent] = useState<{ path: string; text: string } | null>(null)
  const [contentError, setContentError] = useState<string | null>(null)
  const [contentLoading, setContentLoading] = useState(false)
  const [jsonMode, setJsonMode] = useState<'structured' | 'raw'>('structured')

  const loadRuns = useCallback(async () => {
    setRunsLoading(true)
    setRunsError(null)
    try {
      const r = await fetch('/api/evaluation/runs')
      if (!r.ok) throw new Error(`${r.status}`)
      const data = await r.json()
      const list: EvalRun[] = Array.isArray(data?.runs) ? data.runs : []
      setRuns(list)
      setRunId(cur => (cur && list.some(x => x.run_id === cur) ? cur : (list[0]?.run_id ?? '')))
    } catch (e: any) {
      setRunsError(e?.message ?? String(e))
    } finally {
      setRunsLoading(false)
    }
  }, [])

  useEffect(() => { loadRuns() }, [loadRuns])

  useEffect(() => {
    setDetail(null); setDetailError(null); setSelected(null); setContent(null); setContentError(null)
    if (!runId) return
    let cancelled = false
    fetch(`/api/evaluation/runs/${encodeURIComponent(runId)}`)
      .then(async r => { if (!r.ok) throw new Error(`${r.status}`); return r.json() })
      .then((d: EvalRunDetail) => { if (!cancelled) setDetail({ run_id: d?.run_id ?? runId, manifest: isPlainObject(d?.manifest) ? d.manifest : null, files: Array.isArray(d?.files) ? d.files : [] }) })
      .catch((e: any) => { if (!cancelled) setDetailError(e?.message ?? String(e)) })
    return () => { cancelled = true }
  }, [runId])

  useEffect(() => {
    setContent(null); setContentError(null)
    if (!selected || !runId || selected.kind === 'png') return
    let cancelled = false
    setContentLoading(true)
    fetch(fileUrl(runId, selected.path))
      .then(async r => { if (!r.ok) throw new Error(`${r.status}`); return r.text() })
      .then(text => { if (!cancelled) setContent({ path: selected.path, text }) })
      .catch((e: any) => { if (!cancelled) setContentError(e?.message ?? String(e)) })
      .finally(() => { if (!cancelled) setContentLoading(false) })
    return () => { cancelled = true }
  }, [selected, runId])

  const run = runs.find(r => r.run_id === runId) ?? null
  const sizeText = useCallback((bytes: number) => (bytes >= 1e6 ? fmt.bytesMb(bytes / 1e6) : t('evaluation.kb', { n: fmt.integer(Math.max(1, Math.round(bytes / 1024))) })), [fmt, t])

  // run-level files first (scene null), then scene > variant > phase
  const nodes = useMemo<TreeNodeData<EvalFile>[]>(() => {
    if (!detail) return []
    const fileNode = (f: EvalFile): TreeNodeData<EvalFile> => ({
      id: `f:${f.path}`, label: basename(f.path), title: f.path, icon: KIND_ICON[f.kind], meta: sizeText(f.size), data: f,
    })
    const sortedKeys = (m: Map<string, unknown>) => [...m.keys()].sort((a, b) => a.localeCompare(b))
    const out: TreeNodeData<EvalFile>[] = []
    const top = detail.files.filter(f => f.scene == null)
    if (top.length) out.push({ id: 'run', label: t('evaluation.runFiles'), icon: <Folder aria-hidden />, badge: top.length, defaultExpanded: true, children: top.map(fileNode) })
    const scenes = new Map<string, Map<string, Map<string, EvalFile[]>>>()
    for (const f of detail.files) {
      if (f.scene == null) continue
      const sc = scenes.get(f.scene) ?? new Map<string, Map<string, EvalFile[]>>()
      scenes.set(f.scene, sc)
      const va = sc.get(f.variant ?? '') ?? new Map<string, EvalFile[]>()
      sc.set(f.variant ?? '', va)
      const ph = va.get(f.phase ?? '') ?? []
      va.set(f.phase ?? '', ph)
      ph.push(f)
    }
    for (const scene of sortedKeys(scenes)) {
      const sc = scenes.get(scene)!
      const variants: TreeNodeData<EvalFile>[] = []
      for (const variant of sortedKeys(sc)) {
        const va = sc.get(variant)!
        const phases: TreeNodeData<EvalFile>[] = []
        for (const phase of sortedKeys(va)) {
          const files = va.get(phase)!.map(fileNode)
          if (phase === '') phases.push(...files)
          else phases.push({ id: `ph:${scene}/${variant}/${phase}`, label: phase, icon: <Folder aria-hidden />, badge: files.length, children: files })
        }
        if (variant === '') variants.push(...phases)
        else variants.push({ id: `va:${scene}/${variant}`, label: variant, icon: <Folder aria-hidden />, badge: va.size, children: phases })
      }
      out.push({ id: `sc:${scene}`, label: scene, icon: <FlaskConical aria-hidden />, badge: sc.size, defaultExpanded: scenes.size === 1, children: variants })
    }
    return out
  }, [detail, sizeText, t])

  const renderContent = () => {
    if (!runId) return <EmptyState compact icon={<FlaskConical aria-hidden />} title={t('evaluation.noRuns')} description={t('evaluation.noRunsHint')} />
    if (detailError) return <Banner tone="err" compact>{t('evaluation.loadFailed', { detail: detailError })}</Banner>
    if (!selected) {
      if (detail?.manifest) return (
        <Stack gap={2}>
          <div className="stac-section__title">{t('evaluation.manifest')}</div>
          <JsonView value={detail.manifest} depth={0} />
        </Stack>
      )
      return <EmptyState compact icon={<FlaskConical aria-hidden />} title={t('evaluation.noFile')} description={t('evaluation.noFileHint')} />
    }
    if (selected.kind === 'png') return <img className="stac-eval__img" src={fileUrl(runId, selected.path)} alt={selected.path} />
    if (contentError) return <Banner tone="err" compact>{t('evaluation.fileFailed', { path: selected.path, detail: contentError })}</Banner>
    if (contentLoading || !content || content.path !== selected.path) return <span className="stac-muted">{t('common.loading')}</span>
    if (selected.kind === 'csv') return <CsvView text={content.text} />
    if (selected.kind === 'json') {
      let parsed: unknown = undefined
      let bad = false
      try { parsed = JSON.parse(content.text) } catch { bad = true }
      if (bad || jsonMode === 'raw') return <pre className="stac-eval__pre stac-mono">{content.text}</pre>
      return <JsonView value={parsed} depth={0} />
    }
    return <pre className="stac-eval__pre stac-mono">{content.text}</pre>
  }

  return (
    <div className="stac-eval">
      <div className="stac-eval__side">
        <Row>
          <Select<string> size="sm" value={runId} onChange={setRunId} placeholder={runs.length ? undefined : t('evaluation.noRuns')} disabled={!runs.length} className="stac-eval__select"
            options={runs.map(r => ({ value: r.run_id, label: r.run_id }))} aria-label={t('evaluation.run')} />
          <IconButton size="sm" label={t('evaluation.refresh')} icon={<RefreshCw aria-hidden />} loading={runsLoading} onClick={loadRuns} />
        </Row>
        {runsError && <Banner tone="err" compact>{t('evaluation.loadFailed', { detail: runsError })}</Banner>}
        {run && (
          <div className="stac-eval__run">
            <Row wrap>
              <Badge size="sm" tone={STATUS_TONE[run.status] ?? 'neutral'}>{t(`evaluation.status.${run.status}`)}</Badge>
              <span className="stac-eval__meta">{fmt.dateTime(run.created)}</span>
              <span className="stac-eval__meta">{t('evaluation.files', { n: run.n_files })}</span>
            </Row>
            <div className="stac-eval__meta">{t('evaluation.scenes', { n: run.scenes.length, list: run.scenes.join(', ') })}</div>
            <div className="stac-eval__meta">{t('evaluation.variants', { n: run.variants.length, list: run.variants.join(', ') })}</div>
          </div>
        )}
        <div className="stac-eval__tree">
          {runId && !detail && !detailError && <span className="stac-muted stac-eval__loading">{t('common.loading')}</span>}
          {detail && nodes.length === 0 && <EmptyState compact title={t('evaluation.noFiles')} />}
          {nodes.length > 0 && (
            <Tree<EvalFile> nodes={nodes} dense ariaLabel={t('evaluation.fileTree')} selectedId={selected ? `f:${selected.path}` : null}
              onSelect={n => { if (n.data) setSelected(n.data) }} />
          )}
        </div>
      </div>
      <div className="stac-eval__main">
        {selected && (
          <div className="stac-eval__head">
            <span className="stac-eval__icon">{KIND_ICON[selected.kind]}</span>
            <span className="stac-eval__path" title={selected.path}>{selected.path}</span>
            <Badge size="sm" tone="neutral">{t(`evaluation.kind.${selected.kind}`)}</Badge>
            <span className="stac-eval__meta">{sizeText(selected.size)}</span>
            {selected.kind === 'json' && (
              <SegmentedControl<'structured' | 'raw'> size="sm" ariaLabel={t('evaluation.jsonView')} value={jsonMode} onChange={setJsonMode}
                options={[{ value: 'structured', label: t('evaluation.structured') }, { value: 'raw', label: t('evaluation.raw') }]} />
            )}
          </div>
        )}
        <div className="stac-eval__content">{renderContent()}</div>
      </div>
    </div>
  )
}
