/**
 * SessionsPanel — the left "Sessions" tab: search, tree of sessions with
 * their scans as children (reference marked, double-click opens the scan
 * as a tab), row actions (load, flythrough, reconstruct / upload video,
 * segment, manager, unload, rename, delete) and "New session" for
 * admins/managers. Presentational: every handler comes from App.
 */
import { useState } from 'react'
import { ArrowUpFromLine, Building2, Crosshair, FolderOpen, Hammer, Layers, Loader2, Pencil, Play, Plug, Plus, Star, Tag, Trash2, Check, X } from 'lucide-react'
import { Panel, Row, Stack } from '../components/ui/Panel'
import { Tree, type TreeNodeData } from '../components/ui/Tree'
import { IconButton } from '../components/ui/IconButton'
import { Button } from '../components/ui/Button'
import { Input, SearchInput } from '../components/ui/Field'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface SessionRow {
  id: string
  name: string
  frameCount: number
  hasCloud: boolean
  hasSegments: boolean
  hasBim: boolean
  bimCount: number
  cloudSizeMb: number
}

export interface ScanRow {
  key: string
  label: string
  date: string
  kind?: 'scan' | 'fused'
  is_reference?: boolean
  points?: number
  recon_state?: string
}

interface SessionsPanelProps {
  connected: boolean
  sessions: SessionRow[]
  selectedSession: string | null
  activeSession: string | null
  scansOf: (sessionId: string) => ScanRow[]
  activeScanKey: string | null
  rebuildingSession: string | null
  extracting: Record<string, number>
  canManage: boolean
  onConnect: () => void
  onSelect: (id: string) => void
  onLoad: (id: string) => void
  onFlythrough: (id: string) => void
  onReconstruct: (id: string) => void
  onPickVideo: (id: string) => void
  onSegment: (id: string) => void
  onOpenManager: (id: string) => void
  onUnload: () => void
  onRename: (id: string, name: string) => Promise<void>
  onDelete: (id: string) => void
  onCreate: (name: string) => Promise<void>
  onOpenScan: (sessionId: string, scan: ScanRow) => void
  onSetReference: (sessionId: string, scan: ScanRow) => void
}

export function SessionsPanel(p: SessionsPanelProps) {
  const t = useT()
  const fmt = useFmt()
  const [filter, setFilter] = useState('')
  const [renaming, setRenaming] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [creating, setCreating] = useState(false)
  const [newName, setNewName] = useState('')

  const commitRename = async (id: string) => {
    const v = renameValue.trim()
    setRenaming(null)
    if (v && v !== id) await p.onRename(id, v)
  }
  const commitCreate = async () => {
    const v = newName.trim()
    if (!v) return
    await p.onCreate(v)
    setNewName(''); setCreating(false)
  }

  const nodes: TreeNodeData<{ kind: 'session' | 'scan'; sessionId: string; scan?: ScanRow }>[] = p.sessions
    .filter(s => !filter || s.name.toLowerCase().includes(filter.toLowerCase()))
    .map(s => {
      const loaded = p.activeSession === s.id
      const rebuilding = p.rebuildingSession === s.id
      const extractingPct = p.extracting[s.id]
      const kids = p.scansOf(s.id)
      return {
        id: `s:${s.id}`,
        data: { kind: 'session', sessionId: s.id },
        icon: <FolderOpen aria-hidden />,
        label: renaming === s.id ? (
          <Input size="sm" autoFocus value={renameValue} onChange={e => setRenameValue(e.target.value)}
            onClick={e => e.stopPropagation()}
            onKeyDown={e => { if (e.key === 'Enter') commitRename(s.id); if (e.key === 'Escape') setRenaming(null) }}
            onBlur={() => commitRename(s.id)} aria-label={t('sessions.renameLabel')} />
        ) : s.name,
        title: s.id,
        meta: (
          <Row className="stac-sessions__meta">
            <span>{t.plural('sessions.frames', s.frameCount)}</span>
            {s.hasCloud && <span>{fmt.bytesMb(s.cloudSizeMb)}</span>}
            {s.hasSegments && <Tag aria-hidden />}
            {s.hasBim && <><Building2 aria-hidden />{s.bimCount > 1 ? <span>{s.bimCount}</span> : null}</>}
          </Row>
        ),
        badge: rebuilding ? <Badge tone="brand" size="sm" dot>{t('sessions.rebuilding')}</Badge> : loaded ? <Badge tone="ok" size="sm">{t('sessions.loaded')}</Badge> : undefined,
        muted: !s.hasCloud && s.frameCount === 0,
        twoLine: true,
        defaultExpanded: loaded,
        actions: (
          <>
            <IconButton size="sm" label={s.hasCloud ? t('sessions.load') : t('sessions.noCloudYet')} icon={<FolderOpen aria-hidden />} disabled={!s.hasCloud} onClick={() => p.onLoad(s.id)} />
            <IconButton size="sm" label={s.hasCloud ? t('sessions.flythrough') : t('sessions.noCloudYet')} icon={<Play aria-hidden />} disabled={!s.hasCloud} onClick={() => p.onFlythrough(s.id)} />
            {extractingPct != null ? (
              <IconButton size="sm" label={t('sessions.extracting', { pct: fmt.percent(extractingPct / 100) })} icon={<Loader2 className="stac-spin" aria-hidden />} disabled onClick={() => {}} />
            ) : s.frameCount > 0 ? (
              <IconButton size="sm" label={t('sessions.reconstruct')} icon={<Hammer aria-hidden />} onClick={() => p.onReconstruct(s.id)} />
            ) : p.canManage ? (
              <IconButton size="sm" label={t('sessions.uploadVideo')} icon={<Plus aria-hidden />} onClick={() => p.onPickVideo(s.id)} />
            ) : null}
            {loaded && <IconButton size="sm" label={t('sessions.segmentAuto')} icon={<Tag aria-hidden />} onClick={() => p.onSegment(s.id)} />}
            {loaded && <IconButton size="sm" label={t('sessions.segmentManual')} icon={<Crosshair aria-hidden />} onClick={() => p.onOpenManager(s.id)} />}
            {loaded && <IconButton size="sm" label={t('sessions.unload')} icon={<ArrowUpFromLine aria-hidden />} onClick={p.onUnload} />}
            {p.canManage && <IconButton size="sm" label={t('sessions.rename')} icon={<Pencil aria-hidden />} onClick={() => { setRenaming(s.id); setRenameValue(s.id) }} />}
            {p.canManage && <IconButton size="sm" variant="danger" label={t('sessions.delete')} icon={<Trash2 aria-hidden />} onClick={() => p.onDelete(s.id)} />}
          </>
        ),
        children: kids.map((sc, i) => ({
          id: `sc:${s.id}:${sc.key}:${i}`,
          data: { kind: 'scan', sessionId: s.id, scan: sc },
          icon: sc.kind === 'fused' ? <Layers aria-hidden /> : undefined,
          label: sc.kind === 'fused' ? sc.label : `${sc.date} ${sc.label}`,
          title: sc.kind === 'fused' ? t('scans.fusedHint') : t('scans.openHint'),
          meta: sc.points ? fmt.millions(sc.points) : sc.recon_state ? t(`reconState.${sc.recon_state}`) : undefined,
          badge: sc.is_reference ? <Star className="stac-sessions__star" aria-hidden /> : undefined,
          actions: sc.kind !== 'fused' && !sc.is_reference ? (
            <IconButton size="sm" label={t('scans.setReference')} icon={<Star aria-hidden />} onClick={() => p.onSetReference(s.id, sc)} />
          ) : undefined,
        })),
      }
    })

  const selectedId = p.selectedSession ? `s:${p.selectedSession}` : null
  const activeId = p.activeSession && p.activeScanKey ? nodes.flatMap(n => n.children ?? []).find(c => c.data?.scan?.key === p.activeScanKey && c.data?.sessionId === p.activeSession)?.id ?? null : null

  return (
    <Panel title={t('sessions.title')} subtitle={p.connected ? t.plural('welcome.sessionsAvailable', p.sessions.length) : t('header.notConnected')}
      toolbar={p.connected ? <SearchInput size="sm" value={filter} onChange={e => setFilter(e.target.value)} onClear={() => setFilter('')} placeholder={t('sessions.search')} /> : undefined}
      isEmpty={!p.connected || p.sessions.length === 0}
      empty={!p.connected
        ? <EmptyState icon={<Plug aria-hidden />} title={t('sessions.notConnectedTitle')} description={t('welcome.connectHint')} action={<Button variant="primary" icon={<Plug aria-hidden />} onClick={p.onConnect}>{t('welcome.connect')}</Button>} />
        : <EmptyState icon={<FolderOpen aria-hidden />} title={t('sessions.emptyTitle')} description={t('sessions.emptyDesc')} action={p.canManage ? <Button variant="primary" icon={<Plus aria-hidden />} onClick={() => setCreating(true)}>{t('sessions.new')}</Button> : undefined} />}
      footer={p.connected && p.canManage ? (
        creating ? (
          <Row className="stac-sessions__create">
            <Input size="sm" autoFocus placeholder={t('sessions.newPlaceholder')} value={newName} onChange={e => setNewName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') commitCreate(); if (e.key === 'Escape') { setCreating(false); setNewName('') } }} aria-label={t('sessions.newPlaceholder')} />
            <IconButton size="sm" label={t('sessions.create')} icon={<Check aria-hidden />} variant="primary" disabled={!newName.trim()} onClick={commitCreate} />
            <IconButton size="sm" label={t('common.cancel')} icon={<X aria-hidden />} onClick={() => { setCreating(false); setNewName('') }} />
          </Row>
        ) : <Button block icon={<Plus aria-hidden />} onClick={() => setCreating(true)}>{t('sessions.new')}</Button>
      ) : undefined}>
      <Stack gap={1}>
        <Tree ariaLabel={t('sessions.title')} nodes={nodes} selectedId={selectedId} activeId={activeId}
          onSelect={n => { if (n.data?.kind === 'session') p.onSelect(n.data.sessionId) }}
          onActivate={n => { if (n.data?.kind === 'scan' && n.data.scan) p.onOpenScan(n.data.sessionId, n.data.scan); else if (n.data?.kind === 'session') { const s = p.sessions.find(x => x.id === n.data!.sessionId); if (s?.hasCloud) p.onLoad(s.id) } }} />
      </Stack>
    </Panel>
  )
}
