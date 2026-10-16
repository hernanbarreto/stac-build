/**
 * InstancesPanel — the "Instances" tab: visibility tree of segments (colour
 * dot, point count, rename, delete), the unsegmented remainder, placed
 * objects, generated meshes (Object / Mesh) and the floor-at-y=0 selector;
 * footer buttons open Segmentation and Meshing.
 */
import { memo, useState } from 'react'
import { Box, CheckSquare, Crosshair, Grid3x3, Pencil, Puzzle, Spline, Square, Tag, Trash2, Wand2 } from 'lucide-react'
import type { SegmentInstance } from '../components/Viewport'
import { Panel, Section, Stack } from '../components/ui/Panel'
import { Tree, type TreeNodeData } from '../components/ui/Tree'
import { IconButton } from '../components/ui/IconButton'
import { Button } from '../components/ui/Button'
import { Input, SearchInput, Select, Field } from '../components/ui/Field'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface MeshListItem { instanceId: number; label: string; folder: string; visible: boolean; wireframe?: boolean }
export interface PlacedObject { id: number; name: string; visible: boolean }
export interface VolumeItem { volume_id: number; name: string }
export interface ListedRows { segmentKeys: string[]; unsegmented: boolean; placedIds: number[]; shapeFolders: string[]; tsdfFolders: string[] }
export interface FloorLevelState { candidates: { instance_id: number; label: string; height_m: number | null }[]; selected: number | null }

interface InstancesPanelProps {
  segments: SegmentInstance[]
  unsegmentedVisible: boolean
  unsegmentedCount: number | null
  /** masks the matching resolved into another object (or that matched no
   *  point): not listed, but counted, so the fusion is never silent */
  absorbedCount: number | null
  floorLevel: FloorLevelState
  placedObjects: PlacedObject[]
  /** evaluation volumes (the assistant's or the user's): listed, framed on click, deletable —
   *  they stay in the scene when the chat closes (USER 2026-10-09) */
  volumes: VolumeItem[]
  onFocusVolume: (id: number) => void
  onDeleteVolume: (id: number) => void
  /** every row has a VISIBLE box and an OBB box (USER 2026-10-09): visible on, OBB off by default */
  obbKeys: Set<string>
  onToggleObb: (segKey: string, on: boolean) => void
  placedObb: Set<number>
  onTogglePlacedObb: (id: number, on: boolean) => void
  volumeHidden: Set<number>
  volumeObb: Set<number>
  onToggleVolume: (id: number, visible: boolean) => void
  onToggleVolumeObb: (id: number, on: boolean) => void
  shapeMeshes: MeshListItem[]
  tsdfMeshes: MeshListItem[]
  selectedSegmentId: number | null
  onSelectSegment: (id: number | null) => void
  /** show / hide EVERY row the list shows (the search applies): segments, Unsegmented, placed objects,
   *  generated objects, meshes (USER 2026-10-01) */
  onSetAllVisible: (visible: boolean, listed: ListedRows) => void
  onToggleSegment: (seg: SegmentInstance, visible: boolean) => void
  onRenameSegment: (seg: SegmentInstance, label: string) => Promise<void>
  /** the Manhattan mark (USER 2026-10-15), on by default */
  onToggleManhattan: (seg: SegmentInstance, on: boolean) => Promise<void>
  onDeleteSegment: (seg: SegmentInstance) => void
  onToggleUnsegmented: (visible: boolean) => void
  onFloorLevel: (instanceId: number) => void
  onTogglePlaced: (id: number, visible: boolean) => void
  onRemovePlaced: (id: number) => void
  onRenamePlaced: (id: number, name: string) => Promise<void>
  /** a row that is not a segment was clicked: fly to it */
  onFocusItem: (kind: 'placed' | 'shape' | 'tsdf', key: string | number, instanceId?: number) => void
  onToggleShape: (m: MeshListItem, visible: boolean) => void
  onRenameShape: (m: MeshListItem, label: string) => Promise<void>
  onDeleteShape: (m: MeshListItem) => void
  onToggleTsdf: (m: MeshListItem, visible: boolean) => void
  /** draw a GLB as wireframe (USER 2026-10-15) */
  onToggleShapeWireframe: (m: MeshListItem, on: boolean) => void
  onToggleTsdfWireframe: (m: MeshListItem, on: boolean) => void
  onDeleteTsdf: (m: MeshListItem) => void
  onOpenSegmentation: () => void
  onOpenMeshing: () => void
  /** the segmentation chain on demand, prompts editable (USER 2026-10-05) */
  onOpenAutosegment: () => void
}

function InstancesPanelImpl(p: InstancesPanelProps) {
  const t = useT()
  const fmt = useFmt()
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')

  const commit = async (seg: SegmentInstance) => {
    const v = editValue.trim()
    setEditing(null)
    if (v && v !== seg.label) await p.onRenameSegment(seg, v)
  }

  // ONE search over the whole list (USER 2026-10-01): segments, placed objects, generated objects, meshes
  const q = search.trim().toLowerCase()
  const hit = (...txt: Array<string | number | undefined>) => !q || txt.some(x => x != null && String(x).toLowerCase().includes(q))
  const segs = p.segments.filter(s => hit(s.label, s.id))
  const placed = p.placedObjects.filter(o => hit(o.name, o.id))
  const shapes = p.shapeMeshes.filter(m => hit(m.label, m.folder, m.instanceId))
  const tsdfs = p.tsdfMeshes.filter(m => hit(m.label, m.folder, m.instanceId))
  const vols = p.volumes.filter(v => hit(v.name, v.volume_id))
  const unsegListed = hit(t('instances.unsegmented'))
  const listed: ListedRows = { segmentKeys: segs.map(s => s.key), unsegmented: unsegListed, placedIds: placed.map(o => o.id),
                               shapeFolders: shapes.map(m => m.folder), tsdfFolders: tsdfs.map(m => m.folder) }
  const anyListed = segs.length + placed.length + shapes.length + tsdfs.length + vols.length > 0 || unsegListed
  const commitPlaced = async (o: PlacedObject) => {
    const v = editValue.trim()
    setEditing(null)
    if (v && v !== o.name) await p.onRenamePlaced(o.id, v)
  }
  const commitShape = async (m: MeshListItem) => {
    const v = editValue.trim()
    setEditing(null)
    if (v && v !== m.label) await p.onRenameShape(m, v)
  }

  // a generated mesh's OBB is its instance's
  const obbOf = (instanceId: number): Partial<TreeNodeData> => {
    const key = p.segments.find(s => s.id === instanceId)?.key
    return key ? { obb: p.obbKeys.has(key), onObb: (v: boolean) => p.onToggleObb(key, v) } : {}
  }

  const segNodes: TreeNodeData<SegmentInstance>[] = segs
    .map(seg => ({
      id: seg.key,
      data: seg,
      color: seg.color,
      visible: seg.visible,
      onVisible: v => p.onToggleSegment(seg, v),
      obb: p.obbKeys.has(seg.key),
      onObb: v => p.onToggleObb(seg.key, v),
      label: editing === seg.key ? (
        <Input size="sm" autoFocus value={editValue} onChange={e => setEditValue(e.target.value)} onClick={e => e.stopPropagation()}
          onKeyDown={e => { if (e.key === 'Enter') commit(seg); if (e.key === 'Escape') setEditing(null) }} onBlur={() => commit(seg)} aria-label={t('instances.renameLabel')} />
      ) : seg.label,
      meta: fmt.integer(seg.totalPoints),
      muted: seg.excluded,
      actions: (
        <>
          <IconButton size="sm" label={seg.manhattan !== false ? t('instances.manhattanOn') : t('instances.manhattanOff')}
            icon={seg.manhattan !== false ? <Box aria-hidden /> : <Spline aria-hidden />}
            onClick={() => p.onToggleManhattan(seg, seg.manhattan === false)} />
          <IconButton size="sm" label={t('instances.rename')} icon={<Pencil aria-hidden />} onClick={() => { setEditing(seg.key); setEditValue(seg.label) }} />
          <IconButton size="sm" variant="danger" label={t('instances.delete')} icon={<Trash2 aria-hidden />} onClick={() => p.onDeleteSegment(seg)} />
        </>
      ),
    }))

  const unsegNode: TreeNodeData = {
    id: 'unsegmented', label: t('instances.unsegmented'), muted: true, visible: p.unsegmentedVisible, onVisible: p.onToggleUnsegmented,
    meta: p.unsegmentedCount != null ? fmt.integer(p.unsegmentedCount) : undefined,
  }

  return (
    <Panel title={t('instances.title')}
      subtitle={p.absorbedCount
        ? `${t.plural('instances.count', p.segments.length)} · ${t('instances.absorbed', { n: fmt.integer(p.absorbedCount) })}`
        : t.plural('instances.count', p.segments.length)}
      actions={anyListed ? (
        <>
          <IconButton size="sm" label={t('instances.showAll')} icon={<CheckSquare aria-hidden />} onClick={() => p.onSetAllVisible(true, listed)} />
          <IconButton size="sm" label={t('instances.hideAll')} icon={<Square aria-hidden />} onClick={() => p.onSetAllVisible(false, listed)} />
        </>
      ) : undefined}
      toolbar={<SearchInput size="sm" value={search} onChange={e => setSearch(e.target.value)} onClear={() => setSearch('')} placeholder={t('instances.search')} />}
      footer={
        <div className="stac-instances__footer">
          <Button size="sm" icon={<Crosshair aria-hidden />} onClick={p.onOpenSegmentation}>{t('instances.segmentation')}</Button>
          <Button size="sm" icon={<Wand2 aria-hidden />} onClick={p.onOpenAutosegment} title={t('instances.autosegmentHint')}>{t('instances.autosegment')}</Button>
          <Button size="sm" icon={<Puzzle aria-hidden />} onClick={p.onOpenMeshing} title={t('instances.meshingHint')}>{t('instances.meshing')}</Button>
        </div>
      }>
      <Stack gap={1}>
        {p.floorLevel.candidates.length > 0 && (
          <div className="stac-instances__floor">
            <Field inline label={t('instances.floorAtZero')} hint={undefined}>
              <Select<string> size="sm" value={p.floorLevel.selected != null ? String(p.floorLevel.selected) : ''} placeholder={p.floorLevel.selected == null ? t('instances.floorAuto') : undefined}
                onChange={v => { const iid = parseInt(v); if (!Number.isNaN(iid)) p.onFloorLevel(iid) }}
                options={p.floorLevel.candidates.map(c => ({ value: String(c.instance_id), label: `${c.label} ${c.instance_id}${c.height_m != null ? ` (${fmt.lengthText(c.height_m)})` : ''}` }))} />
            </Field>
          </div>
        )}
        {p.segments.length === 0 && (
          <EmptyState compact icon={<Tag aria-hidden />} title={t('instances.emptyTitle')} description={t('instances.emptyDesc')}
            action={<Button variant="primary" icon={<Crosshair aria-hidden />} onClick={p.onOpenSegmentation}>{t('instances.segmentation')}</Button>} />
        )}
        <Tree ariaLabel={t('instances.title')} nodes={unsegListed ? [...segNodes, unsegNode] : segNodes} dense
          selectedId={p.selectedSegmentId != null ? p.segments.find(s => s.id === p.selectedSegmentId)?.key ?? null : null}
          onSelect={n => p.onSelectSegment(n.data ? (n.data as SegmentInstance).id : null)} />
        {placed.length > 0 && (
          <Section title={t('instances.placedObjects', { n: placed.length })} flush>
            <Tree ariaLabel={t('instances.placedObjects', { n: placed.length })} dense
              onSelect={n => { const o = placed.find(x => `pobj-${x.id}` === n.id); if (o) p.onFocusItem('placed', o.id) }}
              nodes={placed.map(o => ({
              id: `pobj-${o.id}`, icon: <Box aria-hidden />, visible: o.visible, onVisible: v => p.onTogglePlaced(o.id, v),
              obb: p.placedObb.has(o.id), onObb: v => p.onTogglePlacedObb(o.id, v),
              label: editing === `pobj-${o.id}` ? (
                <Input size="sm" autoFocus value={editValue} onChange={e => setEditValue(e.target.value)} onClick={e => e.stopPropagation()}
                  onKeyDown={e => { if (e.key === 'Enter') commitPlaced(o); if (e.key === 'Escape') setEditing(null) }} onBlur={() => commitPlaced(o)} aria-label={t('instances.renameLabel')} />
              ) : o.name,
              actions: (
                <>
                  <IconButton size="sm" label={t('instances.rename')} icon={<Pencil aria-hidden />} onClick={() => { setEditing(`pobj-${o.id}`); setEditValue(o.name) }} />
                  <IconButton size="sm" variant="danger" label={t('instances.removePlacedHint')} icon={<Trash2 aria-hidden />} onClick={() => p.onRemovePlaced(o.id)} />
                </>
              ),
            }))} />
          </Section>
        )}
        {vols.length > 0 && (
          <Section title={t('instances.volumes', { n: vols.length })} flush>
            <Tree ariaLabel={t('instances.volumes', { n: vols.length })} dense
              onSelect={n => { const v = vols.find(x => `vol-${x.volume_id}` === n.id); if (v) p.onFocusVolume(v.volume_id) }}
              nodes={vols.map(v => ({
              id: `vol-${v.volume_id}`, icon: <Box aria-hidden />, label: v.name, meta: `#${v.volume_id}`,
              visible: !p.volumeHidden.has(v.volume_id), onVisible: on => p.onToggleVolume(v.volume_id, on),
              obb: p.volumeObb.has(v.volume_id), onObb: on => p.onToggleVolumeObb(v.volume_id, on),
              actions: <IconButton size="sm" variant="danger" label={t('instances.deleteVolume')} icon={<Trash2 aria-hidden />} onClick={() => p.onDeleteVolume(v.volume_id)} />,
            }))} />
          </Section>
        )}
        {shapes.length > 0 && (
          <Section title={t('instances.objectMeshes', { n: shapes.length })} flush>
            <Tree ariaLabel={t('instances.objectMeshes', { n: shapes.length })} dense
              onSelect={n => { const m = shapes.find(x => `shape-${x.folder}` === n.id); if (m) p.onFocusItem('shape', m.instanceId, m.instanceId) }}
              nodes={shapes.map(m => ({
              id: `shape-${m.folder}`, visible: m.visible, onVisible: v => p.onToggleShape(m, v),
              ...obbOf(m.instanceId),
              label: editing === `shape-${m.folder}` ? (
                <Input size="sm" autoFocus value={editValue} onChange={e => setEditValue(e.target.value)} onClick={e => e.stopPropagation()}
                  onKeyDown={e => { if (e.key === 'Enter') commitShape(m); if (e.key === 'Escape') setEditing(null) }} onBlur={() => commitShape(m)} aria-label={t('instances.renameLabel')} />
              ) : m.label,
              actions: (
                <>
                  <IconButton size="sm" label={m.wireframe ? t('instances.wireframeOff') : t('instances.wireframeOn')} icon={<Grid3x3 aria-hidden />}
                    onClick={() => p.onToggleShapeWireframe(m, !m.wireframe)} />
                  <IconButton size="sm" label={t('instances.rename')} icon={<Pencil aria-hidden />} onClick={() => { setEditing(`shape-${m.folder}`); setEditValue(m.label) }} />
                  <IconButton size="sm" variant="danger" label={t('instances.delete')} icon={<Trash2 aria-hidden />} onClick={() => p.onDeleteShape(m)} />
                </>
              ),
            }))} />
          </Section>
        )}
        {tsdfs.length > 0 && (
          <Section title={t('instances.meshes', { n: tsdfs.length })} flush>
            <Tree ariaLabel={t('instances.meshes', { n: tsdfs.length })} dense
              onSelect={n => { const m = tsdfs.find(x => `tsdf-${x.folder}` === n.id); if (m) p.onFocusItem('tsdf', m.folder, m.instanceId) }}
              nodes={tsdfs.map(m => ({
              id: `tsdf-${m.folder}`, label: m.label, meta: m.folder, visible: m.visible, onVisible: v => p.onToggleTsdf(m, v),
              ...obbOf(m.instanceId),
              actions: (
                <>
                  <IconButton size="sm" label={m.wireframe ? t('instances.wireframeOff') : t('instances.wireframeOn')} icon={<Grid3x3 aria-hidden />}
                    onClick={() => p.onToggleTsdfWireframe(m, !m.wireframe)} />
                  <IconButton size="sm" variant="danger" label={t('instances.deleteMeshHint', { folder: m.folder })} icon={<Trash2 aria-hidden />} onClick={() => p.onDeleteTsdf(m)} />
                </>
              ),
            }))} />
          </Section>
        )}
      </Stack>
    </Panel>
  )
}

/** memoized (USER 2026-10-16: "se pone muy lenta la UI al mover la nube con la lista de instancias
 *  abierta"): the App re-renders on every status update (visible points, fps) and this list's
 *  hundreds of rows re-rendered with it; its data props are state, its callbacks come stable from
 *  App.useStableCallbacks, so it renders only when the list itself changes */
export const InstancesPanel = memo(InstancesPanelImpl)
