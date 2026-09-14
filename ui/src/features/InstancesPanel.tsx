/**
 * InstancesPanel — the "Instances" tab: visibility tree of segments (colour
 * dot, point count, rename, delete), the unsegmented remainder, placed
 * objects, generated meshes (Object / Mesh) and the floor-at-y=0 selector;
 * footer buttons open Segmentation, Meshing, Correction and Fusion.
 */
import { useState } from 'react'
import { Box, CheckSquare, Crosshair, Layers, Pencil, Puzzle, Square, Tag, Trash2, Wrench } from 'lucide-react'
import type { SegmentInstance } from '../components/Viewport'
import { Panel, Section, Stack } from '../components/ui/Panel'
import { Tree, type TreeNodeData } from '../components/ui/Tree'
import { IconButton } from '../components/ui/IconButton'
import { Button } from '../components/ui/Button'
import { Input, SearchInput, Select, Field } from '../components/ui/Field'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface MeshListItem { instanceId: number; label: string; folder: string; visible: boolean }
export interface PlacedObject { id: number; name: string; visible: boolean }
export interface FloorLevelState { candidates: { instance_id: number; label: string; height_m: number | null }[]; selected: number | null }

interface InstancesPanelProps {
  segments: SegmentInstance[]
  unsegmentedVisible: boolean
  unsegmentedCount: number | null
  floorLevel: FloorLevelState
  placedObjects: PlacedObject[]
  shapeMeshes: MeshListItem[]
  tsdfMeshes: MeshListItem[]
  selectedSegmentId: number | null
  canFuse: boolean
  onSelectSegment: (id: number | null) => void
  onSelectAll: () => void
  onDeselectAll: () => void
  onToggleSegment: (seg: SegmentInstance, visible: boolean) => void
  onRenameSegment: (seg: SegmentInstance, label: string) => Promise<void>
  onDeleteSegment: (seg: SegmentInstance) => void
  onToggleUnsegmented: (visible: boolean) => void
  onFloorLevel: (instanceId: number) => void
  onTogglePlaced: (id: number, visible: boolean) => void
  onRemovePlaced: (id: number) => void
  onToggleShape: (m: MeshListItem, visible: boolean) => void
  onToggleTsdf: (m: MeshListItem, visible: boolean) => void
  onDeleteTsdf: (m: MeshListItem) => void
  onOpenSegmentation: () => void
  onOpenMeshing: () => void
  onOpenCorrection: () => void
  onOpenFuse: () => void
}

export function InstancesPanel(p: InstancesPanelProps) {
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

  const segNodes: TreeNodeData<SegmentInstance>[] = p.segments
    .filter(s => !search || s.label.toLowerCase().includes(search.toLowerCase()))
    .map(seg => ({
      id: seg.key,
      data: seg,
      color: seg.color,
      visible: seg.visible,
      onVisible: v => p.onToggleSegment(seg, v),
      label: editing === seg.key ? (
        <Input size="sm" autoFocus value={editValue} onChange={e => setEditValue(e.target.value)} onClick={e => e.stopPropagation()}
          onKeyDown={e => { if (e.key === 'Enter') commit(seg); if (e.key === 'Escape') setEditing(null) }} onBlur={() => commit(seg)} aria-label={t('instances.renameLabel')} />
      ) : seg.label,
      meta: fmt.integer(seg.totalPoints),
      muted: seg.excluded,
      actions: (
        <>
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
    <Panel title={t('instances.title')} subtitle={t.plural('instances.count', p.segments.length)}
      actions={p.segments.length > 0 ? (
        <>
          <IconButton size="sm" label={t('instances.showAll')} icon={<CheckSquare aria-hidden />} onClick={p.onSelectAll} />
          <IconButton size="sm" label={t('instances.hideAll')} icon={<Square aria-hidden />} onClick={p.onDeselectAll} />
        </>
      ) : undefined}
      toolbar={<SearchInput size="sm" value={search} onChange={e => setSearch(e.target.value)} onClear={() => setSearch('')} placeholder={t('instances.search')} />}
      footer={
        <div className="stac-instances__footer">
          <Button size="sm" icon={<Crosshair aria-hidden />} onClick={p.onOpenSegmentation}>{t('instances.segmentation')}</Button>
          <Button size="sm" icon={<Puzzle aria-hidden />} onClick={p.onOpenMeshing} title={t('instances.meshingHint')}>{t('instances.meshing')}</Button>
          <Button size="sm" icon={<Wrench aria-hidden />} onClick={p.onOpenCorrection} title={t('instances.correctionHint')}>{t('instances.correction')}</Button>
          <Button size="sm" icon={<Layers aria-hidden />} disabled={!p.canFuse} onClick={p.onOpenFuse} title={p.canFuse ? t('instances.fuseHint') : t('instances.fuseNeedsTwo')}>{t('instances.fuse')}</Button>
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
        <Tree ariaLabel={t('instances.title')} nodes={[...segNodes, unsegNode]} dense
          selectedId={p.selectedSegmentId != null ? p.segments.find(s => s.id === p.selectedSegmentId)?.key ?? null : null}
          onSelect={n => p.onSelectSegment(n.data ? (n.data as SegmentInstance).id : null)} />
        {p.placedObjects.length > 0 && (
          <Section title={t('instances.placedObjects', { n: p.placedObjects.length })} flush>
            <Tree ariaLabel={t('instances.placedObjects', { n: p.placedObjects.length })} dense nodes={p.placedObjects.map(o => ({
              id: `pobj-${o.id}`, label: o.name, icon: <Box aria-hidden />, visible: o.visible, onVisible: v => p.onTogglePlaced(o.id, v),
              actions: <IconButton size="sm" variant="danger" label={t('instances.removePlacedHint')} icon={<Trash2 aria-hidden />} onClick={() => p.onRemovePlaced(o.id)} />,
            }))} />
          </Section>
        )}
        {p.shapeMeshes.length > 0 && (
          <Section title={t('instances.objectMeshes', { n: p.shapeMeshes.length })} flush>
            <Tree ariaLabel={t('instances.objectMeshes', { n: p.shapeMeshes.length })} dense nodes={p.shapeMeshes.map(m => ({
              id: `shape-${m.folder}`, label: m.label, visible: m.visible, onVisible: v => p.onToggleShape(m, v),
            }))} />
          </Section>
        )}
        {p.tsdfMeshes.length > 0 && (
          <Section title={t('instances.meshes', { n: p.tsdfMeshes.length })} flush>
            <Tree ariaLabel={t('instances.meshes', { n: p.tsdfMeshes.length })} dense nodes={p.tsdfMeshes.map(m => ({
              id: `tsdf-${m.folder}`, label: m.label, meta: m.folder, visible: m.visible, onVisible: v => p.onToggleTsdf(m, v),
              actions: <IconButton size="sm" variant="danger" label={t('instances.deleteMeshHint', { folder: m.folder })} icon={<Trash2 aria-hidden />} onClick={() => p.onDeleteTsdf(m)} />,
            }))} />
          </Section>
        )}
      </Stack>
    </Panel>
  )
}
