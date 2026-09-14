/**
 * BIM Navigator — tree panel for exploring the IFC hierarchy
 * (Project -> Site -> Building -> Storey -> Elements) with search, visibility
 * toggles, opacity control and selection highlighting. Built on the
 * catalog Tree; type icons come from lucide by IFC class.
 */
import { useState, useMemo, useCallback, useRef, type ReactNode } from 'react'
import { Building2, Box, Columns3, DoorOpen, Eye, EyeOff, FileBox, Globe, Layers, Minus, Square, Trash2, Upload, Wrench, AppWindow, Armchair, Home, Grid2X2, Landmark, Ruler } from 'lucide-react'
import type { IFCTreeNode, IFCLoadResult } from './IFCLoader'
import { countDescendants, collectMeshNames } from './IFCLoader'
import { Panel, Section } from './ui/Panel'
import { Tree, type TreeNodeData } from './ui/Tree'
import { IconButton } from './ui/IconButton'
import { Button } from './ui/Button'
import { SearchInput, Slider } from './ui/Field'
import { Badge } from './ui/Badge'
import { EmptyState } from './ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface BIMNavigatorProps {
  models: IFCLoadResult[]
  userRole: string
  onToggleVisibility: (meshNames: string[], visible: boolean) => void
  onSelectElement: (meshNames: string[]) => void
  onSetOpacity: (meshNames: string[], opacity: number) => void
  onUploadIFC: (file: File) => void
  onDeleteIFC: (filename: string) => void
}

function matchesSearch(node: IFCTreeNode, text: string): boolean {
  const lower = text.toLowerCase()
  if (node.name.toLowerCase().includes(lower)) return true
  if (node.type.toLowerCase().includes(lower)) return true
  return node.children.some(c => matchesSearch(c, lower))
}

function typeIcon(type: string): ReactNode {
  if (type.includes('Project')) return <FileBox aria-hidden />
  if (type.includes('Site')) return <Globe aria-hidden />
  if (type.includes('Building') && !type.includes('Storey')) return <Building2 aria-hidden />
  if (type.includes('Storey')) return <Layers aria-hidden />
  if (type.includes('Wall')) return <Square aria-hidden />
  if (type.includes('Slab')) return <Minus aria-hidden />
  if (type.includes('Roof')) return <Home aria-hidden />
  if (type.includes('Door')) return <DoorOpen aria-hidden />
  if (type.includes('Window') || type.includes('Curtain')) return <AppWindow aria-hidden />
  if (type.includes('Stair')) return <Ruler aria-hidden />
  if (type.includes('Column')) return <Columns3 aria-hidden />
  if (type.includes('Beam')) return <Ruler aria-hidden />
  if (type.includes('Furnishing')) return <Armchair aria-hidden />
  if (type.includes('Opening') || type.includes('Space')) return <Grid2X2 aria-hidden />
  if (type.includes('Member') || type.includes('Plate')) return <Wrench aria-hidden />
  if (type.includes('Landmark')) return <Landmark aria-hidden />
  return <Box aria-hidden />
}

export default function BIMNavigator({ models, userRole, onToggleVisibility, onSelectElement, onSetOpacity, onUploadIFC, onDeleteIFC }: BIMNavigatorProps) {
  const t = useT()
  const fmt = useFmt()
  const [searchText, setSearchText] = useState('')
  const [hiddenNodes, setHiddenNodes] = useState<Set<number>>(new Set())
  const [selectedNode, setSelectedNode] = useState<IFCTreeNode | null>(null)
  const [nodeOpacities, setNodeOpacities] = useState<Map<number, number>>(new Map())
  const fileInputRef = useRef<HTMLInputElement>(null)
  const isAdmin = userRole === 'admin' || userRole === 'manager'

  const handleToggleVisibility = useCallback((node: IFCTreeNode, visible: boolean) => {
    const meshNames = collectMeshNames(node)
    const newHidden = new Set(hiddenNodes)
    const toggleAll = (n: IFCTreeNode, hide: boolean) => {
      if (hide) newHidden.add(n.expressID); else newHidden.delete(n.expressID)
      n.children.forEach(c => toggleAll(c, hide))
    }
    toggleAll(node, !visible)
    onToggleVisibility(meshNames, visible)
    setHiddenNodes(newHidden)
  }, [hiddenNodes, onToggleVisibility])

  const handleSelect = useCallback((node: IFCTreeNode) => {
    setSelectedNode(prev => {
      if (prev?.expressID === node.expressID) { onSelectElement([]); return null }
      onSelectElement(collectMeshNames(node))
      return node
    })
  }, [onSelectElement])

  const handleOpacityChange = useCallback((node: IFCTreeNode, opacity: number) => {
    const meshNames = collectMeshNames(node)
    const newOpacities = new Map(nodeOpacities)
    const setAll = (n: IFCTreeNode) => { newOpacities.set(n.expressID, opacity); n.children.forEach(c => setAll(c)) }
    setAll(node)
    setNodeOpacities(newOpacities)
    onSetOpacity(meshNames, opacity)
  }, [nodeOpacities, onSetOpacity])

  const totalElements = useMemo(() => models.reduce((count, model) => count + model.hierarchy.reduce((c, root) => c + countDescendants(root), 0), 0), [models])

  const toTreeNode = useCallback((node: IFCTreeNode, depth: number): TreeNodeData<IFCTreeNode> | null => {
    if (searchText && !matchesSearch(node, searchText)) return null
    const isHidden = hiddenNodes.has(node.expressID)
    const childCount = countDescendants(node) - 1
    const opacity = nodeOpacities.get(node.expressID) ?? 1
    return {
      id: String(node.expressID),
      data: node,
      icon: typeIcon(node.type),
      label: <><span className="stac-bim__type">{node.type.replace('Ifc', '')}</span> {node.name}</>,
      title: `${node.type} ${node.name} (${node.expressID})`,
      badge: childCount > 0 ? fmt.integer(childCount) : undefined,
      meta: opacity < 1 ? fmt.percent(opacity) : undefined,
      muted: isHidden,
      defaultExpanded: depth < 2,
      actions: <IconButton size="sm" label={isHidden ? t('bim.show') : t('bim.hide')} icon={isHidden ? <EyeOff aria-hidden /> : <Eye aria-hidden />} onClick={() => handleToggleVisibility(node, isHidden)} />,
      children: node.children.map(c => toTreeNode(c, depth + 1)).filter((n): n is TreeNodeData<IFCTreeNode> => n !== null),
    }
  }, [searchText, hiddenNodes, nodeOpacities, handleToggleVisibility, t, fmt])

  const selectedOpacity = selectedNode ? nodeOpacities.get(selectedNode.expressID) ?? 1 : 1

  return (
    <Panel title={t('bim.title')} subtitle={totalElements > 0 ? t.plural('bim.elements', totalElements) : undefined}
      toolbar={<SearchInput size="sm" value={searchText} onChange={e => setSearchText(e.target.value)} onClear={() => setSearchText('')} placeholder={t('bim.search')} />}
      isEmpty={models.length === 0}
      empty={<EmptyState icon={<Building2 aria-hidden />} title={t('bim.emptyTitle')} description={isAdmin ? t('bim.emptyDescAdmin') : t('bim.emptyDesc')}
        action={isAdmin ? <Button variant="primary" icon={<Upload aria-hidden />} onClick={() => fileInputRef.current?.click()}>{t('bim.loadIfc')}</Button> : undefined} />}
      footer={isAdmin ? (
        <>
          <Button block icon={<Upload aria-hidden />} onClick={() => fileInputRef.current?.click()}>{t('bim.loadIfc')}</Button>
          <input ref={fileInputRef} type="file" accept=".ifc" className="stac-hidden" onChange={e => { const f = e.target.files?.[0]; if (f) { onUploadIFC(f); e.target.value = '' } }} />
        </>
      ) : undefined}>
      {selectedNode && !hiddenNodes.has(selectedNode.expressID) && (
        <div className="stac-bim__opacity">
          <Slider min={0} max={100} step={1} value={Math.round(selectedOpacity * 100)} onChange={v => handleOpacityChange(selectedNode, v / 100)} label={t('bim.opacity')} format={v => `${v} %`} />
        </div>
      )}
      {models.map(model => (
        <Section key={model.filename} flush title={<span className="stac-mono">{model.filename}</span>}
          actions={isAdmin ? <IconButton size="sm" variant="danger" label={t('bim.deleteModel')} icon={<Trash2 aria-hidden />} onClick={() => onDeleteIFC(model.filename)} /> : <Badge size="sm">{t('bim.model')}</Badge>}>
          <Tree ariaLabel={model.filename} dense nodes={model.hierarchy.map(r => toTreeNode(r, 1)).filter((n): n is TreeNodeData<IFCTreeNode> => n !== null)}
            selectedId={selectedNode ? String(selectedNode.expressID) : null} onSelect={n => n.data && handleSelect(n.data)} />
        </Section>
      ))}
    </Panel>
  )
}
