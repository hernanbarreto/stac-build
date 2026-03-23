/**
 * BIM Navigator — Tree panel for exploring IFC model hierarchy.
 * Shows Project → Site → Building → Storey → Elements with
 * search, visibility toggles, opacity control, and selection highlighting.
 */
import { useState, useMemo, useCallback, useRef } from 'react'
import { Search } from 'lucide-react'
import type { IFCTreeNode, IFCLoadResult } from './IFCLoader'
import { countDescendants, collectMeshNames } from './IFCLoader'

export interface BIMNavigatorProps {
    models: IFCLoadResult[]
    userRole: string
    onToggleVisibility: (meshNames: string[], visible: boolean) => void
    onSelectElement: (meshNames: string[]) => void
    onSetOpacity: (meshNames: string[], opacity: number) => void
    onUploadIFC: (file: File) => void
    onDeleteIFC: (filename: string) => void
}

// ── Tree Node Component ─────────────────────────────────────────

interface TreeNodeProps {
    node: IFCTreeNode
    depth: number
    searchText: string
    hiddenNodes: Set<number>
    selectedNode: number | null
    nodeOpacities: Map<number, number>
    onToggleVisibility: (node: IFCTreeNode, visible: boolean) => void
    onSelect: (node: IFCTreeNode) => void
    onOpacityChange: (node: IFCTreeNode, opacity: number) => void
}

function matchesSearch(node: IFCTreeNode, text: string): boolean {
    const lower = text.toLowerCase()
    if (node.name.toLowerCase().includes(lower)) return true
    if (node.type.toLowerCase().includes(lower)) return true
    return node.children.some(c => matchesSearch(c, lower))
}

function TreeNode({ node, depth, searchText, hiddenNodes, selectedNode, nodeOpacities, onToggleVisibility, onSelect, onOpacityChange }: TreeNodeProps) {
    const [expanded, setExpanded] = useState(depth < 2)
    const hasChildren = node.children.length > 0
    const isHidden = hiddenNodes.has(node.expressID)
    const isSelected = selectedNode === node.expressID
    const childCount = countDescendants(node) - 1
    const opacity = nodeOpacities.get(node.expressID) ?? 1.0

    // Filter by search
    if (searchText && !matchesSearch(node, searchText)) return null

    const typeIcon = getTypeIcon(node.type)

    return (
        <div className="bim-tree-node">
            <div
                className={`bim-tree-row ${isSelected ? 'selected' : ''} ${isHidden ? 'hidden-node' : ''}`}
                style={{ paddingLeft: `${12 + depth * 16}px` }}
                onClick={() => onSelect(node)}
            >
                {/* Expand/collapse toggle */}
                <span
                    className={`bim-tree-toggle ${hasChildren ? 'has-children' : ''}`}
                    onClick={(e) => { e.stopPropagation(); if (hasChildren) setExpanded(!expanded) }}
                >
                    {hasChildren ? (expanded ? '▼' : '▶') : ''}
                </span>

                {/* Type icon */}
                <span className="bim-tree-icon">{typeIcon}</span>

                {/* Name + type + count */}
                <span className="bim-tree-label" title={`${node.type}::${node.name} (#${node.expressID})`}>
                    <span className="bim-tree-type">{node.type}</span>
                    <span className="bim-tree-separator">::</span>
                    <span className="bim-tree-name">{node.name}</span>
                    {childCount > 0 && <span className="bim-tree-count"> ({childCount})</span>}
                </span>

                {/* Opacity indicator — shows % if not 100% */}
                {opacity < 1.0 && (
                    <span className="bim-tree-opacity-badge" title={`${Math.round(opacity * 100)}% opacity`}>
                        {Math.round(opacity * 100)}%
                    </span>
                )}

                {/* Visibility toggle */}
                <span
                    className={`bim-tree-eye ${isHidden ? 'off' : ''}`}
                    onClick={(e) => { e.stopPropagation(); onToggleVisibility(node, isHidden) }}
                    title={isHidden ? 'Show' : 'Hide'}
                >
                    {isHidden ? '👁️‍🗨️' : '👁️'}
                </span>
            </div>

            {/* Opacity slider — shown when selected */}
            {isSelected && !isHidden && (
                <div className="bim-opacity-row" style={{ paddingLeft: `${28 + depth * 16}px` }}>
                    <span className="bim-opacity-label">Opacity</span>
                    <input
                        type="range"
                        className="bim-opacity-slider"
                        min="0"
                        max="100"
                        value={Math.round(opacity * 100)}
                        onChange={(e) => onOpacityChange(node, parseInt(e.target.value) / 100)}
                        onClick={(e) => e.stopPropagation()}
                    />
                    <span className="bim-opacity-value">{Math.round(opacity * 100)}%</span>
                </div>
            )}

            {/* Children */}
            {expanded && hasChildren && (
                <div className="bim-tree-children">
                    {node.children.map(child => (
                        <TreeNode
                            key={child.expressID}
                            node={child}
                            depth={depth + 1}
                            searchText={searchText}
                            hiddenNodes={hiddenNodes}
                            selectedNode={selectedNode}
                            nodeOpacities={nodeOpacities}
                            onToggleVisibility={onToggleVisibility}
                            onSelect={onSelect}
                            onOpacityChange={onOpacityChange}
                        />
                    ))}
                </div>
            )}
        </div>
    )
}

function getTypeIcon(type: string): string {
    if (type.includes('Project')) return '📋'
    if (type.includes('Site')) return '🌍'
    if (type.includes('Building') && !type.includes('Storey')) return '🏢'
    if (type.includes('Storey')) return '🏗️'
    if (type.includes('Wall')) return '🧱'
    if (type.includes('Slab')) return '⬜'
    if (type.includes('Roof')) return '🏠'
    if (type.includes('Door')) return '🚪'
    if (type.includes('Window')) return '🪟'
    if (type.includes('Stair')) return '🪜'
    if (type.includes('Column')) return '🏛️'
    if (type.includes('Beam')) return '📏'
    if (type.includes('Furnishing')) return '🪑'
    if (type.includes('Opening')) return '⬜'
    if (type.includes('Space')) return '📐'
    if (type.includes('Member')) return '🔩'
    if (type.includes('Plate')) return '🔲'
    if (type.includes('Curtain')) return '🪟'
    return '📦'
}

// ── Main BIM Navigator Component ────────────────────────────────

export default function BIMNavigator({
    models,
    userRole,
    onToggleVisibility,
    onSelectElement,
    onSetOpacity,
    onUploadIFC,
    onDeleteIFC,
}: BIMNavigatorProps) {
    const [searchText, setSearchText] = useState('')
    const [hiddenNodes, setHiddenNodes] = useState<Set<number>>(new Set())
    const [selectedNode, setSelectedNode] = useState<number | null>(null)
    const [nodeOpacities, setNodeOpacities] = useState<Map<number, number>>(new Map())
    const fileInputRef = useRef<HTMLInputElement>(null)
    const isAdmin = userRole === 'admin' || userRole === 'manager'

    const handleToggleVisibility = useCallback((node: IFCTreeNode, wasHidden: boolean) => {
        const meshNames = collectMeshNames(node)
        const newHidden = new Set(hiddenNodes)

        const toggleAll = (n: IFCTreeNode, hide: boolean) => {
            if (hide) newHidden.add(n.expressID)
            else newHidden.delete(n.expressID)
            n.children.forEach(c => toggleAll(c, hide))
        }

        if (wasHidden) {
            toggleAll(node, false)
            onToggleVisibility(meshNames, true)
        } else {
            toggleAll(node, true)
            onToggleVisibility(meshNames, false)
        }
        setHiddenNodes(newHidden)
    }, [hiddenNodes, onToggleVisibility])

    const handleSelect = useCallback((node: IFCTreeNode) => {
        setSelectedNode(prev => {
            const isDeselecting = prev === node.expressID
            if (isDeselecting) {
                onSelectElement([])
                return null
            } else {
                const meshNames = collectMeshNames(node)
                onSelectElement(meshNames)
                return node.expressID
            }
        })
    }, [onSelectElement])

    const handleOpacityChange = useCallback((node: IFCTreeNode, opacity: number) => {
        const meshNames = collectMeshNames(node)
        // Update local opacity state for all descendants
        const newOpacities = new Map(nodeOpacities)
        const setAll = (n: IFCTreeNode) => {
            newOpacities.set(n.expressID, opacity)
            n.children.forEach(c => setAll(c))
        }
        setAll(node)
        setNodeOpacities(newOpacities)
        // Apply in 3D
        onSetOpacity(meshNames, opacity)
    }, [nodeOpacities, onSetOpacity])

    const handleUpload = useCallback(() => {
        fileInputRef.current?.click()
    }, [])

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (file) {
            onUploadIFC(file)
            e.target.value = ''
        }
    }, [onUploadIFC])

    const totalElements = useMemo(() => {
        let count = 0
        for (const model of models) {
            for (const root of model.hierarchy) {
                count += countDescendants(root)
            }
        }
        return count
    }, [models])

    return (
        <div className="bim-navigator">
            <div className="panel-header">
                Model
                <span className="panel-header-badge">{totalElements}</span>
            </div>

            {/* Search */}
            <div className="bim-search">
                <span className="bim-search-icon"><Search size={12} /></span>
                <input
                    type="text"
                    className="bim-search-input"
                    placeholder="Search..."
                    value={searchText}
                    onChange={e => setSearchText(e.target.value)}
                />
                {searchText && (
                    <span className="bim-search-clear" onClick={() => setSearchText('')}>✕</span>
                )}
            </div>

            {/* Tree */}
            <div className="bim-tree">
                {models.length === 0 ? (
                    <div className="bim-empty">No BIM models loaded</div>
                ) : (
                    models.map((model) => (
                        <div key={model.filename} className="bim-model-group">
                            <div className="bim-model-header">
                                <span className="bim-model-icon">📄</span>
                                <span className="bim-model-name">{model.filename}</span>
                                {isAdmin && (
                                    <span
                                        className="bim-model-delete"
                                        onClick={() => onDeleteIFC(model.filename)}
                                        title="Delete model"
                                    >🗑️</span>
                                )}
                            </div>
                            {model.hierarchy.map(root => (
                                <TreeNode
                                    key={root.expressID}
                                    node={root}
                                    depth={1}
                                    searchText={searchText}
                                    hiddenNodes={hiddenNodes}
                                    selectedNode={selectedNode}
                                    nodeOpacities={nodeOpacities}
                                    onToggleVisibility={handleToggleVisibility}
                                    onSelect={handleSelect}
                                    onOpacityChange={handleOpacityChange}
                                />
                            ))}
                        </div>
                    ))
                )}
            </div>

            {/* Actions */}
            {isAdmin && (
                <div className="bim-actions">
                    <button className="bim-action-btn upload" onClick={handleUpload}>
                        + Load IFC
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept=".ifc"
                        style={{ display: 'none' }}
                        onChange={handleFileChange}
                    />
                </div>
            )}
        </div>
    )
}
