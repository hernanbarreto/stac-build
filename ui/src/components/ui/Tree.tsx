/**
 * Tree — sessions / instances / scans / BIM: 16 px indent, chevrons, count
 * badge, visibility checkbox, categorical colour dot (§6). Selection is
 * --surface-4 + 2 px brand-orange left border; hover --surface-3.
 * Rows expose `actions` that appear on hover/focus and `onActivate` on
 * double-click / Enter.
 */
import { useState, type ReactNode } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { ColorDot } from './Badge'
import { Checkbox } from './Field'

export interface TreeNodeData<T = unknown> {
  id: string
  label: ReactNode
  title?: string
  meta?: ReactNode
  badge?: ReactNode
  icon?: ReactNode
  color?: string
  visible?: boolean
  onVisible?: (v: boolean) => void
  actions?: ReactNode
  /** render meta + actions on a second line (always visible) instead of inline */
  twoLine?: boolean
  children?: TreeNodeData<T>[]
  data?: T
  muted?: boolean
  defaultExpanded?: boolean
}

interface TreeProps<T> {
  nodes: TreeNodeData<T>[]
  selectedId?: string | null
  activeId?: string | null
  onSelect?: (node: TreeNodeData<T>) => void
  onActivate?: (node: TreeNodeData<T>) => void
  ariaLabel: string
  dense?: boolean
  className?: string
}

export function Tree<T>({ nodes, selectedId, activeId, onSelect, onActivate, ariaLabel, dense = false, className = '' }: TreeProps<T>) {
  return (
    <ul role="tree" aria-label={ariaLabel} className={`stac-tree ${dense ? 'stac-tree--dense' : ''} ${className}`.trim()}>
      {nodes.map(n => <TreeRow key={n.id} node={n} depth={0} selectedId={selectedId} activeId={activeId} onSelect={onSelect} onActivate={onActivate} />)}
    </ul>
  )
}

function TreeRow<T>({ node, depth, selectedId, activeId, onSelect, onActivate }: { node: TreeNodeData<T>; depth: number; selectedId?: string | null; activeId?: string | null; onSelect?: (n: TreeNodeData<T>) => void; onActivate?: (n: TreeNodeData<T>) => void }) {
  const [expanded, setExpanded] = useState(node.defaultExpanded ?? depth < 1)
  const hasChildren = !!node.children?.length
  const selected = node.id === selectedId
  const active = node.id === activeId
  return (
    <li role="treeitem" aria-expanded={hasChildren ? expanded : undefined} aria-selected={selected} className="stac-tree__item">
      <div
        className={`stac-tree__row ${selected ? 'stac-tree__row--selected' : ''} ${active ? 'stac-tree__row--active' : ''} ${node.muted ? 'stac-tree__row--muted' : ''}`.trim()}
        style={{ '--depth': depth } as React.CSSProperties}
        tabIndex={0}
        title={node.title}
        onClick={() => onSelect?.(node)}
        onDoubleClick={() => onActivate?.(node)}
        onKeyDown={e => {
          if (e.key === 'Enter') { e.preventDefault(); onActivate ? onActivate(node) : onSelect?.(node) }
          else if (e.key === ' ') { e.preventDefault(); onSelect?.(node) }
          else if (e.key === 'ArrowRight' && hasChildren) setExpanded(true)
          else if (e.key === 'ArrowLeft' && hasChildren) setExpanded(false)
        }}
      >
        {node.twoLine ? (
          <div className="stac-tree__lines">
            <div className="stac-tree__line">
              <span className="stac-tree__indent" aria-hidden />
              {hasChildren ? (
                <button type="button" className="stac-tree__chevron" tabIndex={-1} aria-hidden onClick={e => { e.stopPropagation(); setExpanded(v => !v) }}>
                  {expanded ? <ChevronDown /> : <ChevronRight />}
                </button>
              ) : <span className="stac-tree__chevron stac-tree__chevron--leaf" aria-hidden />}
              {node.color && <ColorDot color={node.color} />}
              {node.icon && <span className="stac-tree__icon" aria-hidden>{node.icon}</span>}
              <span className="stac-tree__label">{node.label}</span>
              {node.badge != null && <span className="stac-tree__badge">{node.badge}</span>}
            </div>
            <div className="stac-tree__line stac-tree__line--second">
              <span className="stac-tree__indent" aria-hidden />
              <span className="stac-tree__chevron stac-tree__chevron--leaf" aria-hidden />
              {node.meta && <span className="stac-tree__meta">{node.meta}</span>}
              {node.actions && <span className="stac-tree__actions stac-tree__actions--always" onClick={e => e.stopPropagation()} onDoubleClick={e => e.stopPropagation()}>{node.actions}</span>}
            </div>
          </div>
        ) : (<>
        <span className="stac-tree__indent" aria-hidden />
        {hasChildren ? (
          <button type="button" className="stac-tree__chevron" tabIndex={-1} aria-hidden onClick={e => { e.stopPropagation(); setExpanded(v => !v) }}>
            {expanded ? <ChevronDown /> : <ChevronRight />}
          </button>
        ) : <span className="stac-tree__chevron stac-tree__chevron--leaf" aria-hidden />}
        {node.onVisible && (
          <span className="stac-tree__check" onClick={e => e.stopPropagation()} onDoubleClick={e => e.stopPropagation()}>
            <Checkbox checked={node.visible ?? true} onChange={node.onVisible} aria-label={typeof node.label === 'string' ? node.label : undefined} />
          </span>
        )}
        {node.color && <ColorDot color={node.color} />}
        {node.icon && <span className="stac-tree__icon" aria-hidden>{node.icon}</span>}
        <span className="stac-tree__label">{node.label}</span>
        {node.meta && <span className="stac-tree__meta">{node.meta}</span>}
        {node.badge != null && <span className="stac-tree__badge">{node.badge}</span>}
        {node.actions && <span className="stac-tree__actions" onClick={e => e.stopPropagation()} onDoubleClick={e => e.stopPropagation()}>{node.actions}</span>}
        </>)}
      </div>
      {hasChildren && expanded && (
        <ul role="group" className="stac-tree__children">
          {node.children!.map(c => <TreeRow key={c.id} node={c} depth={depth + 1} selectedId={selectedId} activeId={activeId} onSelect={onSelect} onActivate={onActivate} />)}
        </ul>
      )}
    </li>
  )
}
