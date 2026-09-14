/**
 * Table — dense, sticky header, sortable columns, 28/36 px rows, numbers
 * right-aligned in mono, row selection = --surface-4 + 2 px brand-orange
 * left border, integrated empty state, simple windowing above 500 rows (§6).
 */
import { useMemo, useRef, useState, type ReactNode } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { EmptyState } from './EmptyState'

export interface Column<T> {
  id: string
  header: ReactNode
  cell: (row: T) => ReactNode
  align?: 'left' | 'right' | 'center'
  mono?: boolean
  width?: string
  sortValue?: (row: T) => number | string | null | undefined
}

interface TableProps<T> {
  columns: Column<T>[]
  rows: T[]
  rowKey: (row: T, index: number) => string | number
  selectedKey?: string | number | null
  onRowClick?: (row: T) => void
  emptyTitle?: ReactNode
  emptyDescription?: ReactNode
  emptyAction?: ReactNode
  caption?: string
  className?: string
  maxHeight?: boolean
  initialSort?: { id: string; dir: 'asc' | 'desc' }
}

const VIRTUAL_THRESHOLD = 500
const OVERSCAN = 20

export function Table<T>({ columns, rows, rowKey, selectedKey, onRowClick, emptyTitle, emptyDescription, emptyAction, caption, className = '', maxHeight = true, initialSort }: TableProps<T>) {
  const [sort, setSort] = useState<{ id: string; dir: 'asc' | 'desc' } | null>(initialSort ?? null)
  const [scrollTop, setScrollTop] = useState(0)
  const [rowH, setRowH] = useState(28)
  const bodyRef = useRef<HTMLDivElement>(null)

  const sorted = useMemo(() => {
    if (!sort) return rows
    const col = columns.find(c => c.id === sort.id)
    if (!col?.sortValue) return rows
    const sv = col.sortValue
    return [...rows].sort((a, b) => {
      const va = sv(a), vb = sv(b)
      if (va == null && vb == null) return 0
      if (va == null) return 1
      if (vb == null) return -1
      const cmp = typeof va === 'number' && typeof vb === 'number' ? va - vb : String(va).localeCompare(String(vb))
      return sort.dir === 'asc' ? cmp : -cmp
    })
  }, [rows, sort, columns])

  const virtual = sorted.length > VIRTUAL_THRESHOLD
  const viewportH = bodyRef.current?.clientHeight ?? 400
  const first = virtual ? Math.max(0, Math.floor(scrollTop / rowH) - OVERSCAN) : 0
  const last = virtual ? Math.min(sorted.length, Math.ceil((scrollTop + viewportH) / rowH) + OVERSCAN) : sorted.length
  const visible = sorted.slice(first, last)

  const toggleSort = (id: string) => {
    setSort(s => (s?.id === id ? (s.dir === 'asc' ? { id, dir: 'desc' } : null) : { id, dir: 'asc' }))
  }

  return (
    <div className={`stac-table ${maxHeight ? 'stac-table--scroll' : ''} ${className}`.trim()} ref={bodyRef}
      onScroll={virtual ? e => { setScrollTop(e.currentTarget.scrollTop); const r = e.currentTarget.querySelector<HTMLElement>('.stac-table__row'); if (r && r.offsetHeight) setRowH(r.offsetHeight) } : undefined}>
      <table className="stac-table__el">
        {caption && <caption className="stac-visually-hidden">{caption}</caption>}
        <thead className="stac-table__head">
          <tr>
            {columns.map(c => (
              <th key={c.id} scope="col" className={`stac-table__th stac-table__th--${c.align ?? 'left'} ${c.sortValue ? 'stac-table__th--sortable' : ''}`.trim()}
                style={{ '--col-w': c.width ?? 'auto' } as React.CSSProperties}
                aria-sort={sort?.id === c.id ? (sort.dir === 'asc' ? 'ascending' : 'descending') : undefined}>
                {c.sortValue ? (
                  <button type="button" className="stac-table__sort" onClick={() => toggleSort(c.id)}>
                    <span>{c.header}</span>
                    {sort?.id === c.id && (sort.dir === 'asc' ? <ChevronUp aria-hidden /> : <ChevronDown aria-hidden />)}
                  </button>
                ) : c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {virtual && first > 0 && <tr className="stac-table__spacer"><td className="stac-table__spacer-cell" colSpan={columns.length} style={{ '--spacer-h': `${first * rowH}px` } as React.CSSProperties} /></tr>}
          {visible.map((row, i) => {
            const key = rowKey(row, first + i)
            const selected = selectedKey != null && key === selectedKey
            return (
              <tr key={key} className={`stac-table__row ${selected ? 'stac-table__row--selected' : ''} ${onRowClick ? 'stac-table__row--clickable' : ''}`.trim()}
                onClick={onRowClick ? () => onRowClick(row) : undefined} aria-selected={selected || undefined}
                tabIndex={onRowClick ? 0 : undefined}
                onKeyDown={onRowClick ? e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onRowClick(row) } } : undefined}>
                {columns.map(c => (
                  <td key={c.id} className={`stac-table__td stac-table__td--${c.align ?? 'left'} ${c.mono ? 'stac-mono' : ''}`.trim()}>{c.cell(row)}</td>
                ))}
              </tr>
            )
          })}
          {virtual && last < sorted.length && <tr className="stac-table__spacer"><td className="stac-table__spacer-cell" colSpan={columns.length} style={{ '--spacer-h': `${(sorted.length - last) * rowH}px` } as React.CSSProperties} /></tr>}
        </tbody>
      </table>
      {rows.length === 0 && emptyTitle && (
        <EmptyState compact title={emptyTitle} description={emptyDescription} action={emptyAction} />
      )}
    </div>
  )
}
