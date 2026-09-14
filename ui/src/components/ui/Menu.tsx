/**
 * Menu — dropdown / context menu: items with icon, label, shortcut (<Kbd>),
 * checked state, separators, disabled. Opens below/right of its trigger,
 * closes on outside click and Escape; arrows move, Enter activates.
 */
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Check, ChevronDown } from 'lucide-react'
import { Kbd } from './Kbd'

export type MenuEntry =
  | { type?: 'item'; id: string; label: ReactNode; icon?: ReactNode; shortcut?: string; checked?: boolean; disabled?: boolean; danger?: boolean; onSelect: () => void; hint?: ReactNode }
  | { type: 'separator'; id: string }
  | { type: 'header'; id: string; label: ReactNode }
  | { type: 'custom'; id: string; render: () => ReactNode }

interface MenuProps {
  entries: MenuEntry[]
  open: boolean
  onClose: () => void
  anchor: { x: number; y: number } | DOMRect | null
  align?: 'left' | 'right'
  ariaLabel: string
  minWidth?: 'sm' | 'md' | 'lg'
}

export function Menu({ entries, open, onClose, anchor, align = 'left', ariaLabel, minWidth = 'md' }: MenuProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [focusIdx, setFocusIdx] = useState(-1)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) onClose() }
    const onKey = (e: KeyboardEvent) => {
      const items = entries.filter(en => (!en.type || en.type === 'item') && !('disabled' in en && en.disabled))
      if (e.key === 'Escape') { e.stopPropagation(); onClose() }
      else if (e.key === 'ArrowDown') { e.preventDefault(); setFocusIdx(i => (i + 1) % items.length) }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setFocusIdx(i => (i - 1 + items.length) % items.length) }
      else if (e.key === 'Enter' && focusIdx >= 0) {
        const it = items[focusIdx]
        if (it && (!it.type || it.type === 'item')) { it.onSelect(); onClose() }
      }
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('mousedown', onDown); document.removeEventListener('keydown', onKey) }
  }, [open, onClose, entries, focusIdx])

  useEffect(() => { if (!open) setFocusIdx(-1) }, [open])

  if (!open || !anchor) return null
  const isRect = 'width' in anchor
  const x = isRect ? (align === 'right' ? (anchor as DOMRect).right : (anchor as DOMRect).left) : anchor.x
  const y = isRect ? (anchor as DOMRect).bottom : anchor.y
  let itemCounter = -1
  return createPortal(
    <div ref={ref} role="menu" aria-label={ariaLabel}
      className={`stac-menu stac-menu--${align} stac-menu--${minWidth}`}
      style={{ '--menu-x': `${x}px`, '--menu-y': `${y}px` } as React.CSSProperties}>
      {entries.map(en => {
        if (en.type === 'separator') return <div key={en.id} role="separator" className="stac-menu__sep" />
        if (en.type === 'header') return <div key={en.id} className="stac-menu__header">{en.label}</div>
        if (en.type === 'custom') return <div key={en.id} className="stac-menu__custom" onClick={e => e.stopPropagation()}>{en.render()}</div>
        if (!en.disabled) itemCounter++
        const focused = !en.disabled && itemCounter === focusIdx
        return (
          <button key={en.id} role="menuitemcheckbox" aria-checked={en.checked ?? undefined} type="button" disabled={en.disabled}
            className={`stac-menu__item ${focused ? 'stac-menu__item--focus' : ''} ${en.danger ? 'stac-menu__item--danger' : ''}`.trim()}
            onClick={() => { en.onSelect(); onClose() }}>
            <span className="stac-menu__icon" aria-hidden>{en.icon}</span>
            <span className="stac-menu__label">{en.label}</span>
            {en.hint && <span className="stac-menu__hint">{en.hint}</span>}
            {en.checked && <Check className="stac-menu__check" aria-hidden />}
            {en.shortcut && <Kbd combo={en.shortcut} className="stac-menu__kbd" />}
          </button>
        )
      })}
    </div>,
    document.body,
  )
}

/** Trigger button + Menu pair for the header menu bar and dropdown selectors. */
export function MenuButton({ label, entries, ariaLabel, chevron = false, variant = 'bar', align = 'left', minWidth, icon, className = '', disabled }: {
  label: ReactNode
  entries: MenuEntry[]
  ariaLabel: string
  chevron?: boolean
  variant?: 'bar' | 'select'
  align?: 'left' | 'right'
  minWidth?: 'sm' | 'md' | 'lg'
  icon?: ReactNode
  className?: string
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const btn = useRef<HTMLButtonElement>(null)
  return (
    <>
      <button ref={btn} type="button" disabled={disabled} aria-haspopup="menu" aria-expanded={open}
        className={`stac-menubtn stac-menubtn--${variant} ${open ? 'stac-menubtn--open' : ''} ${className}`.trim()}
        onClick={() => setOpen(o => !o)}>
        {icon && <span className="stac-menubtn__icon" aria-hidden>{icon}</span>}
        <span className="stac-menubtn__label">{label}</span>
        {chevron && <ChevronDown className="stac-menubtn__chevron" aria-hidden />}
      </button>
      <Menu open={open} onClose={() => setOpen(false)} entries={entries} anchor={btn.current?.getBoundingClientRect() ?? null} align={align} ariaLabel={ariaLabel} minWidth={minWidth} />
    </>
  )
}
