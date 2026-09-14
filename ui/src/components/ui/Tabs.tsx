/**
 * Tabs — 2 px brand-orange underline; active --text-1, inactive --text-2.
 * `closable` tabs (scan tabs over the viewport) get an inline close button.
 * Keyboard: arrows move, Home/End jump, Delete closes (when closable).
 */
import { X } from 'lucide-react'
import type { ReactNode } from 'react'
import { useT } from '../../i18n'

export interface TabItem<T extends string = string> {
  id: T
  label: ReactNode
  icon?: ReactNode
  badge?: ReactNode
  disabled?: boolean
  title?: string
  closable?: boolean
}

interface TabsProps<T extends string> {
  items: TabItem<T>[]
  value: T | null
  onChange: (id: T) => void
  onClose?: (id: T) => void
  ariaLabel: string
  size?: 'sm' | 'md'
  variant?: 'underline' | 'chrome'
  className?: string
  fill?: boolean
}

export function Tabs<T extends string>({ items, value, onChange, onClose, ariaLabel, size = 'md', variant = 'underline', className = '', fill = false }: TabsProps<T>) {
  const t = useT()
  const onKey = (e: React.KeyboardEvent, idx: number) => {
    const enabled = items.filter(i => !i.disabled)
    const pos = enabled.findIndex(i => i.id === items[idx].id)
    let next = -1
    if (e.key === 'ArrowRight') next = (pos + 1) % enabled.length
    else if (e.key === 'ArrowLeft') next = (pos - 1 + enabled.length) % enabled.length
    else if (e.key === 'Home') next = 0
    else if (e.key === 'End') next = enabled.length - 1
    else if (e.key === 'Delete' && onClose && items[idx].closable) { onClose(items[idx].id); return }
    if (next >= 0) { e.preventDefault(); onChange(enabled[next].id) }
  }
  return (
    <div role="tablist" aria-label={ariaLabel} className={`stac-tabs stac-tabs--${size} stac-tabs--${variant} ${fill ? 'stac-tabs--fill' : ''} ${className}`.trim()}>
      {items.map((item, idx) => {
        const active = item.id === value
        return (
          <div key={item.id} className={`stac-tab ${active ? 'stac-tab--active' : ''} ${item.disabled ? 'stac-tab--disabled' : ''}`.trim()}>
            <button
              role="tab"
              type="button"
              className="stac-tab__button"
              aria-selected={active}
              tabIndex={active ? 0 : -1}
              disabled={item.disabled}
              title={item.title}
              onClick={() => onChange(item.id)}
              onKeyDown={e => onKey(e, idx)}
            >
              {item.icon && <span className="stac-tab__icon" aria-hidden>{item.icon}</span>}
              <span className="stac-tab__label">{item.label}</span>
              {item.badge != null && <span className="stac-tab__badge">{item.badge}</span>}
            </button>
            {item.closable && onClose && (
              <button type="button" className="stac-tab__close" aria-label={t('common.closeTab')} onClick={e => { e.stopPropagation(); onClose(item.id) }}>
                <X aria-hidden />
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}
