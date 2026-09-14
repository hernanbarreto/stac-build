/**
 * Drawer — right-side sheet for long documents (acta, reports, ledgers).
 * Non-modal by default so the viewport stays navigable; Escape closes.
 */
import { useEffect, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { IconButton } from './IconButton'
import { useT } from '../../i18n'

interface DrawerProps {
  open: boolean
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  onClose: () => void
  children?: ReactNode
  footer?: ReactNode
  wide?: boolean
}

export function Drawer({ open, title, subtitle, actions, onClose, children, footer, wide = false }: DrawerProps) {
  const t = useT()
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null
  return createPortal(
    <aside role="complementary" className={`stac-drawer ${wide ? 'stac-drawer--wide' : ''}`.trim()} aria-label={typeof title === 'string' ? title : undefined}>
      <header className="stac-drawer__header">
        <div className="stac-drawer__titles">
          <h2 className="stac-drawer__title">{title}</h2>
          {subtitle && <div className="stac-drawer__subtitle">{subtitle}</div>}
        </div>
        {actions}
        <IconButton label={t('common.close')} icon={<X aria-hidden />} onClick={onClose} />
      </header>
      <div className="stac-drawer__body">{children}</div>
      {footer && <footer className="stac-drawer__footer">{footer}</footer>}
    </aside>,
    document.body,
  )
}
