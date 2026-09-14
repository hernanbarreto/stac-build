/**
 * Dialog — modal with a scrim, focus trap (Tab cycles inside), Escape closes,
 * initial focus on the cancel button when the dialog is destructive (§6).
 * `nonBlocking` renders without scrim and lets the viewport receive input
 * (the correction verdict must be judged by orbiting the cloud).
 */
import { useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { X } from 'lucide-react'
import { Button } from './Button'
import { useT } from '../../i18n'

export interface DialogProps {
  open: boolean
  title: ReactNode
  subtitle?: ReactNode
  icon?: ReactNode
  tone?: 'default' | 'danger' | 'warn' | 'measure'
  size?: 'sm' | 'md' | 'lg' | 'full'
  children?: ReactNode
  footer?: ReactNode
  onClose?: () => void
  closeLabel?: string
  nonBlocking?: boolean
  /** anchor a non-blocking dialog to the top instead of the centre */
  position?: 'center' | 'top' | 'right'
  className?: string
  busy?: boolean
}

export function Dialog({ open, title, subtitle, icon, tone = 'default', size = 'md', children, footer, onClose, closeLabel, nonBlocking = false, position = 'center', className = '', busy = false }: DialogProps) {
  const t = useT()
  const ref = useRef<HTMLDivElement>(null)
  const lastActive = useRef<Element | null>(null)

  useEffect(() => {
    if (!open) return
    lastActive.current = document.activeElement
    const node = ref.current
    const focusables = () => Array.from(node?.querySelectorAll<HTMLElement>('button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])') ?? [])
    const initial = node?.querySelector<HTMLElement>('[data-autofocus]') ?? focusables()[0]
    initial?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onClose && !busy) { e.stopPropagation(); onClose() }
      if (e.key === 'Tab' && !nonBlocking) {
        const els = focusables()
        if (!els.length) return
        const first = els[0], last = els[els.length - 1]
        if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
        else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
      }
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      const prev = lastActive.current as HTMLElement | null
      prev?.focus?.()
    }
  }, [open, onClose, nonBlocking, busy])

  if (!open) return null
  const dialog = (
    <div ref={ref} role="dialog" aria-modal={!nonBlocking} aria-labelledby="stac-dialog-title" className={`stac-dialog stac-dialog--${size} stac-dialog--${tone} ${className}`.trim()}>
      <header className="stac-dialog__header">
        {icon && <span className="stac-dialog__icon">{icon}</span>}
        <div className="stac-dialog__titles">
          <h2 id="stac-dialog-title" className="stac-dialog__title">{title}</h2>
          {subtitle && <div className="stac-dialog__subtitle">{subtitle}</div>}
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" className="stac-dialog__close" aria-label={closeLabel ?? t('common.close')} onClick={onClose} disabled={busy} icon={<X aria-hidden />} />
        )}
      </header>
      <div className="stac-dialog__body">{children}</div>
      {footer && <footer className="stac-dialog__footer">{footer}</footer>}
    </div>
  )
  return createPortal(
    <div className={`stac-dialog-layer stac-dialog-layer--${position} ${nonBlocking ? 'stac-dialog-layer--passthrough' : ''}`.trim()}
      onMouseDown={e => { if (!nonBlocking && e.target === e.currentTarget && onClose && !busy) onClose() }}>
      {dialog}
    </div>,
    document.body,
  )
}

/** Confirmation dialog: destructive variant uses a danger button and focuses Cancel first. */
export function ConfirmDialog({ open, title, message, variant = 'confirm', confirmLabel, cancelLabel, onConfirm, onCancel, icon }: {
  open: boolean
  title?: ReactNode
  message: ReactNode
  variant?: 'confirm' | 'alert' | 'danger'
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel: () => void
  icon?: ReactNode
}) {
  const t = useT()
  const confirmText = confirmLabel ?? (variant === 'alert' ? t('common.ok') : variant === 'danger' ? t('common.delete') : t('common.confirm'))
  return (
    <Dialog open={open} size="sm" tone={variant === 'danger' ? 'danger' : 'default'} icon={icon}
      title={title ?? (variant === 'danger' ? t('dialog.confirmDeleteTitle') : variant === 'alert' ? t('dialog.noticeTitle') : t('dialog.confirmTitle'))}
      onClose={onCancel}
      footer={
        <>
          {variant !== 'alert' && (
            <Button variant="secondary" onClick={onCancel} {...(variant === 'danger' ? { 'data-autofocus': true } : {})}>{cancelLabel ?? t('common.cancel')}</Button>
          )}
          <Button variant={variant === 'danger' ? 'danger' : 'primary'} onClick={onConfirm} {...(variant !== 'danger' ? { 'data-autofocus': true } : {})}>{confirmText}</Button>
        </>
      }>
      <p className="stac-dialog__message">{message}</p>
    </Dialog>
  )
}
