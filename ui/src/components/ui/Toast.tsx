/**
 * Toast — bottom-right, 4 s, with an optional action ("Undo") (§6).
 * `useToast()` gives `notify(message, { tone, action })`. Errors stay until
 * dismissed; successes auto-hide. Mounted once by <ToastProvider>.
 */
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react'
import { Button } from './Button'
import { useT } from '../../i18n'

export type ToastTone = 'info' | 'ok' | 'warn' | 'err'

export interface ToastOptions {
  tone?: ToastTone
  title?: ReactNode
  action?: { label: string; onClick: () => void }
  durationMs?: number
}

interface ToastItem extends ToastOptions {
  id: number
  message: ReactNode
}

interface ToastContextValue {
  notify: (message: ReactNode, opts?: ToastOptions) => number
  dismiss: (id: number) => void
}

const ToastContext = createContext<ToastContextValue>({ notify: () => 0, dismiss: () => {} })
const DEFAULT_MS = 4000
const ICONS: Record<ToastTone, ReactNode> = {
  info: <Info aria-hidden />, ok: <CheckCircle2 aria-hidden />, warn: <AlertTriangle aria-hidden />, err: <XCircle aria-hidden />,
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const seq = useRef(0)
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>())
  const t = useT()

  const dismiss = useCallback((id: number) => {
    setItems(list => list.filter(i => i.id !== id))
    const tm = timers.current.get(id)
    if (tm) { clearTimeout(tm); timers.current.delete(id) }
  }, [])

  const notify = useCallback((message: ReactNode, opts: ToastOptions = {}) => {
    const id = ++seq.current
    const tone = opts.tone ?? 'info'
    setItems(list => [...list.slice(-4), { id, message, ...opts, tone }])
    const ms = opts.durationMs ?? (tone === 'err' ? 0 : DEFAULT_MS)
    if (ms > 0) timers.current.set(id, setTimeout(() => dismiss(id), ms))
    return id
  }, [dismiss])

  const value = useMemo(() => ({ notify, dismiss }), [notify, dismiss])
  return (
    <ToastContext.Provider value={value}>
      {children}
      {createPortal(
        <div className="stac-toasts" aria-live="polite">
          {items.map(item => (
            <div key={item.id} role={item.tone === 'err' ? 'alert' : 'status'} className={`stac-toast stac-toast--${item.tone}`}>
              <span className="stac-toast__icon">{ICONS[item.tone ?? 'info']}</span>
              <div className="stac-toast__body">
                {item.title && <div className="stac-toast__title">{item.title}</div>}
                <div className="stac-toast__message">{item.message}</div>
              </div>
              {item.action && (
                <Button size="sm" variant="ghost" onClick={() => { item.action?.onClick(); dismiss(item.id) }}>{item.action.label}</Button>
              )}
              <button type="button" className="stac-toast__close" aria-label={t('common.dismiss')} onClick={() => dismiss(item.id)}><X aria-hidden /></button>
            </div>
          ))}
        </div>,
        document.body,
      )}
    </ToastContext.Provider>
  )
}

export function useToast(): ToastContextValue {
  return useContext(ToastContext)
}
