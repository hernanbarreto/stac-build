/**
 * Banner — inline notice with icon, text and an optional action; never text
 * alone (§6). Tones map to the state colours; `glass` variant floats over
 * the viewport (pipeline running, correction pending).
 */
import { AlertTriangle, CheckCircle2, Info, X, XCircle } from 'lucide-react'
import type { ReactNode } from 'react'
import { useT } from '../../i18n'

export type BannerTone = 'info' | 'ok' | 'warn' | 'err' | 'brand' | 'neutral'

interface BannerProps {
  tone?: BannerTone
  icon?: ReactNode
  title?: ReactNode
  children?: ReactNode
  action?: ReactNode
  onClose?: () => void
  glass?: boolean
  compact?: boolean
  className?: string
}

const ICONS: Record<BannerTone, ReactNode> = {
  info: <Info aria-hidden />,
  ok: <CheckCircle2 aria-hidden />,
  warn: <AlertTriangle aria-hidden />,
  err: <XCircle aria-hidden />,
  brand: <Info aria-hidden />,
  neutral: <Info aria-hidden />,
}

export function Banner({ tone = 'info', icon, title, children, action, onClose, glass = false, compact = false, className = '' }: BannerProps) {
  const t = useT()
  return (
    <div role={tone === 'err' ? 'alert' : 'status'} className={`stac-banner stac-banner--${tone} ${glass ? 'stac-banner--glass' : ''} ${compact ? 'stac-banner--compact' : ''} ${className}`.trim()}>
      <span className="stac-banner__icon">{icon ?? ICONS[tone]}</span>
      <div className="stac-banner__body">
        {title && <div className="stac-banner__title">{title}</div>}
        {children && <div className="stac-banner__text">{children}</div>}
      </div>
      {action && <div className="stac-banner__action">{action}</div>}
      {onClose && (
        <button type="button" className="stac-banner__close" aria-label={t('common.dismiss')} onClick={onClose}>
          <X aria-hidden />
        </button>
      )}
    </div>
  )
}
