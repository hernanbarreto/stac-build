/**
 * SplashOverlay — session / pipeline loading screen redesigned on
 * --surface-brand with the logo and a progress bar by stages (§7).
 * No particles, no rings, no decorative animation: the only motion is the
 * progress indicator.
 */
import { Loader2 } from 'lucide-react'
import { Progress, type ProgressStage } from '../ui/Progress'
import { useT } from '../../i18n'

interface SplashOverlayProps {
  title: string
  status?: string
  stages?: ProgressStage[]
  value?: number
}

export function SplashOverlay({ title, status, stages, value }: SplashOverlayProps) {
  const t = useT()
  return (
    <div className="stac-splash" role="status" aria-live="polite" aria-label={title}>
      <div className="stac-splash__card">
        <img src="/logo.png" alt={t('app.name')} className="stac-splash__logo" />
        <div className="stac-splash__title">{title}</div>
        {status && (
          <div className="stac-splash__status">
            <Loader2 className="stac-spin" aria-hidden />
            <span>{status}</span>
          </div>
        )}
        <div className="stac-splash__progress">
          <Progress value={value} stages={stages} showValue={value != null || !!stages?.length} tone="brand" size="md" />
        </div>
      </div>
    </div>
  )
}
