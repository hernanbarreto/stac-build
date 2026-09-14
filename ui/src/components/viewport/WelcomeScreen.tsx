/**
 * WelcomeScreen — shown in the viewport when no session is loaded: the logo,
 * one line of context and the first step (connect / pick a session).
 */
import { Plug } from 'lucide-react'
import { Button } from '../ui/Button'
import { useT } from '../../i18n'

export function WelcomeScreen({ connected, sessionCount, onConnect }: { connected: boolean; sessionCount: number; onConnect: () => void }) {
  const t = useT()
  return (
    <div className="stac-welcome">
      <img src="/logo.png" alt={t('app.name')} className="stac-welcome__logo" />
      <p className="stac-welcome__text">{connected ? t('welcome.pickSession') : t('welcome.connectHint')}</p>
      {!connected && <Button variant="primary" icon={<Plug aria-hidden />} onClick={onConnect}>{t('welcome.connect')}</Button>}
      <p className="stac-welcome__hint">{connected ? t.plural('welcome.sessionsAvailable', sessionCount) : t('welcome.defaultServer')}</p>
    </div>
  )
}
