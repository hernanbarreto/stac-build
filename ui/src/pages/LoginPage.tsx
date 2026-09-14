/**
 * LoginPage — the gate: logo on --surface-brand, server status, two fields,
 * one primary action. Same /api/auth/login flow through AuthContext.
 */
import { useState, useEffect, FormEvent } from 'react'
import { LogIn } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import { Button } from '../components/ui/Button'
import { Field, Input } from '../components/ui/Field'
import { Banner } from '../components/ui/Banner'
import { Stack } from '../components/ui/Panel'
import { useT } from '../i18n'

export default function LoginPage() {
  const { login } = useAuth()
  const t = useT()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [serverUp, setServerUp] = useState<boolean | null>(null) // null = checking

  // Poll server health
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null
    const check = async () => {
      try {
        const r = await fetch('/health', { signal: AbortSignal.timeout(15000) })
        setServerUp(r.ok)
      } catch {
        setServerUp(false)
      }
      timer = setTimeout(check, serverUp === false ? 8000 : 5000)
    }
    check()
    return () => { if (timer) clearTimeout(timer) }
  }, [])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    const result = await login(username, password)
    setLoading(false)
    if (!result.ok) setError(result.error || t('login.failed'))
  }

  return (
    <div className="stac-login">
      <div className="stac-login__card">
        <img src="/logo.png" alt={t('app.name')} className="stac-login__logo" />
        <p className="stac-login__subtitle">{t('login.subtitle')}</p>
        <div className="stac-login__server" role="status">
          <span className={`stac-live-dot ${serverUp === null ? '' : serverUp ? 'stac-live-dot--ok stac-live-dot--pulse' : 'stac-live-dot--err'}`} aria-hidden />
          <span>{serverUp === null ? t('login.checkingServer') : serverUp ? t('status.serverUp') : t('status.serverDown')}</span>
        </div>
        <form className="stac-login__form" onSubmit={handleSubmit}>
          <Stack gap={3}>
            <Field label={t('login.username')} htmlFor="username">
              <Input id="username" type="text" value={username} onChange={e => setUsername(e.target.value)} placeholder={t('login.usernamePlaceholder')} autoFocus autoComplete="username" required />
            </Field>
            <Field label={t('login.password')} htmlFor="password">
              <Input id="password" type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder={t('login.passwordPlaceholder')} autoComplete="current-password" required />
            </Field>
            {error && <Banner tone="err" compact>{error}</Banner>}
            <Button type="submit" variant="primary" block icon={<LogIn aria-hidden />} loading={loading} disabled={!username || !password || serverUp === false}>
              {serverUp === false ? t('status.serverDown') : t('login.signIn')}
            </Button>
          </Stack>
        </form>
        <div className="stac-login__footer">{t('login.footer')}</div>
      </div>
    </div>
  )
}
