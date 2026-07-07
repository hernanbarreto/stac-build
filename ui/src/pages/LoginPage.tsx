// STAC-BUILD: Login Page
// Premium dark login screen

import { useState, useEffect, FormEvent } from 'react'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
    const { login } = useAuth()
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
        if (!result.ok) {
            setError(result.error || 'Login failed')
        }
    }

    return (
        <div className="login-page">
            <div className="login-card">
                <img src="/logo.png" alt="STAC Build" className="login-logo-img" />
                <p className="login-subtitle">Ingerop IN3 — Spatio-Temporal Awareness Core</p>

                {/* Server status indicator */}
                <div style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    gap: 6, margin: '8px 0 12px', fontSize: 12, opacity: 0.85
                }}>
                    <span style={{
                        width: 8, height: 8, borderRadius: '50%',
                        background: serverUp === null ? 'var(--text-muted)' : serverUp ? 'var(--success)' : 'var(--error)',
                        boxShadow: serverUp ? '0 0 6px rgba(63, 185, 80, 0.6)' : serverUp === false ? '0 0 6px rgba(248, 81, 73, 0.6)' : 'none',
                        display: 'inline-block',
                    }} />
                    <span style={{ color: serverUp === null ? 'var(--text-muted)' : serverUp ? 'var(--success)' : 'var(--error)' }}>
                        {serverUp === null ? 'Checking server…' : serverUp ? 'Server online' : 'Server offline'}
                    </span>
                </div>

                <form className="login-form" onSubmit={handleSubmit}>
                    <div className="login-field">
                        <label htmlFor="username">Username</label>
                        <input
                            id="username"
                            type="text"
                            value={username}
                            onChange={e => setUsername(e.target.value)}
                            placeholder="Enter username"
                            autoFocus
                            autoComplete="username"
                            required
                        />
                    </div>

                    <div className="login-field">
                        <label htmlFor="password">Password</label>
                        <input
                            id="password"
                            type="password"
                            value={password}
                            onChange={e => setPassword(e.target.value)}
                            placeholder="Enter password"
                            autoComplete="current-password"
                            required
                        />
                    </div>

                    {error && <div className="login-error">{error}</div>}

                    <button
                        type="submit"
                        className="login-btn"
                        disabled={loading || !username || !password || serverUp === false}
                    >
                        {loading ? 'Signing in…' : serverUp === false ? 'Server offline' : 'Sign In'}
                    </button>
                </form>

                <div className="login-footer">
                    Hernán Barreto — Ingerop IN3
                </div>
            </div>
        </div>
    )
}
