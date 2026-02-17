// STAC-BUILD: Login Page
// Premium dark login screen

import { useState, FormEvent } from 'react'
import { useAuth } from '../context/AuthContext'

export default function LoginPage() {
    const { login } = useAuth()
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [error, setError] = useState('')
    const [loading, setLoading] = useState(false)

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
                <div className="login-logo">S</div>
                <h1 className="login-title">STAC Build</h1>
                <p className="login-subtitle">Spatio-Temporal Awareness Core</p>

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
                        disabled={loading || !username || !password}
                    >
                        {loading ? 'Signing in…' : 'Sign In'}
                    </button>
                </form>

                <div className="login-footer">
                    Hernán Barreto — Ingerop IN3
                </div>
            </div>
        </div>
    )
}
