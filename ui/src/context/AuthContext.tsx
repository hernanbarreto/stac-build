// STAC-BUILD: Auth Context
// Manages JWT token, user state, login/logout

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react'

export interface AuthUser {
    id: number
    username: string
    email: string | null
    role: string
    full_name: string | null
    avatar_url: string | null
    is_active: boolean
    created_at: string | null
    last_login: string | null
}

interface AuthContextType {
    user: AuthUser | null
    token: string | null
    loading: boolean
    login: (username: string, password: string) => Promise<{ ok: boolean; error?: string }>
    logout: () => void
}

const AuthContext = createContext<AuthContextType>({
    user: null,
    token: null,
    loading: true,
    login: async () => ({ ok: false }),
    logout: () => { },
})

export const useAuth = () => useContext(AuthContext)

/** The login is renewed with less than this left of its 24 hours — looked at every CHECK_EVERY_MS and
 *  whenever the page comes back to the front. */
const RENEW_WHEN_LEFT_S = 12 * 3600
const CHECK_EVERY_MS = 10 * 60 * 1000

/** A token's expiry in seconds since the epoch — null when it cannot be read. */
function expiryOf(token: string): number | null {
    try {
        const exp = JSON.parse(atob(token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/'))).exp
        return typeof exp === 'number' ? exp : null
    } catch { return null }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
    const [user, setUser] = useState<AuthUser | null>(null)
    const [token, setToken] = useState<string | null>(null)
    const [loading, setLoading] = useState(true)

    // On mount: restore token from localStorage and validate
    useEffect(() => {
        const saved = localStorage.getItem('stac_token')
        if (!saved) {
            setLoading(false)
            return
        }

        fetch('/api/auth/me', {
            headers: { Authorization: `Bearer ${saved}` },
        })
            .then(async (res) => {
                if (res.ok) {
                    const data = await res.json()
                    setUser(data.user)
                    setToken(saved)
                } else {
                    localStorage.removeItem('stac_token')
                }
            })
            .catch(() => {
                localStorage.removeItem('stac_token')
            })
            .finally(() => setLoading(false))
    }, [])

    // THE LOGIN DOES NOT EXPIRE UNDER A USER WHO IS WORKING, AND AN EXPIRED ONE IS SAID (USER 2026-11-19: a page
    // left open over a day kept LOOKING signed in while the server refused it request by request — the viewer
    // preferences came back empty, "me deja los puntos el 5", and the session download answered "Invalid or
    // expired token"). With under half of its 24 hours left the token is renewed; one that is no longer
    // valid sends the user back to the sign-in screen.
    useEffect(() => {
        if (!token) return
        let live = true
        const check = async () => {
            const exp = expiryOf(token)
            if (exp === null || exp - Date.now() / 1000 > RENEW_WHEN_LEFT_S) return
            try {
                const res = await fetch('/api/auth/refresh', { method: 'POST', headers: { Authorization: `Bearer ${token}` } })
                if (!live) return
                if (res.ok) {
                    const data = await res.json()
                    if (typeof data.token === 'string') {
                        localStorage.setItem('stac_token', data.token)
                        setToken(data.token)
                    }
                } else if (res.status === 401 || res.status === 403) {
                    localStorage.removeItem('stac_token')
                    setToken(null)
                    setUser(null)
                }
            } catch { /* the server cannot be reached: nothing is concluded about the login */ }
        }
        void check()
        const timer = window.setInterval(() => { void check() }, CHECK_EVERY_MS)
        const onFront = () => { if (document.visibilityState === 'visible') void check() }
        document.addEventListener('visibilitychange', onFront)
        window.addEventListener('focus', onFront)
        return () => {
            live = false
            window.clearInterval(timer)
            document.removeEventListener('visibilitychange', onFront)
            window.removeEventListener('focus', onFront)
        }
    }, [token])

    const login = useCallback(async (username: string, password: string) => {
        try {
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            })

            if (!res.ok) {
                const err = await res.json().catch(() => ({}))
                return { ok: false, error: err.detail || 'Login failed' }
            }

            const data = await res.json()
            localStorage.setItem('stac_token', data.token)
            setToken(data.token)
            setUser(data.user)
            return { ok: true }
        } catch (e) {
            return { ok: false, error: 'Connection failed' }
        }
    }, [])

    const logout = useCallback(() => {
        localStorage.removeItem('stac_token')
        setToken(null)
        setUser(null)
    }, [])

    return (
        <AuthContext.Provider value={{ user, token, loading, login, logout }}>
            {children}
        </AuthContext.Provider>
    )
}
