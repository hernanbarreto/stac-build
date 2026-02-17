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
