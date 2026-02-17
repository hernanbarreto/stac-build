// STAC-BUILD: Admin Panel — User Management
// Create, edit, and manage users (admin only)

import { useState, useEffect, useCallback } from 'react'
import { useAuth, AuthUser } from '../context/AuthContext'

interface Props {
    onClose: () => void
}

export default function AdminPage({ onClose }: Props) {
    const { token, user: currentUser } = useAuth()
    const [users, setUsers] = useState<AuthUser[]>([])
    const [loading, setLoading] = useState(true)
    const [showCreate, setShowCreate] = useState(false)


    // Create form
    const [newUsername, setNewUsername] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [newEmail, setNewEmail] = useState('')
    const [newFullName, setNewFullName] = useState('')
    const [newRole, setNewRole] = useState('viewer')

    const headers = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
    }

    const fetchUsers = useCallback(async () => {
        try {
            const res = await fetch('/api/auth/users', { headers })
            if (res.ok) {
                const data = await res.json()
                setUsers(data.users)
            }
        } catch { /* ignore */ }
        setLoading(false)
    }, [token])

    useEffect(() => { fetchUsers() }, [fetchUsers])

    const handleCreate = async () => {
        const res = await fetch('/api/auth/users', {
            method: 'POST',
            headers,
            body: JSON.stringify({
                username: newUsername,
                password: newPassword,
                email: newEmail || null,
                full_name: newFullName || null,
                role: newRole,
            }),
        })
        if (res.ok) {
            setShowCreate(false)
            setNewUsername(''); setNewPassword(''); setNewEmail(''); setNewFullName(''); setNewRole('viewer')
            fetchUsers()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to create user')
        }
    }

    const handleToggleActive = async (u: AuthUser) => {
        await fetch(`/api/auth/users/${u.id}`, {
            method: 'PUT',
            headers,
            body: JSON.stringify({ is_active: !u.is_active }),
        })
        fetchUsers()
    }

    const handleRoleChange = async (u: AuthUser, role: string) => {
        await fetch(`/api/auth/users/${u.id}`, {
            method: 'PUT',
            headers,
            body: JSON.stringify({ role }),
        })
        fetchUsers()
    }

    const handleDelete = async (u: AuthUser) => {
        if (!confirm(`Delete user "${u.username}"? This cannot be undone.`)) return
        await fetch(`/api/auth/users/${u.id}`, { method: 'DELETE', headers })
        fetchUsers()
    }

    const ROLES = ['admin', 'manager', 'editor', 'viewer', 'recorder']
    const roleLabel: Record<string, string> = {
        admin: '🔑 Admin',
        manager: '📋 Manager',
        editor: '✏️ Editor',
        viewer: '👁️ Viewer',
        recorder: '🎥 Recorder',
    }

    return (
        <div className="admin-overlay">
            <div className="admin-panel">
                <div className="admin-header">
                    <h2>👥 User Management</h2>
                    <button className="admin-close" onClick={onClose}>✕</button>
                </div>

                <div className="admin-toolbar">
                    <button className="admin-create-btn" onClick={() => setShowCreate(!showCreate)}>
                        {showCreate ? 'Cancel' : '+ New User'}
                    </button>
                    <span className="admin-count">{users.length} users</span>
                </div>

                {showCreate && (
                    <div className="admin-create-form">
                        <div className="admin-form-row">
                            <input placeholder="Username *" value={newUsername}
                                onChange={e => setNewUsername(e.target.value)} />
                            <input placeholder="Password *" type="password" value={newPassword}
                                onChange={e => setNewPassword(e.target.value)} />
                        </div>
                        <div className="admin-form-row">
                            <input placeholder="Full Name" value={newFullName}
                                onChange={e => setNewFullName(e.target.value)} />
                            <input placeholder="Email" value={newEmail}
                                onChange={e => setNewEmail(e.target.value)} />
                        </div>
                        <div className="admin-form-row">
                            <select value={newRole} onChange={e => setNewRole(e.target.value)}>
                                {ROLES.map(r => <option key={r} value={r}>{roleLabel[r]}</option>)}
                            </select>
                            <button className="admin-save-btn" onClick={handleCreate}
                                disabled={!newUsername || !newPassword}>
                                Create User
                            </button>
                        </div>
                    </div>
                )}

                <div className="admin-user-list">
                    {loading && <div className="admin-loading">Loading…</div>}
                    {users.map(u => (
                        <div key={u.id} className={`admin-user-row ${!u.is_active ? 'disabled' : ''}`}>
                            <div className="admin-user-info">
                                <div className="admin-user-avatar">
                                    {u.username[0].toUpperCase()}
                                </div>
                                <div className="admin-user-details">
                                    <div className="admin-user-name">
                                        {u.full_name || u.username}
                                        {u.id === currentUser?.id && <span className="admin-you-badge">you</span>}
                                    </div>
                                    <div className="admin-user-meta">
                                        @{u.username} · {u.email || 'no email'}
                                        {u.last_login && ` · Last login: ${new Date(u.last_login).toLocaleDateString()}`}
                                    </div>
                                </div>
                            </div>
                            <div className="admin-user-actions">
                                <select value={u.role}
                                    onChange={e => handleRoleChange(u, e.target.value)}
                                    disabled={u.id === currentUser?.id}>
                                    {ROLES.map(r => <option key={r} value={r}>{roleLabel[r]}</option>)}
                                </select>
                                <button
                                    className={`admin-toggle-btn ${u.is_active ? 'active' : 'inactive'}`}
                                    onClick={() => handleToggleActive(u)}
                                    disabled={u.id === currentUser?.id}
                                    title={u.is_active ? 'Disable' : 'Enable'}>
                                    {u.is_active ? '●' : '○'}
                                </button>
                                <button className="admin-delete-btn"
                                    onClick={() => handleDelete(u)}
                                    disabled={u.id === currentUser?.id}
                                    title="Delete user">
                                    🗑️
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}
