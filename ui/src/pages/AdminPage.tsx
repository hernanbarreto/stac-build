// STAC-BUILD: Admin Panel — User & Team Management
// Create, edit, and manage users and teams (admin only)

import { useState, useEffect, useCallback } from 'react'
import { useAuth, AuthUser } from '../context/AuthContext'
import { useConfirmDialog } from '../components/ConfirmDialog'

interface Props {
    onClose: () => void
}

interface TeamData {
    id: number
    name: string
    description: string | null
    manager_id: number
    manager_name: string
    is_active: boolean
    members: { user_id: number; username: string; full_name: string | null; role: string }[]
    sessions: { id: number; session_id: string }[]
}

export default function AdminPage({ onClose }: Props) {
    const { token, user: currentUser } = useAuth()
    const { confirmDanger, alert, dialogElement } = useConfirmDialog()
    const [activeTab, setActiveTab] = useState<'users' | 'teams'>('users')

    // ── Users state ───────────────────────────────────────────
    const [users, setUsers] = useState<AuthUser[]>([])
    const [loading, setLoading] = useState(true)
    const [showCreate, setShowCreate] = useState(false)
    const [newUsername, setNewUsername] = useState('')
    const [newPassword, setNewPassword] = useState('')
    const [newEmail, setNewEmail] = useState('')
    const [newFullName, setNewFullName] = useState('')
    const [newRole, setNewRole] = useState('viewer')
    const [editingUserId, setEditingUserId] = useState<number | null>(null)
    const [editFullName, setEditFullName] = useState('')
    const [editEmail, setEditEmail] = useState('')
    const [editPassword, setEditPassword] = useState('')

    // ── Teams state ───────────────────────────────────────────
    const [teams, setTeams] = useState<TeamData[]>([])
    const [teamsLoading, setTeamsLoading] = useState(true)
    const [showCreateTeam, setShowCreateTeam] = useState(false)
    const [teamName, setTeamName] = useState('')
    const [teamDesc, setTeamDesc] = useState('')
    const [teamManagerId, setTeamManagerId] = useState<number | ''>('')
    const [expandedTeam, setExpandedTeam] = useState<number | null>(null)
    const [addMemberUserId, setAddMemberUserId] = useState<number | ''>('')
    const [addSessionId, setAddSessionId] = useState('')
    const [newSessionName, setNewSessionName] = useState('')
    const [showCreateSession, setShowCreateSession] = useState(false)
    const [allSessions, setAllSessions] = useState<{ id: string; frame_count: number; has_cloud: boolean }[]>([])

    const headers: HeadersInit = {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
    }

    // ── Fetch users ───────────────────────────────────────────
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

    // ── Fetch teams ───────────────────────────────────────────
    const fetchTeams = useCallback(async () => {
        try {
            const res = await fetch('/api/teams', { headers })
            if (res.ok) {
                const data = await res.json()
                setTeams(data.teams || [])
            }
        } catch { /* ignore */ }
        setTeamsLoading(false)
    }, [token])

    useEffect(() => { fetchUsers(); fetchTeams(); fetchAllSessions() }, [fetchUsers, fetchTeams])

    // ── Fetch all sessions (admin sees all) ───────────────────
    const fetchAllSessions = useCallback(async () => {
        try {
            const res = await fetch('/sessions', { headers })
            if (res.ok) {
                const data = await res.json()
                setAllSessions(Array.isArray(data) ? data : [])
            }
        } catch { /* ignore */ }
    }, [token])

    // ── User CRUD ─────────────────────────────────────────────
    const handleCreate = async () => {
        const res = await fetch('/api/auth/users', {
            method: 'POST',
            headers,
            body: JSON.stringify({
                username: newUsername, password: newPassword,
                email: newEmail || null, full_name: newFullName || null, role: newRole,
            }),
        })
        if (res.ok) {
            setShowCreate(false)
            setNewUsername(''); setNewPassword(''); setNewEmail(''); setNewFullName(''); setNewRole('viewer')
            fetchUsers()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to create user', 'Error')
        }
    }

    const handleToggleActive = async (u: AuthUser) => {
        await fetch(`/api/auth/users/${u.id}`, {
            method: 'PUT', headers,
            body: JSON.stringify({ is_active: !u.is_active }),
        })
        fetchUsers()
    }

    const handleRoleChange = async (u: AuthUser, role: string) => {
        await fetch(`/api/auth/users/${u.id}`, {
            method: 'PUT', headers,
            body: JSON.stringify({ role }),
        })
        fetchUsers()
    }

    const startEdit = (u: AuthUser) => {
        setEditingUserId(editingUserId === u.id ? null : u.id)
        setEditFullName(u.full_name || '')
        setEditEmail(u.email || '')
        setEditPassword('')
    }

    const handleSaveEdit = async (u: AuthUser) => {
        const body: Record<string, string | null> = {}
        if (editFullName !== (u.full_name || '')) body.full_name = editFullName || null
        if (editEmail !== (u.email || '')) body.email = editEmail || null
        if (editPassword) body.password = editPassword
        if (Object.keys(body).length === 0) { setEditingUserId(null); return }
        const res = await fetch(`/api/auth/users/${u.id}`, {
            method: 'PUT', headers,
            body: JSON.stringify(body),
        })
        if (res.ok) {
            setEditingUserId(null)
            setEditPassword('')
            fetchUsers()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to update user', 'Error')
        }
    }

    const handleDelete = async (u: AuthUser) => {
        const ok = await confirmDanger(`Delete user "${u.username}"? This cannot be undone.`, 'Delete User')
        if (!ok) return
        await fetch(`/api/auth/users/${u.id}`, { method: 'DELETE', headers })
        fetchUsers()
    }

    // ── Team CRUD ─────────────────────────────────────────────
    const handleCreateTeam = async () => {
        if (!teamName || !teamManagerId) return
        const res = await fetch('/api/teams', {
            method: 'POST', headers,
            body: JSON.stringify({ name: teamName, description: teamDesc || null, manager_id: teamManagerId }),
        })
        if (res.ok) {
            setShowCreateTeam(false)
            setTeamName(''); setTeamDesc(''); setTeamManagerId('')
            fetchTeams()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to create team', 'Error')
        }
    }

    const handleDeleteTeam = async (teamId: number, name: string) => {
        const ok = await confirmDanger(`Delete team "${name}"? This removes all members and session assignments.`, 'Delete Team')
        if (!ok) return
        await fetch(`/api/teams/${teamId}`, { method: 'DELETE', headers })
        fetchTeams()
    }

    const handleAddMember = async (teamId: number) => {
        if (!addMemberUserId) return
        const res = await fetch(`/api/teams/${teamId}/members`, {
            method: 'POST', headers,
            body: JSON.stringify({ user_id: addMemberUserId }),
        })
        if (res.ok) {
            setAddMemberUserId('')
            fetchTeams()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to add member', 'Error')
        }
    }

    const handleRemoveMember = async (teamId: number, userId: number, username: string) => {
        const ok = await confirmDanger(`Remove ${username} from this team?`, 'Remove Member')
        if (!ok) return
        await fetch(`/api/teams/${teamId}/members/${userId}`, { method: 'DELETE', headers })
        fetchTeams()
    }

    const handleAssignSession = async (teamId: number) => {
        if (!addSessionId.trim()) return
        const res = await fetch(`/api/teams/${teamId}/sessions`, {
            method: 'POST', headers,
            body: JSON.stringify({ session_id: addSessionId.trim() }),
        })
        if (res.ok) {
            setAddSessionId('')
            fetchTeams()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to assign session', 'Error')
        }
    }

    const handleUnassignSession = async (teamId: number, sessionId: string) => {
        await fetch(`/api/teams/${teamId}/sessions/${sessionId}`, { method: 'DELETE', headers })
        fetchTeams()
    }

    const handleCreateSession = async (teamId: number) => {
        if (!newSessionName.trim()) return
        const res = await fetch('/sessions', {
            method: 'POST', headers,
            body: JSON.stringify({ name: newSessionName.trim() }),
        })
        if (res.ok) {
            const data = await res.json()
            // Auto-assign to this team
            await fetch(`/api/teams/${teamId}/sessions`, {
                method: 'POST', headers,
                body: JSON.stringify({ session_id: data.session_id }),
            })
            setNewSessionName('')
            setShowCreateSession(false)
            fetchTeams()
            fetchAllSessions()
        } else {
            const err = await res.json().catch(() => ({}))
            alert(err.detail || 'Failed to create session', 'Error')
        }
    }

    const handleChangeManager = async (teamId: number, managerId: number) => {
        await fetch(`/api/teams/${teamId}`, {
            method: 'PUT', headers,
            body: JSON.stringify({ manager_id: managerId }),
        })
        fetchTeams()
    }

    const ROLES = ['admin', 'manager', 'editor', 'viewer', 'recorder']
    const roleLabel: Record<string, string> = {
        admin: '🔑 Admin', manager: '📋 Manager', editor: '✏️ Editor',
        viewer: '👁️ Viewer', recorder: '🎥 Recorder',
    }

    const managers = users.filter(u => u.role === 'manager' || u.role === 'admin')

    return (
        <div className="admin-overlay">
            <div className="admin-panel" style={{ maxWidth: 680 }}>
                <div className="admin-header">
                    <h2>⚙️ Administration</h2>
                    <button className="admin-close" onClick={onClose}>✕</button>
                </div>

                {/* Tab bar */}
                <div className="admin-tabs">
                    <button className={activeTab === 'users' ? 'active' : ''}
                        onClick={() => setActiveTab('users')}>
                        👤 Users
                    </button>
                    <button className={activeTab === 'teams' ? 'active' : ''}
                        onClick={() => setActiveTab('teams')}>
                        👥 Teams
                    </button>
                </div>

                {/* ═══════════ USERS TAB ═══════════ */}
                {activeTab === 'users' && (
                    <>
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
                                        <button className="admin-edit-btn"
                                            onClick={() => startEdit(u)}
                                            disabled={u.id === currentUser?.id}
                                            title="Edit user">
                                            ✏️
                                        </button>
                                        <button className="admin-delete-btn"
                                            onClick={() => handleDelete(u)}
                                            disabled={u.id === currentUser?.id}
                                            title="Delete user">
                                            🗑️
                                        </button>
                                    </div>
                                    {editingUserId === u.id && (
                                        <div className="admin-edit-form">
                                            <div className="admin-form-row">
                                                <input placeholder="Full Name" value={editFullName}
                                                    onChange={e => setEditFullName(e.target.value)} />
                                                <input placeholder="Email" value={editEmail}
                                                    onChange={e => setEditEmail(e.target.value)} />
                                            </div>
                                            <div className="admin-form-row">
                                                <input placeholder="New Password (leave empty to keep)" type="password"
                                                    value={editPassword}
                                                    onChange={e => setEditPassword(e.target.value)} />
                                                <button className="admin-save-btn" onClick={() => handleSaveEdit(u)}>
                                                    💾 Save
                                                </button>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </>
                )}

                {/* ═══════════ TEAMS TAB ═══════════ */}
                {activeTab === 'teams' && (
                    <>
                        <div className="admin-toolbar">
                            <button className="admin-create-btn" onClick={() => setShowCreateTeam(!showCreateTeam)}>
                                {showCreateTeam ? 'Cancel' : '+ New Team'}
                            </button>
                            <span className="admin-count">{teams.length} teams</span>
                        </div>

                        {/* Create team form */}
                        {showCreateTeam && (
                            <div className="admin-create-form">
                                <div className="admin-form-row">
                                    <input placeholder="Team Name *" value={teamName}
                                        onChange={e => setTeamName(e.target.value)} />
                                    <input placeholder="Description" value={teamDesc}
                                        onChange={e => setTeamDesc(e.target.value)} />
                                </div>
                                <div className="admin-form-row">
                                    <select value={teamManagerId}
                                        onChange={e => setTeamManagerId(e.target.value ? Number(e.target.value) : '')}>
                                        <option value="">— Select Manager —</option>
                                        {managers.map(m => (
                                            <option key={m.id} value={m.id}>
                                                {m.full_name || m.username} ({m.role})
                                            </option>
                                        ))}
                                    </select>
                                    <button className="admin-save-btn" onClick={handleCreateTeam}
                                        disabled={!teamName || !teamManagerId}>
                                        Create Team
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Team list */}
                        <div className="admin-user-list">
                            {teamsLoading && <div className="admin-loading">Loading…</div>}
                            {teams.map(team => (
                                <div key={team.id} className="admin-team-card">
                                    {/* Team header row */}
                                    <div className="admin-team-row"
                                        onClick={() => setExpandedTeam(expandedTeam === team.id ? null : team.id)}>
                                        <div className="admin-team-info">
                                            <div className="admin-team-icon">🏗️</div>
                                            <div>
                                                <div className="admin-team-name">{team.name}</div>
                                                <div className="admin-user-meta">
                                                    📋 {team.manager_name} · {team.members.length} members · {team.sessions.length} sessions
                                                </div>
                                            </div>
                                        </div>
                                        <div className="admin-user-actions">
                                            <span className="admin-expand">{expandedTeam === team.id ? '▼' : '▶'}</span>
                                            <button className="admin-delete-btn"
                                                onClick={(e) => { e.stopPropagation(); handleDeleteTeam(team.id, team.name) }}
                                                title="Delete team">
                                                🗑️
                                            </button>
                                        </div>
                                    </div>

                                    {/* Expanded details */}
                                    {expandedTeam === team.id && (
                                        <div className="admin-team-details">
                                            {team.description && (
                                                <div className="admin-team-desc">{team.description}</div>
                                            )}

                                            {/* Manager */}
                                            <div className="admin-team-section">
                                                <div className="admin-team-section-title">📋 Project Manager</div>
                                                <div className="admin-team-section-row">
                                                    <select value={team.manager_id}
                                                        onChange={e => handleChangeManager(team.id, Number(e.target.value))}>
                                                        {managers.map(m => (
                                                            <option key={m.id} value={m.id}>
                                                                {m.full_name || m.username}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                            </div>

                                            {/* Members */}
                                            <div className="admin-team-section">
                                                <div className="admin-team-section-title">👥 Members</div>
                                                {team.members.map(m => (
                                                    <div key={m.user_id} className="admin-team-member">
                                                        <span>{m.full_name || m.username}</span>
                                                        <span className="admin-team-member-role">{roleLabel[m.role] || m.role}</span>
                                                        <button className="admin-team-remove-btn"
                                                            onClick={() => handleRemoveMember(team.id, m.user_id, m.username)}
                                                            title="Remove">✕</button>
                                                    </div>
                                                ))}
                                                <div className="admin-team-add-row">
                                                    <select value={addMemberUserId}
                                                        onChange={e => setAddMemberUserId(e.target.value ? Number(e.target.value) : '')}>
                                                        <option value="">— Add member —</option>
                                                        {users
                                                            .filter(u => !team.members.some(m => m.user_id === u.id))
                                                            .map(u => (
                                                                <option key={u.id} value={u.id}>
                                                                    {u.full_name || u.username} ({u.role})
                                                                </option>
                                                            ))}
                                                    </select>
                                                    <button className="admin-team-add-btn"
                                                        onClick={() => handleAddMember(team.id)}
                                                        disabled={!addMemberUserId}>➕</button>
                                                </div>
                                            </div>

                                            {/* Sessions */}
                                            <div className="admin-team-section">
                                                <div className="admin-team-section-title">📂 Assigned Sessions</div>
                                                {team.sessions.map(s => (
                                                    <div key={s.session_id} className="admin-team-session">
                                                        <span className="admin-team-session-id">📁 {s.session_id}</span>
                                                        <button className="admin-team-remove-btn"
                                                            onClick={() => handleUnassignSession(team.id, s.session_id)}
                                                            title="Unassign">✕</button>
                                                    </div>
                                                ))}
                                                <div className="admin-team-add-row">
                                                    {!showCreateSession ? (
                                                        <>
                                                            <select value={addSessionId}
                                                                onChange={e => setAddSessionId(e.target.value)}>
                                                                <option value="">— Select session to assign —</option>
                                                                {allSessions
                                                                    .filter(s => !team.sessions.some(ts => ts.session_id === s.id))
                                                                    .map(s => (
                                                                        <option key={s.id} value={s.id}>
                                                                            📁 {s.id} ({s.frame_count} frames{s.has_cloud ? ', ☁️' : ''})
                                                                        </option>
                                                                    ))}
                                                            </select>
                                                            <button className="admin-team-add-btn"
                                                                onClick={() => handleAssignSession(team.id)}
                                                                disabled={!addSessionId}>➕</button>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <input placeholder="New session name (e.g. site-north-2026)"
                                                                value={newSessionName}
                                                                onChange={e => setNewSessionName(e.target.value)}
                                                                onKeyDown={e => e.key === 'Enter' && handleCreateSession(team.id)} />
                                                            <button className="admin-team-add-btn"
                                                                onClick={() => handleCreateSession(team.id)}
                                                                disabled={!newSessionName.trim()}>✅</button>
                                                        </>
                                                    )}
                                                </div>
                                                <div className="admin-team-add-row" style={{ justifyContent: 'flex-end', marginTop: 4 }}>
                                                    <button className="admin-team-toggle-create"
                                                        onClick={() => { setShowCreateSession(!showCreateSession); setNewSessionName('') }}>
                                                        {showCreateSession ? '← Back to list' : '+ Create New Session'}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            ))}
                            {!teamsLoading && teams.length === 0 && (
                                <div className="admin-loading" style={{ opacity: 0.5 }}>
                                    No teams yet. Create one above.
                                </div>
                            )}
                        </div>
                    </>
                )}
            </div>
            {dialogElement}
        </div>
    )
}
