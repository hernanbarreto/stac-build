/**
 * STAC Build — Team Panel Component
 * Real-time team presence, chat, activity feed, member management
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { useAuth } from '../context/AuthContext'

// ─── Types ────────────────────────────────────────────────────

interface TeamData {
    id: number
    name: string
    description: string | null
    manager_id: number
    manager_name: string
    is_active: boolean
    members: TeamMemberData[]
    sessions: { session_id: string }[]
}

interface TeamMemberData {
    user_id: number
    username: string
    full_name: string | null
    role: string
    avatar_url: string | null
}

interface OnlineUser {
    user_id: number
    username: string
    task: string
}

interface ChatMessage {
    id: number
    team_id: number
    user_id: number
    username: string
    content: string
    timestamp: string
}

interface ActivityEntry {
    id: number
    user_id: number
    username: string
    action: string
    detail: string | null
    timestamp: string
}

interface AvailableUser {
    id: number
    username: string
    full_name: string | null
    role: string
}

interface TeamPanelProps {
    onCallUser?: (userId: number, username: string) => void
}

// ─── Component ────────────────────────────────────────────────

export default function TeamPanel({ onCallUser }: TeamPanelProps) {
    const { user, token } = useAuth()
    const [teams, setTeams] = useState<TeamData[]>([])
    const [selectedTeam, setSelectedTeam] = useState<number | null>(null)
    const [onlineUsers, setOnlineUsers] = useState<OnlineUser[]>([])
    const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
    const [activity, setActivity] = useState<ActivityEntry[]>([])
    const [chatInput, setChatInput] = useState('')
    const [activeTab, setActiveTab] = useState<'members' | 'chat' | 'activity'>('members')
    const [addMemberOpen, setAddMemberOpen] = useState(false)
    const [availableUsers, setAvailableUsers] = useState<AvailableUser[]>([])
    const wsRef = useRef<WebSocket | null>(null)
    const chatEndRef = useRef<HTMLDivElement>(null)

    const isManager = useCallback((teamId: number) => {
        const team = teams.find(t => t.id === teamId)
        return team && user && (team.manager_id === user.id || user.role === 'admin')
    }, [teams, user])

    const authHeaders = useCallback(() => ({
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
    }), [token])

    // ── Fetch teams ─────────────────────────────────────────────
    const fetchTeams = useCallback(async () => {
        if (!token) return
        try {
            const res = await fetch('/api/teams', { headers: authHeaders() })
            const data = await res.json()
            setTeams(data.teams || [])
            // Auto-select first team if none selected
            if (!selectedTeam && data.teams?.length > 0) {
                setSelectedTeam(data.teams[0].id)
            }
        } catch (e) {
            console.error('[TeamPanel] Failed to fetch teams:', e)
        }
    }, [token, authHeaders, selectedTeam])

    useEffect(() => { fetchTeams() }, [fetchTeams])

    // ── Fetch messages when team changes ────────────────────────
    useEffect(() => {
        if (!selectedTeam || !token) return
        fetch(`/api/teams/${selectedTeam}/messages?limit=100`, { headers: authHeaders() })
            .then(r => r.json())
            .then(data => setChatMessages(data.messages || []))
            .catch(console.error)
    }, [selectedTeam, token, authHeaders])

    // ── Fetch activity when tab changes ─────────────────────────
    useEffect(() => {
        if (activeTab !== 'activity' || !selectedTeam || !token) return
        fetch(`/api/teams/${selectedTeam}/activity?limit=50`, { headers: authHeaders() })
            .then(r => r.json())
            .then(data => setActivity(data.activity || []))
            .catch(console.error)
    }, [activeTab, selectedTeam, token, authHeaders])

    // ── WebSocket for presence + realtime chat ──────────────────
    useEffect(() => {
        if (!token) return

        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
        const ws = new WebSocket(`${proto}//${location.host}/ws/team`)
        wsRef.current = ws

        ws.onopen = () => {
            ws.send(JSON.stringify({ type: 'team_auth', token }))
        }

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data)
                if (msg.type === 'presence_update') {
                    setOnlineUsers(msg.online || [])
                } else if (msg.type === 'team_message' && msg.message) {
                    setChatMessages(prev => [...prev, msg.message])
                }
            } catch { /* ignore */ }
        }

        ws.onclose = () => { wsRef.current = null }

        return () => { ws.close() }
    }, [token])

    // Auto-scroll chat
    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [chatMessages])

    // ── Send chat message ───────────────────────────────────────
    const sendMessage = useCallback(() => {
        if (!chatInput.trim() || !selectedTeam || !wsRef.current) return
        wsRef.current.send(JSON.stringify({
            type: 'team_message',
            team_id: selectedTeam,
            content: chatInput.trim(),
        }))
        setChatInput('')
    }, [chatInput, selectedTeam])

    // ── Add member ──────────────────────────────────────────────
    const openAddMember = useCallback(async () => {
        try {
            const res = await fetch('/api/teams/available-users', { headers: authHeaders() })
            const data = await res.json()
            setAvailableUsers(data.users || [])
            setAddMemberOpen(true)
        } catch (e) {
            console.error('[TeamPanel] Failed to fetch users:', e)
        }
    }, [authHeaders])

    const addMember = useCallback(async (userId: number) => {
        if (!selectedTeam) return
        try {
            await fetch(`/api/teams/${selectedTeam}/members`, {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ user_id: userId }),
            })
            setAddMemberOpen(false)
            fetchTeams()
        } catch (e) {
            console.error('[TeamPanel] Failed to add member:', e)
        }
    }, [selectedTeam, authHeaders, fetchTeams])

    const removeMember = useCallback(async (userId: number) => {
        if (!selectedTeam) return
        if (!confirm('Remove this member from the team?')) return
        try {
            await fetch(`/api/teams/${selectedTeam}/members/${userId}`, {
                method: 'DELETE',
                headers: authHeaders(),
            })
            fetchTeams()
        } catch (e) {
            console.error('[TeamPanel] Failed to remove member:', e)
        }
    }, [selectedTeam, authHeaders, fetchTeams])

    // ── Helpers ─────────────────────────────────────────────────
    const isOnline = (userId: number) => onlineUsers.some(u => u.user_id === userId)
    const getUserTask = (userId: number) => onlineUsers.find(u => u.user_id === userId)?.task || ''
    const team = teams.find(t => t.id === selectedTeam)

    const roleBadge = (role: string) => {
        const map: Record<string, string> = { admin: '👑', manager: '📋', editor: '✏️', viewer: '👁️', recorder: '📹' }
        return map[role] || '👤'
    }

    const actionIcon = (action: string) => {
        const map: Record<string, string> = {
            login: '🔑', logout: '🚪', session_loaded: '📂', pipeline_started: '▶️',
            pipeline_completed: '✅', segment_edited: '🏷️', message_sent: '💬',
            team_joined: '➕', team_left: '➖', member_added: '👤', member_removed: '❌',
            session_assigned: '📌', team_created: '🏗️',
        }
        return map[action] || '📝'
    }

    if (!user) return null

    return (
        <div className="team-panel">
            {/* Team selector */}
            {teams.length > 1 && (
                <div className="team-selector">
                    <select
                        value={selectedTeam || ''}
                        onChange={e => setSelectedTeam(Number(e.target.value))}
                    >
                        {teams.map(t => (
                            <option key={t.id} value={t.id}>{t.name}</option>
                        ))}
                    </select>
                </div>
            )}

            {/* Team header */}
            {team && (
                <div className="team-header">
                    <div className="team-name">{team.name}</div>
                    <div className="team-meta">
                        {team.members.length} members · {team.sessions.length} sessions
                    </div>
                </div>
            )}

            {/* Tab bar */}
            <div className="team-tabs">
                <button className={activeTab === 'members' ? 'active' : ''} onClick={() => setActiveTab('members')}>
                    👥 Members
                </button>
                <button className={activeTab === 'chat' ? 'active' : ''} onClick={() => setActiveTab('chat')}>
                    💬 Chat
                </button>
                <button className={activeTab === 'activity' ? 'active' : ''} onClick={() => setActiveTab('activity')}>
                    📋 Activity
                </button>
            </div>

            {/* ── Members Tab ──────────────────────────────────── */}
            {activeTab === 'members' && team && (
                <div className="team-members">
                    {team.members
                        .filter(member => member.user_id !== user.id)
                        .map(member => (
                            <div key={member.user_id} className={`team-member-card ${isOnline(member.user_id) ? 'online' : 'offline'}`}>
                                <div className="member-avatar">
                                    {member.avatar_url
                                        ? <img src={member.avatar_url} alt="" />
                                        : <span>{(member.full_name || member.username)[0].toUpperCase()}</span>
                                    }
                                    <div className={`member-status-dot ${isOnline(member.user_id) ? 'online' : 'offline'}`} />
                                </div>
                                <div className="member-info">
                                    <div className="member-name">
                                        {member.full_name || member.username}
                                        <span className="member-role-badge">{roleBadge(member.role)}</span>
                                    </div>
                                    <div className="member-task">
                                        {isOnline(member.user_id)
                                            ? getUserTask(member.user_id) || 'Online'
                                            : 'Offline'
                                        }
                                    </div>
                                </div>
                                <div className="member-actions">
                                    <button
                                        className="member-action-btn"
                                        title="Video Call"
                                        onClick={() => onCallUser?.(member.user_id, member.username)}
                                    >
                                        📹
                                    </button>
                                    {isManager(team.id) && member.user_id !== team.manager_id && (
                                        <button
                                            className="member-action-btn danger"
                                            title="Remove from team"
                                            onClick={() => removeMember(member.user_id)}
                                        >
                                            ✕
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))}
                    {team.members.filter(m => m.user_id !== user.id).length === 0 && (
                        <div className="team-empty">No other members in this team yet</div>
                    )}
                    {/* Add member button — manager only */}
                    {selectedTeam && isManager(selectedTeam) && (
                        <button className="team-add-member-btn" onClick={openAddMember}>
                            ➕ Add Member
                        </button>
                    )}
                </div>
            )}

            {/* ── Chat Tab ─────────────────────────────────────── */}
            {activeTab === 'chat' && (
                <div className="team-chat">
                    <div className="team-chat-messages">
                        {chatMessages.length === 0 && (
                            <div className="team-empty" style={{ padding: '40px 16px', textAlign: 'center' }}>
                                <div style={{ fontSize: 32, marginBottom: 8 }}>💬</div>
                                <div>Team Chat — {team?.name}</div>
                                <div className="team-empty-hint">
                                    Messages here are visible to all {team?.members.length} team members.
                                    Start a conversation!
                                </div>
                            </div>
                        )}
                        {chatMessages.map(msg => (
                            <div
                                key={msg.id}
                                className={`chat-bubble ${msg.user_id === user.id ? 'mine' : 'theirs'}`}
                            >
                                {msg.user_id !== user.id && (
                                    <span className="chat-author">{msg.username}</span>
                                )}
                                <span className="chat-text">{msg.content}</span>
                                <span className="chat-time">
                                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                        ))}
                        <div ref={chatEndRef} />
                    </div>
                    <div className="team-chat-input">
                        <input
                            type="text"
                            placeholder={`Message ${team?.name || 'team'}...`}
                            value={chatInput}
                            onChange={e => setChatInput(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && sendMessage()}
                        />
                        <button onClick={sendMessage} disabled={!chatInput.trim()}>➤</button>
                    </div>
                </div>
            )}

            {/* ── Activity Tab ─────────────────────────────────── */}
            {activeTab === 'activity' && (
                <div className="team-activity">
                    {activity.map(entry => (
                        <div key={entry.id} className="activity-entry">
                            <span className="activity-icon">{actionIcon(entry.action)}</span>
                            <div className="activity-body">
                                <span className="activity-user">{entry.username}</span>
                                <span className="activity-action">{entry.action.replace(/_/g, ' ')}</span>
                                {entry.detail && <span className="activity-detail">{entry.detail}</span>}
                            </div>
                            <span className="activity-time">
                                {new Date(entry.timestamp).toLocaleString([], {
                                    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
                                })}
                            </span>
                        </div>
                    ))}
                    {activity.length === 0 && (
                        <div className="team-empty" style={{ padding: '40px 16px', textAlign: 'center' }}>
                            <div style={{ fontSize: 32, marginBottom: 8 }}>📋</div>
                            <div>No team activity yet</div>
                            <div className="team-empty-hint">
                                Actions like adding members, assigning sessions,
                                and sending messages will appear here.
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ── Add Member Modal ──────────────────────────────── */}
            {addMemberOpen && (
                <div className="team-modal-backdrop" onClick={() => setAddMemberOpen(false)}>
                    <div className="team-modal" onClick={e => e.stopPropagation()}>
                        <h4>Add Member</h4>
                        <div className="team-modal-list">
                            {availableUsers
                                .filter(u => !team?.members.some(m => m.user_id === u.id))
                                .map(u => (
                                    <div key={u.id} className="team-modal-user" onClick={() => addMember(u.id)}>
                                        <span>{u.full_name || u.username}</span>
                                        <span className="team-modal-role">{u.role}</span>
                                    </div>
                                ))
                            }
                        </div>
                        <button className="team-modal-close" onClick={() => setAddMemberOpen(false)}>Close</button>
                    </div>
                </div>
            )}

            {/* Empty state */}
            {teams.length === 0 && (
                <div className="team-empty">
                    <div>👥</div>
                    <div>No teams assigned yet</div>
                    <div className="team-empty-hint">Ask your administrator to create a team and assign you.</div>
                </div>
            )}
        </div>
    )
}
