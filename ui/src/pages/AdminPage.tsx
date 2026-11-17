// STAC-BUILD: Admin Panel — User & Team Management
// Create, edit, and manage users and teams (admin only)

import { useState, useEffect, useCallback, useRef } from 'react'
import { findByName, registerCommands, type AdminCommandId, type Executors, type Outcome } from '../commands/registry'
import { Settings, Users, Building2, Plus, Save, Trash2, Pencil, Power, PowerOff, ChevronDown, ChevronRight, X, FolderOpen } from 'lucide-react'
import { useAuth, AuthUser } from '../context/AuthContext'
import { useConfirmDialog } from '../components/ConfirmDialog'
import { Dialog } from '../components/ui/Dialog'
import { Tabs } from '../components/ui/Tabs'
import { Button } from '../components/ui/Button'
import { IconButton } from '../components/ui/IconButton'
import { Input, Select } from '../components/ui/Field'
import { Badge } from '../components/ui/Badge'
import { Section, Row, Stack } from '../components/ui/Panel'
import { EmptyState } from '../components/ui/EmptyState'
import { Skeleton } from '../components/ui/Skeleton'
import { useFmt, useT } from '../i18n'

interface Props { onClose: () => void }

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

const ROLES = ['admin', 'manager', 'editor', 'viewer', 'recorder']

export default function AdminPage({ onClose }: Props) {
  const t = useT()
  const fmt = useFmt()
  const { token, user: currentUser } = useAuth()
  const { confirmDanger, alert, dialogElement } = useConfirmDialog()
  const [activeTab, setActiveTab] = useState<'users' | 'teams'>('users')

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

  const headers: HeadersInit = { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }

  const fetchUsers = useCallback(async () => {
    try {
      const res = await fetch('/api/auth/users', { headers })
      if (res.ok) { const data = await res.json(); setUsers(data.users) }
    } catch { /* ignore */ }
    setLoading(false)
  }, [token])

  const fetchTeams = useCallback(async () => {
    try {
      const res = await fetch('/api/teams', { headers })
      if (res.ok) { const data = await res.json(); setTeams(data.teams || []) }
    } catch { /* ignore */ }
    setTeamsLoading(false)
  }, [token])

  const fetchAllSessions = useCallback(async () => {
    try {
      const res = await fetch('/sessions', { headers })
      if (res.ok) { const data = await res.json(); setAllSessions(Array.isArray(data) ? data : []) }
    } catch { /* ignore */ }
  }, [token])

  useEffect(() => { fetchUsers(); fetchTeams(); fetchAllSessions() }, [fetchUsers, fetchTeams, fetchAllSessions])

  const handleCreate = async () => {
    const res = await fetch('/api/auth/users', { method: 'POST', headers, body: JSON.stringify({ username: newUsername, password: newPassword, email: newEmail || null, full_name: newFullName || null, role: newRole }) })
    if (res.ok) { setShowCreate(false); setNewUsername(''); setNewPassword(''); setNewEmail(''); setNewFullName(''); setNewRole('viewer'); fetchUsers() }
    else { const err = await res.json().catch(() => ({})); alert(err.detail || t('admin.createUserFailed'), t('admin.error')) }
  }
  const handleToggleActive = async (u: AuthUser) => { await fetch(`/api/auth/users/${u.id}`, { method: 'PUT', headers, body: JSON.stringify({ is_active: !u.is_active }) }); fetchUsers() }
  const handleRoleChange = async (u: AuthUser, role: string) => { await fetch(`/api/auth/users/${u.id}`, { method: 'PUT', headers, body: JSON.stringify({ role }) }); fetchUsers() }
  const startEdit = (u: AuthUser) => { setEditingUserId(editingUserId === u.id ? null : u.id); setEditFullName(u.full_name || ''); setEditEmail(u.email || ''); setEditPassword('') }
  const handleSaveEdit = async (u: AuthUser) => {
    const body: Record<string, string | null> = {}
    if (editFullName !== (u.full_name || '')) body.full_name = editFullName || null
    if (editEmail !== (u.email || '')) body.email = editEmail || null
    if (editPassword) body.password = editPassword
    if (Object.keys(body).length === 0) { setEditingUserId(null); return }
    const res = await fetch(`/api/auth/users/${u.id}`, { method: 'PUT', headers, body: JSON.stringify(body) })
    if (res.ok) { setEditingUserId(null); setEditPassword(''); fetchUsers() }
    else { const err = await res.json().catch(() => ({})); alert(err.detail || t('admin.updateUserFailed'), t('admin.error')) }
  }
  const handleDelete = async (u: AuthUser) => {
    const ok = await confirmDanger(t('admin.deleteUserMessage', { name: u.username }), t('admin.deleteUserTitle'))
    if (!ok) return
    await fetch(`/api/auth/users/${u.id}`, { method: 'DELETE', headers })
    fetchUsers()
  }

  const handleCreateTeam = async () => {
    if (!teamName || !teamManagerId) return
    const res = await fetch('/api/teams', { method: 'POST', headers, body: JSON.stringify({ name: teamName, description: teamDesc || null, manager_id: teamManagerId }) })
    if (res.ok) { setShowCreateTeam(false); setTeamName(''); setTeamDesc(''); setTeamManagerId(''); fetchTeams() }
    else { const err = await res.json().catch(() => ({})); alert(err.detail || t('admin.createTeamFailed'), t('admin.error')) }
  }
  const handleDeleteTeam = async (teamId: number, name: string) => {
    const ok = await confirmDanger(t('admin.deleteTeamMessage', { name }), t('admin.deleteTeamTitle'))
    if (!ok) return
    await fetch(`/api/teams/${teamId}`, { method: 'DELETE', headers })
    fetchTeams()
  }
  const handleAddMember = async (teamId: number) => {
    if (!addMemberUserId) return
    const res = await fetch(`/api/teams/${teamId}/members`, { method: 'POST', headers, body: JSON.stringify({ user_id: addMemberUserId }) })
    if (res.ok) { setAddMemberUserId(''); fetchTeams() }
    else { const err = await res.json().catch(() => ({})); alert(err.detail || t('admin.addMemberFailed'), t('admin.error')) }
  }
  const handleRemoveMember = async (teamId: number, userId: number, username: string) => {
    const ok = await confirmDanger(t('admin.removeMemberMessage', { name: username }), t('team.removeMemberTitle'))
    if (!ok) return
    await fetch(`/api/teams/${teamId}/members/${userId}`, { method: 'DELETE', headers })
    fetchTeams()
  }
  const handleAssignSession = async (teamId: number) => {
    if (!addSessionId.trim()) return
    const res = await fetch(`/api/teams/${teamId}/sessions`, { method: 'POST', headers, body: JSON.stringify({ session_id: addSessionId.trim() }) })
    if (res.ok) { setAddSessionId(''); fetchTeams() }
    else { const err = await res.json().catch(() => ({})); alert(err.detail || t('admin.assignSessionFailed'), t('admin.error')) }
  }
  const handleUnassignSession = async (teamId: number, sessionId: string) => { await fetch(`/api/teams/${teamId}/sessions/${sessionId}`, { method: 'DELETE', headers }); fetchTeams() }
  const handleCreateSession = async (teamId: number) => {
    if (!newSessionName.trim()) return
    const res = await fetch('/sessions', { method: 'POST', headers, body: JSON.stringify({ name: newSessionName.trim() }) })
    if (res.ok) {
      const data = await res.json()
      await fetch(`/api/teams/${teamId}/sessions`, { method: 'POST', headers, body: JSON.stringify({ session_id: data.session_id }) })
      setNewSessionName(''); setShowCreateSession(false); fetchTeams(); fetchAllSessions()
    } else { const err = await res.json().catch(() => ({})); alert(err.detail || t('admin.createSessionFailed'), t('admin.error')) }
  }
  const handleChangeManager = async (teamId: number, managerId: number) => { await fetch(`/api/teams/${teamId}`, { method: 'PUT', headers, body: JSON.stringify({ manager_id: managerId }) }); fetchTeams() }

  // STAC's commands of the Administration window (server/commands/catalog/270_admin.yaml), while
  // it is open: each one is the request its control sends. A password is never an argument:
  // creating a user fills the form, the password is typed and the form saved by hand.
  const acmd = useRef({ users, teams, allSessions, headers, fetchUsers, fetchTeams, fetchAllSessions, currentId: currentUser?.id })
  acmd.current = { users, teams, allSessions, headers, fetchUsers, fetchTeams, fetchAllSessions, currentId: currentUser?.id }
  useEffect(() => {
    const named = <T,>(items: T[], spoken: string, names: (x: T) => string[]): T | Outcome => {
      const found = findByName(items, spoken, names)
      if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.slice(0, 8).map(x => names(x)[0]).join(', ') }
      return found.item ?? { ok: false, reason: 'not_found', detail: spoken }
    }
    const userBy = (name: string) => named(acmd.current.users, name, u => [u.username, ...(u.full_name ? [u.full_name] : [])])
    const teamBy = (name: string) => named(acmd.current.teams, name, x => [x.name])
    const send = async (method: string, url: string, body?: unknown): Promise<Outcome> => {
      try {
        const r = await fetch(url, { method, headers: acmd.current.headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }) })
        if (r.ok) return { ok: true }
        const d = await r.json().catch(() => ({}))
        return { ok: false, reason: 'failed', detail: String(d.detail || `HTTP ${r.status}`) }
      } catch (e) { return { ok: false, reason: 'failed', detail: e instanceof Error ? e.message : String(e) } }
    }
    const then = (o: Outcome, refresh: () => unknown, detail: string): Outcome => { if (!o.ok) return o; refresh(); return { ok: true, detail } }
    const memberOf = ({ team, user }: { team: string; user: string }) => {
      const tm = teamBy(team)
      if ('ok' in tm) return tm
      const m = named(tm.members, user, x => [x.username, ...(x.full_name ? [x.full_name] : [])])
      return 'ok' in m ? m : { tm, m }
    }
    const executors: Executors<AdminCommandId> = {
      'admin.set_tab': ({ tab }) => { setActiveTab(tab); return { ok: true } },
      'admin.create_user': ({ username, role, full_name, email }) => {
        setActiveTab('users'); setShowCreate(true)
        setNewUsername(username); setNewRole(role); setNewFullName(full_name ?? ''); setNewEmail(email ?? ''); setNewPassword('')
        return { ok: true }
      },
      'admin.set_user_role': async ({ user, role }) => {
        const u = userBy(user)
        if ('ok' in u) return u
        return then(await send('PUT', `/api/auth/users/${u.id}`, { role }), acmd.current.fetchUsers, u.username)
      },
      'admin.toggle_user_active': async ({ user, on }) => {
        const u = userBy(user)
        if ('ok' in u) return u
        if (u.id === acmd.current.currentId && !on) return { ok: false, reason: 'not_allowed' }
        const o = then(await send('PUT', `/api/auth/users/${u.id}`, { is_active: on }), acmd.current.fetchUsers, u.username)
        return o.ok ? { ...o, state: on } : o
      },
      'admin.edit_user': async ({ user, full_name, email }) => {
        const u = userBy(user)
        if ('ok' in u) return u
        const body: Record<string, string> = {}
        if (full_name !== undefined) body.full_name = full_name
        if (email !== undefined) body.email = email
        if (!Object.keys(body).length) return { ok: false, reason: 'nothing_to_do' }
        return then(await send('PUT', `/api/auth/users/${u.id}`, body), acmd.current.fetchUsers, u.username)
      },
      'admin.delete_user': {
        prepare: ({ user }) => { const u = userBy(user); return 'ok' in u ? u : u.id === acmd.current.currentId ? { ok: false, reason: 'not_allowed' } : { ok: true, detail: u.username } },
        run: async ({ user }) => {
          const u = userBy(user)
          if ('ok' in u) return u
          if (u.id === acmd.current.currentId) return { ok: false, reason: 'not_allowed' }
          return then(await send('DELETE', `/api/auth/users/${u.id}`), acmd.current.fetchUsers, u.username)
        },
      },
      'admin.create_team': async ({ name, description, manager }) => {
        const m = named(acmd.current.users.filter(u => u.role === 'manager' || u.role === 'admin'), manager, u => [u.username, ...(u.full_name ? [u.full_name] : [])])
        if ('ok' in m) return m
        return then(await send('POST', '/api/teams', { name, description: description || null, manager_id: m.id }), acmd.current.fetchTeams, name)
      },
      'admin.delete_team': {
        prepare: ({ team }) => { const tm = teamBy(team); return 'ok' in tm ? tm : { ok: true, detail: tm.name } },
        run: async ({ team }) => {
          const tm = teamBy(team)
          if ('ok' in tm) return tm
          return then(await send('DELETE', `/api/teams/${tm.id}`), acmd.current.fetchTeams, tm.name)
        },
      },
      'admin.change_team_manager': async ({ team, manager }) => {
        const tm = teamBy(team)
        if ('ok' in tm) return tm
        const m = named(acmd.current.users.filter(u => u.role === 'manager' || u.role === 'admin'), manager, u => [u.username, ...(u.full_name ? [u.full_name] : [])])
        if ('ok' in m) return m
        return then(await send('PUT', `/api/teams/${tm.id}`, { manager_id: m.id }), acmd.current.fetchTeams, `${tm.name}: ${m.username}`)
      },
      'admin.add_team_member': async ({ team, user }) => {
        const tm = teamBy(team)
        if ('ok' in tm) return tm
        const u = userBy(user)
        if ('ok' in u) return u
        return then(await send('POST', `/api/teams/${tm.id}/members`, { user_id: u.id }), acmd.current.fetchTeams, `${u.username} → ${tm.name}`)
      },
      'admin.remove_team_member': {
        prepare: a => { const x = memberOf(a); return 'ok' in x ? x : { ok: true, detail: `${x.m.username} (${x.tm.name})` } },
        run: async a => {
          const x = memberOf(a)
          if ('ok' in x) return x
          return then(await send('DELETE', `/api/teams/${x.tm.id}/members/${x.m.user_id}`), acmd.current.fetchTeams, `${x.m.username} (${x.tm.name})`)
        },
      },
      'admin.assign_session': async ({ team, session }) => {
        const tm = teamBy(team)
        if ('ok' in tm) return tm
        const sn = named(acmd.current.allSessions, session, x => [x.id])
        if ('ok' in sn) return sn
        return then(await send('POST', `/api/teams/${tm.id}/sessions`, { session_id: sn.id }), acmd.current.fetchTeams, `${sn.id} → ${tm.name}`)
      },
      'admin.unassign_session': async ({ team, session }) => {
        const tm = teamBy(team)
        if ('ok' in tm) return tm
        const sn = named(tm.sessions, session, x => [x.session_id])
        if ('ok' in sn) return sn
        return then(await send('DELETE', `/api/teams/${tm.id}/sessions/${sn.session_id}`), acmd.current.fetchTeams, `${sn.session_id} (${tm.name})`)
      },
      'admin.create_team_session': async ({ team, name }) => {
        const tm = teamBy(team)
        if ('ok' in tm) return tm
        try {
          const res = await fetch('/sessions', { method: 'POST', headers: acmd.current.headers, body: JSON.stringify({ name: name.trim() }) })
          const data = await res.json().catch(() => ({}))
          if (!res.ok) return { ok: false, reason: 'failed', detail: String(data.detail || `HTTP ${res.status}`) }
          const o = await send('POST', `/api/teams/${tm.id}/sessions`, { session_id: data.session_id })
          return then(o, () => { acmd.current.fetchTeams(); acmd.current.fetchAllSessions() }, `${data.session_id} → ${tm.name}`)
        } catch (e) { return { ok: false, reason: 'failed', detail: e instanceof Error ? e.message : String(e) } }
      },
    }
    return registerCommands(executors)
  }, [])

  const roleOptions = ROLES.map(r => ({ value: r, label: t(`role.${r}`) }))
  const managers = users.filter(u => u.role === 'manager' || u.role === 'admin')

  return (
    <>
      <Dialog open size="lg" title={t('admin.title')} icon={<Settings aria-hidden />} onClose={onClose}>
        <Tabs<'users' | 'teams'> ariaLabel={t('admin.title')} value={activeTab} onChange={setActiveTab} fill items={[
          { id: 'users', label: t('admin.users'), icon: <Users aria-hidden />, badge: users.length },
          { id: 'teams', label: t('admin.teams'), icon: <Building2 aria-hidden />, badge: teams.length },
        ]} />

        {activeTab === 'users' && (
          <Stack gap={3}>
            <Row justify="between">
              <Button variant={showCreate ? 'secondary' : 'primary'} icon={showCreate ? <X aria-hidden /> : <Plus aria-hidden />} onClick={() => setShowCreate(!showCreate)}>{showCreate ? t('common.cancel') : t('admin.newUser')}</Button>
              <span className="stac-section__hint">{t.plural('admin.userCount', users.length)}</span>
            </Row>
            {showCreate && (
              <Section flush title={t('admin.newUser')}>
                <Row><Input placeholder={t('admin.usernameRequired')} value={newUsername} onChange={e => setNewUsername(e.target.value)} /><Input placeholder={t('admin.passwordRequired')} type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)} /></Row>
                <Row><Input placeholder={t('admin.fullName')} value={newFullName} onChange={e => setNewFullName(e.target.value)} /><Input placeholder={t('admin.email')} value={newEmail} onChange={e => setNewEmail(e.target.value)} /></Row>
                <Row><Select<string> value={newRole} onChange={setNewRole} options={roleOptions} aria-label={t('admin.role')} /><Button variant="primary" icon={<Save aria-hidden />} onClick={handleCreate} disabled={!newUsername || !newPassword}>{t('admin.createUser')}</Button></Row>
              </Section>
            )}
            {loading && <Skeleton lines={4} />}
            <Stack gap={1}>
              {users.map(u => (
                <div key={u.id} className={`stac-admin__row ${!u.is_active ? 'stac-admin__row--disabled' : ''}`.trim()}>
                  <Row className="stac-admin__head">
                    <span className="stac-admin__avatar" aria-hidden>{u.username[0].toUpperCase()}</span>
                    <span className="stac-admin__info">
                      <span className="stac-admin__name">{u.full_name || u.username}{u.id === currentUser?.id && <Badge tone="measure" size="sm">{t('admin.you')}</Badge>}</span>
                      <span className="stac-admin__meta stac-mono">@{u.username} {u.email ? `— ${u.email}` : ''}{u.last_login ? ` — ${t('admin.lastLogin', { date: fmt.date(u.last_login) })}` : ''}</span>
                    </span>
                    <Select<string> size="sm" value={u.role} onChange={v => handleRoleChange(u, v)} options={roleOptions} disabled={u.id === currentUser?.id} aria-label={t('admin.role')} className="stac-admin__role" />
                    <IconButton size="sm" label={u.is_active ? t('admin.disable') : t('admin.enable')} icon={u.is_active ? <Power aria-hidden /> : <PowerOff aria-hidden />} onClick={() => handleToggleActive(u)} disabled={u.id === currentUser?.id} tone={u.is_active ? 'default' : 'default'} />
                    <IconButton size="sm" label={t('admin.editUser')} icon={<Pencil aria-hidden />} onClick={() => startEdit(u)} disabled={u.id === currentUser?.id} active={editingUserId === u.id} />
                    <IconButton size="sm" variant="danger" label={t('admin.deleteUser')} icon={<Trash2 aria-hidden />} onClick={() => handleDelete(u)} disabled={u.id === currentUser?.id} />
                  </Row>
                  {editingUserId === u.id && (
                    <Stack gap={2} className="stac-admin__edit">
                      <Row><Input size="sm" placeholder={t('admin.fullName')} value={editFullName} onChange={e => setEditFullName(e.target.value)} /><Input size="sm" placeholder={t('admin.email')} value={editEmail} onChange={e => setEditEmail(e.target.value)} /></Row>
                      <Row><Input size="sm" placeholder={t('admin.newPasswordHint')} type="password" value={editPassword} onChange={e => setEditPassword(e.target.value)} /><Button size="sm" variant="primary" icon={<Save aria-hidden />} onClick={() => handleSaveEdit(u)}>{t('common.save')}</Button></Row>
                    </Stack>
                  )}
                </div>
              ))}
            </Stack>
          </Stack>
        )}

        {activeTab === 'teams' && (
          <Stack gap={3}>
            <Row justify="between">
              <Button variant={showCreateTeam ? 'secondary' : 'primary'} icon={showCreateTeam ? <X aria-hidden /> : <Plus aria-hidden />} onClick={() => setShowCreateTeam(!showCreateTeam)}>{showCreateTeam ? t('common.cancel') : t('admin.newTeam')}</Button>
              <span className="stac-section__hint">{t.plural('admin.teamCount', teams.length)}</span>
            </Row>
            {showCreateTeam && (
              <Section flush title={t('admin.newTeam')}>
                <Row><Input placeholder={t('admin.teamNameRequired')} value={teamName} onChange={e => setTeamName(e.target.value)} /><Input placeholder={t('admin.description')} value={teamDesc} onChange={e => setTeamDesc(e.target.value)} /></Row>
                <Row>
                  <Select<string> value={teamManagerId === '' ? '' : String(teamManagerId)} onChange={v => setTeamManagerId(v ? Number(v) : '')} placeholder={t('admin.selectManager')} options={managers.map(m => ({ value: String(m.id), label: `${m.full_name || m.username} (${t(`role.${m.role}`)})` }))} aria-label={t('admin.manager')} />
                  <Button variant="primary" icon={<Save aria-hidden />} onClick={handleCreateTeam} disabled={!teamName || !teamManagerId}>{t('admin.createTeam')}</Button>
                </Row>
              </Section>
            )}
            {teamsLoading && <Skeleton lines={3} />}
            {!teamsLoading && teams.length === 0 && <EmptyState compact icon={<Building2 aria-hidden />} title={t('admin.noTeams')} description={t('admin.noTeamsDesc')} />}
            <Stack gap={1}>
              {teams.map(team => (
                <div key={team.id} className="stac-admin__row">
                  <div className="stac-row stac-admin__head stac-admin__head--clickable" onClick={() => setExpandedTeam(expandedTeam === team.id ? null : team.id)}>
                    <span className="stac-admin__chevron" aria-hidden>{expandedTeam === team.id ? <ChevronDown /> : <ChevronRight />}</span>
                    <span className="stac-admin__info">
                      <span className="stac-admin__name">{team.name}</span>
                      <span className="stac-admin__meta">{t('admin.teamMeta', { manager: team.manager_name, members: fmt.integer(team.members.length), sessions: fmt.integer(team.sessions.length) })}</span>
                    </span>
                    <IconButton size="sm" variant="danger" label={t('admin.deleteTeam')} icon={<Trash2 aria-hidden />} onClick={e => { e.stopPropagation(); handleDeleteTeam(team.id, team.name) }} />
                  </div>
                  {expandedTeam === team.id && (
                    <Stack gap={3} className="stac-admin__edit">
                      {team.description && <p className="stac-section__hint">{team.description}</p>}
                      <Section flush title={t('admin.manager')}>
                        <Select<string> size="sm" value={String(team.manager_id)} onChange={v => handleChangeManager(team.id, Number(v))} options={managers.map(m => ({ value: String(m.id), label: m.full_name || m.username }))} aria-label={t('admin.manager')} />
                      </Section>
                      <Section flush title={t('team.members')}>
                        {team.members.map(m => (
                          <Row key={m.user_id} className="stac-admin__member">
                            <span className="stac-admin__info">{m.full_name || m.username}</span>
                            <Badge size="sm">{t(`role.${m.role}`)}</Badge>
                            <IconButton size="sm" variant="danger" label={t('team.removeMember')} icon={<X aria-hidden />} onClick={() => handleRemoveMember(team.id, m.user_id, m.username)} />
                          </Row>
                        ))}
                        <Row>
                          <Select<string> size="sm" value={addMemberUserId === '' ? '' : String(addMemberUserId)} onChange={v => setAddMemberUserId(v ? Number(v) : '')} placeholder={t('admin.addMemberPlaceholder')}
                            options={users.filter(u => !team.members.some(m => m.user_id === u.id)).map(u => ({ value: String(u.id), label: `${u.full_name || u.username} (${t(`role.${u.role}`)})` }))} aria-label={t('team.addMember')} />
                          <IconButton size="sm" variant="primary" label={t('team.addMember')} icon={<Plus aria-hidden />} onClick={() => handleAddMember(team.id)} disabled={!addMemberUserId} />
                        </Row>
                      </Section>
                      <Section flush title={t('admin.assignedSessions')} actions={
                        <Button size="sm" variant="ghost" onClick={() => { setShowCreateSession(!showCreateSession); setNewSessionName('') }}>{showCreateSession ? t('admin.backToList') : t('admin.createSession')}</Button>}>
                        {team.sessions.map(s => (
                          <Row key={s.session_id} className="stac-admin__member">
                            <FolderOpen aria-hidden className="stac-admin__icon" /><span className="stac-admin__info stac-mono">{s.session_id}</span>
                            <IconButton size="sm" variant="danger" label={t('admin.unassign')} icon={<X aria-hidden />} onClick={() => handleUnassignSession(team.id, s.session_id)} />
                          </Row>
                        ))}
                        <Row>
                          {!showCreateSession ? (
                            <>
                              <Select<string> size="sm" value={addSessionId} onChange={setAddSessionId} placeholder={t('admin.selectSession')}
                                options={allSessions.filter(s => !team.sessions.some(ts => ts.session_id === s.id)).map(s => ({ value: s.id, label: `${s.id} (${t.plural('sessions.frames', s.frame_count)}${s.has_cloud ? `, ${t('admin.hasCloud')}` : ''})` }))} aria-label={t('admin.selectSession')} />
                              <IconButton size="sm" variant="primary" label={t('admin.assign')} icon={<Plus aria-hidden />} onClick={() => handleAssignSession(team.id)} disabled={!addSessionId} />
                            </>
                          ) : (
                            <>
                              <Input size="sm" placeholder={t('admin.newSessionPlaceholder')} value={newSessionName} onChange={e => setNewSessionName(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleCreateSession(team.id)} />
                              <IconButton size="sm" variant="primary" label={t('sessions.create')} icon={<Save aria-hidden />} onClick={() => handleCreateSession(team.id)} disabled={!newSessionName.trim()} />
                            </>
                          )}
                        </Row>
                      </Section>
                    </Stack>
                  )}
                </div>
              ))}
            </Stack>
          </Stack>
        )}
      </Dialog>
      {dialogElement}
    </>
  )
}
