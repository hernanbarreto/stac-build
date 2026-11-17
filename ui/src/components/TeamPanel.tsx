/**
 * STAC Build — Team Panel Component
 * Real-time team presence, chat, activity feed, member management
 * Hernán Barreto — Ingerop IN3
 */
import { useState, useEffect, useRef, useCallback } from 'react'
import { findByName, registerCommands, type Executors, type Outcome, type TeamCommandId } from '../commands/registry'
import { UserPlus, Users, MessageSquare, Activity, Send, Video, X, Crown, ClipboardList, Pencil, Eye, Camera, User } from 'lucide-react'
import type { ReactNode } from 'react'
import { useAuth } from '../context/AuthContext'
import { useConfirmDialog } from './ConfirmDialog'
import { Panel, Row, Stack } from './ui/Panel'
import { Tabs } from './ui/Tabs'
import { Button } from './ui/Button'
import { IconButton } from './ui/IconButton'
import { Input, Select } from './ui/Field'
import { Badge } from './ui/Badge'
import { Dialog } from './ui/Dialog'
import { EmptyState } from './ui/EmptyState'
import { useFmt, useT } from '../i18n'

interface TeamData { id: number; name: string; description: string | null; manager_id: number; manager_name: string; is_active: boolean; members: TeamMemberData[]; sessions: { session_id: string }[] }
interface TeamMemberData { user_id: number; username: string; full_name: string | null; role: string; avatar_url: string | null }
interface OnlineUser { user_id: number; username: string; task: string }
interface ChatMessage { id: number; team_id: number; user_id: number; username: string; content: string; timestamp: string }
interface ActivityEntry { id: number; user_id: number; username: string; action: string; detail: string | null; timestamp: string }
interface AvailableUser { id: number; username: string; full_name: string | null; role: string }
interface TeamPanelProps { onCallUser?: (userId: number, username: string) => void }

const ROLE_ICON: Record<string, ReactNode> = {
  admin: <Crown aria-hidden />, manager: <ClipboardList aria-hidden />, editor: <Pencil aria-hidden />, viewer: <Eye aria-hidden />, recorder: <Camera aria-hidden />,
}

type TeamTab = 'members' | 'chat' | 'activity'

export default function TeamPanel({ onCallUser }: TeamPanelProps) {
  const t = useT()
  const fmt = useFmt()
  const { user, token } = useAuth()
  const { confirmDanger, dialogElement } = useConfirmDialog()
  const [teams, setTeams] = useState<TeamData[]>([])
  const [selectedTeam, setSelectedTeam] = useState<number | null>(null)
  const [onlineUsers, setOnlineUsers] = useState<OnlineUser[]>([])
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [activity, setActivity] = useState<ActivityEntry[]>([])
  const [chatInput, setChatInput] = useState('')
  const [activeTab, setActiveTab] = useState<TeamTab>('members')
  const [addMemberOpen, setAddMemberOpen] = useState(false)
  const [availableUsers, setAvailableUsers] = useState<AvailableUser[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)

  const isManager = useCallback((teamId: number) => {
    const team = teams.find(x => x.id === teamId)
    return team && user && (team.manager_id === user.id || user.role === 'admin')
  }, [teams, user])

  const authHeaders = useCallback(() => ({ 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }), [token])

  const fetchTeams = useCallback(async () => {
    if (!token) return
    try {
      const res = await fetch('/api/teams', { headers: authHeaders() })
      const data = await res.json()
      setTeams(data.teams || [])
      if (!selectedTeam && data.teams?.length > 0) setSelectedTeam(data.teams[0].id)
    } catch (e) { console.error('[TeamPanel] Failed to fetch teams:', e) }
  }, [token, authHeaders, selectedTeam])

  useEffect(() => { fetchTeams() }, [fetchTeams])

  useEffect(() => {
    if (!selectedTeam || !token) return
    fetch(`/api/teams/${selectedTeam}/messages?limit=100`, { headers: authHeaders() }).then(r => r.json()).then(data => setChatMessages(data.messages || [])).catch(console.error)
  }, [selectedTeam, token, authHeaders])

  useEffect(() => {
    if (activeTab !== 'activity' || !selectedTeam || !token) return
    fetch(`/api/teams/${selectedTeam}/activity?limit=50`, { headers: authHeaders() }).then(r => r.json()).then(data => setActivity(data.activity || [])).catch(console.error)
  }, [activeTab, selectedTeam, token, authHeaders])

  useEffect(() => {
    if (!token) return
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${location.host}/ws/team`)
    wsRef.current = ws
    ws.onopen = () => { ws.send(JSON.stringify({ type: 'team_auth', token })) }
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'presence_update') setOnlineUsers(msg.online || [])
        else if (msg.type === 'team_message' && msg.message) setChatMessages(prev => [...prev, msg.message])
      } catch { /* ignore */ }
    }
    ws.onclose = () => { wsRef.current = null }
    return () => { ws.close() }
  }, [token])

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMessages])

  const sendMessage = useCallback(() => {
    if (!chatInput.trim() || !selectedTeam || !wsRef.current) return
    wsRef.current.send(JSON.stringify({ type: 'team_message', team_id: selectedTeam, content: chatInput.trim() }))
    setChatInput('')
  }, [chatInput, selectedTeam])

  const openAddMember = useCallback(async () => {
    try {
      const res = await fetch('/api/teams/available-users', { headers: authHeaders() })
      const data = await res.json()
      setAvailableUsers(data.users || [])
      setAddMemberOpen(true)
    } catch (e) { console.error('[TeamPanel] Failed to fetch users:', e) }
  }, [authHeaders])

  const addMember = useCallback(async (userId: number) => {
    if (!selectedTeam) return
    try {
      await fetch(`/api/teams/${selectedTeam}/members`, { method: 'POST', headers: authHeaders(), body: JSON.stringify({ user_id: userId }) })
      setAddMemberOpen(false)
      fetchTeams()
    } catch (e) { console.error('[TeamPanel] Failed to add member:', e) }
  }, [selectedTeam, authHeaders, fetchTeams])

  const removeMember = useCallback(async (userId: number) => {
    if (!selectedTeam) return
    const ok = await confirmDanger(t('team.removeMemberMessage'), t('team.removeMemberTitle'))
    if (!ok) return
    try {
      await fetch(`/api/teams/${selectedTeam}/members/${userId}`, { method: 'DELETE', headers: authHeaders() })
      fetchTeams()
    } catch (e) { console.error('[TeamPanel] Failed to remove member:', e) }
  }, [selectedTeam, authHeaders, fetchTeams, confirmDanger, t])

  const isOnline = (userId: number) => onlineUsers.some(u => u.user_id === userId)
  const getUserTask = (userId: number) => onlineUsers.find(u => u.user_id === userId)?.task || ''
  // STAC's commands of the team (server/commands/catalog/250_team.yaml), while this panel is open
  const tcmd = useRef({ teams, selectedTeam, isManager, authHeaders, fetchTeams, openAddMember, onCallUser, userId: user?.id })
  tcmd.current = { teams, selectedTeam, isManager, authHeaders, fetchTeams, openAddMember, onCallUser, userId: user?.id }
  useEffect(() => {
    const nothing: Outcome = { ok: false, reason: 'nothing_to_do' }
    const current = () => tcmd.current.teams.find(x => x.id === tcmd.current.selectedTeam)
    /** the selected team, for its manager */
    const managed = (): TeamData | Outcome => {
      const tm = current()
      if (!tm) return nothing
      return tcmd.current.isManager(tm.id) ? tm : { ok: false, reason: 'not_allowed' }
    }
    const named = <T,>(items: T[], spoken: string, names: (x: T) => string[]): T | Outcome => {
      const found = findByName(items, spoken, names)
      if (found.ambiguous) return { ok: false, reason: 'ambiguous', detail: found.ambiguous.slice(0, 8).map(x => names(x)[0]).join(', ') }
      return found.item ?? { ok: false, reason: 'not_found', detail: spoken }
    }
    const person = (m: { username: string; full_name: string | null }) => [m.username, ...(m.full_name ? [m.full_name] : [])]
    const member = (userName: string): { tm: TeamData; m: TeamMemberData } | Outcome => {
      const tm = managed()
      if ('ok' in tm) return tm
      const m = named(tm.members, userName, person)
      return 'ok' in m ? m : { tm, m }
    }
    const executors: Executors<TeamCommandId> = {
      'team.select': ({ team: name }) => {
        const tm = named(tcmd.current.teams, name, x => [x.name])
        if ('ok' in tm) return tm
        setSelectedTeam(tm.id); return { ok: true, detail: tm.name }
      },
      'team.set_tab': ({ tab }) => { setActiveTab(tab); return { ok: true } },
      'team.send_message': ({ text }) => {
        const tm = current()
        const ws = wsRef.current
        if (!tm || !text.trim()) return nothing
        if (!ws || ws.readyState !== WebSocket.OPEN) return { ok: false, reason: 'server_down' }
        ws.send(JSON.stringify({ type: 'team_message', team_id: tm.id, content: text.trim() }))
        return { ok: true, detail: tm.name }
      },
      'team.open_add_member': async () => { const tm = managed(); if ('ok' in tm) return tm; await tcmd.current.openAddMember(); return { ok: true } },
      'team.add_member': async ({ user: userName }) => {
        const tm = managed()
        if ('ok' in tm) return tm
        try {
          const res = await fetch('/api/teams/available-users', { headers: tcmd.current.authHeaders() })
          const u = named(((await res.json()).users || []) as AvailableUser[], userName, person)
          if ('ok' in u) return u
          const r = await fetch(`/api/teams/${tm.id}/members`, { method: 'POST', headers: tcmd.current.authHeaders(), body: JSON.stringify({ user_id: u.id }) })
          if (!r.ok) return { ok: false, reason: 'failed', detail: `HTTP ${r.status}` }
          setAddMemberOpen(false); tcmd.current.fetchTeams()
          return { ok: true, detail: `${u.username} → ${tm.name}` }
        } catch (e) { return { ok: false, reason: 'failed', detail: e instanceof Error ? e.message : String(e) } }
      },
      'team.remove_member': {
        prepare: ({ user: userName }) => { const x = member(userName); return 'ok' in x ? x : { ok: true, detail: `${x.m.username} (${x.tm.name})` } },
        run: async ({ user: userName }) => {
          const x = member(userName)
          if ('ok' in x) return x
          try {
            const r = await fetch(`/api/teams/${x.tm.id}/members/${x.m.user_id}`, { method: 'DELETE', headers: tcmd.current.authHeaders() })
            if (!r.ok) return { ok: false, reason: 'failed', detail: `HTTP ${r.status}` }
            tcmd.current.fetchTeams()
            return { ok: true, detail: `${x.m.username} (${x.tm.name})` }
          } catch (e) { return { ok: false, reason: 'failed', detail: e instanceof Error ? e.message : String(e) } }
        },
      },
      'call.start': ({ user: userName }) => {
        const everyone = tcmd.current.teams.flatMap(x => x.members).filter((m, k, all) => m.user_id !== tcmd.current.userId && all.findIndex(o => o.user_id === m.user_id) === k)
        if (!everyone.length) return nothing
        const m = named(everyone, userName, person)
        if ('ok' in m) return m
        tcmd.current.onCallUser?.(m.user_id, m.username)
        return { ok: true, detail: m.username }
      },
    }
    return registerCommands(executors)
  }, [])

  const team = teams.find(x => x.id === selectedTeam)

  if (!user) return null

  return (
    <Panel title={t('team.title')} subtitle={team ? t('team.meta', { members: fmt.integer(team.members.length), sessions: fmt.integer(team.sessions.length) }) : undefined}
      isEmpty={teams.length === 0}
      empty={<EmptyState icon={<Users aria-hidden />} title={t('team.emptyTitle')} description={t('team.emptyDesc')} />}
      toolbar={
        <Stack gap={2} className="stac-team__toolbar">
          {teams.length > 1 && (
            <Select<string> size="sm" value={selectedTeam != null ? String(selectedTeam) : ''} onChange={v => setSelectedTeam(Number(v))} options={teams.map(x => ({ value: String(x.id), label: x.name }))} aria-label={t('team.selectTeam')} />
          )}
          <Tabs<TeamTab> size="sm" fill ariaLabel={t('team.title')} value={activeTab} onChange={setActiveTab} items={[
            { id: 'members', label: t('team.members'), icon: <Users aria-hidden /> },
            { id: 'chat', label: t('team.chat'), icon: <MessageSquare aria-hidden /> },
            { id: 'activity', label: t('team.activity'), icon: <Activity aria-hidden /> },
          ]} />
        </Stack>
      }
      footer={selectedTeam && isManager(selectedTeam) && activeTab === 'members' ? <Button block icon={<UserPlus aria-hidden />} onClick={openAddMember}>{t('team.addMember')}</Button>
        : activeTab === 'chat' ? (
          <form className="stac-team__chatinput" onSubmit={e => { e.preventDefault(); sendMessage() }}>
            <Input size="sm" placeholder={t('team.messagePlaceholder', { team: team?.name || '' })} value={chatInput} onChange={e => setChatInput(e.target.value)} aria-label={t('team.chat')} />
            <IconButton type="submit" size="sm" variant="primary" label={t('assistant.send')} icon={<Send aria-hidden />} disabled={!chatInput.trim()} />
          </form>
        ) : undefined}>
      {activeTab === 'members' && team && (
        <Stack gap={1}>
          {team.members.filter(m => m.user_id !== user.id).map(member => {
            const online = isOnline(member.user_id)
            return (
              <Row key={member.user_id} className="stac-team__member">
                <span className="stac-team__avatar" aria-hidden>
                  {member.avatar_url ? <img className="stac-team__avatar-img" src={member.avatar_url} alt="" /> : <span>{(member.full_name || member.username)[0].toUpperCase()}</span>}
                  <span className={`stac-live-dot stac-team__presence ${online ? 'stac-live-dot--ok' : ''}`} />
                </span>
                <span className="stac-team__info">
                  <span className="stac-team__name">{member.full_name || member.username} <Badge size="sm">{ROLE_ICON[member.role] ?? <User aria-hidden />} {t(`role.${member.role}`)}</Badge></span>
                  <span className="stac-team__task">{online ? getUserTask(member.user_id) || t('team.online') : t('team.offline')}</span>
                </span>
                <IconButton size="sm" label={t('team.videoCall')} icon={<Video aria-hidden />} onClick={() => onCallUser?.(member.user_id, member.username)} />
                {isManager(team.id) && member.user_id !== team.manager_id && (
                  <IconButton size="sm" variant="danger" label={t('team.removeMember')} icon={<X aria-hidden />} onClick={() => removeMember(member.user_id)} />
                )}
              </Row>
            )
          })}
          {team.members.filter(m => m.user_id !== user.id).length === 0 && <EmptyState compact icon={<Users aria-hidden />} title={t('team.noOtherMembers')} />}
        </Stack>
      )}
      {activeTab === 'chat' && (
        <div className="stac-team__chat">
          {chatMessages.length === 0 && <EmptyState compact icon={<MessageSquare aria-hidden />} title={t('team.chatEmptyTitle', { team: team?.name || '' })} description={t('team.chatEmptyDesc', { n: team?.members.length ?? 0 })} />}
          {chatMessages.map(msg => (
            <div key={msg.id} className={`stac-team__bubble ${msg.user_id === user.id ? 'stac-team__bubble--mine' : ''}`.trim()}>
              {msg.user_id !== user.id && <span className="stac-team__author">{msg.username}</span>}
              <span className="stac-selectable">{msg.content}</span>
              <span className="stac-team__time stac-mono">{fmt.time(msg.timestamp)}</span>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>
      )}
      {activeTab === 'activity' && (
        <Stack gap={1}>
          {activity.map(entry => (
            <Row key={entry.id} className="stac-team__activity">
              <Activity aria-hidden className="stac-team__activity-icon" />
              <span className="stac-team__info">
                <span><strong>{entry.username}</strong> {t.has(`teamAction.${entry.action}`) ? t(`teamAction.${entry.action}`) : entry.action.replace(/_/g, ' ')}</span>
                {entry.detail && <span className="stac-team__task">{entry.detail}</span>}
              </span>
              <span className="stac-team__time stac-mono">{fmt.dateTime(entry.timestamp)}</span>
            </Row>
          ))}
          {activity.length === 0 && <EmptyState compact icon={<Activity aria-hidden />} title={t('team.activityEmptyTitle')} description={t('team.activityEmptyDesc')} />}
        </Stack>
      )}
      <Dialog open={addMemberOpen} size="sm" title={t('team.addMember')} onClose={() => setAddMemberOpen(false)}>
        <ul className="stac-library__list">
          {availableUsers.filter(u => !team?.members.some(m => m.user_id === u.id)).map(u => (
            <li key={u.id}>
              <button type="button" className="stac-library__item" onClick={() => addMember(u.id)}>
                <span className="stac-library__name">{u.full_name || u.username}</span>
                <Badge size="sm">{t(`role.${u.role}`)}</Badge>
              </button>
            </li>
          ))}
        </ul>
      </Dialog>
      {dialogElement}
    </Panel>
  )
}
