/**
 * AppHeader — 36 px, --surface-brand (prompt_ui.txt §5):
 * logo · menu bar (File / View / Tools / Help — the existing actions) ·
 * session ▾ · scan ▾ · search (Ctrl+K) · assistant toggle · user ▾.
 * Purely presentational: every entry is built by App from its handlers.
 */
import { ChevronRight, Search, Sparkles, User } from 'lucide-react'
import { Kbd } from '../components/ui/Kbd'
import { IconButton } from '../components/ui/IconButton'
import { MenuButton, type MenuEntry } from '../components/ui/Menu'
import { useT } from '../i18n'

export interface HeaderMenu {
  id: string
  label: string
  entries: MenuEntry[]
  disabled?: boolean
}

export interface HeaderSession { id: string; name: string; loaded: boolean; hasCloud: boolean }
export interface HeaderScan { key: string; label: string; date: string; kind: 'scan' | 'fused'; isReference: boolean }

interface AppHeaderProps {
  menus: HeaderMenu[]
  sessions: HeaderSession[]
  activeSession: string | null
  onSelectSession: (id: string) => void
  scans: HeaderScan[]
  activeScan: string | null
  onSelectScan: (key: string) => void
  onOpenPalette: () => void
  inspectorOpen: boolean
  onToggleInspector: () => void
  vlmStatus: 'up' | 'loading' | 'busy' | 'down' | null
  userName: string
  userEntries: MenuEntry[]
  connected: boolean
}

export function AppHeader({ menus, sessions, activeSession, onSelectSession, scans, activeScan, onSelectScan, onOpenPalette, inspectorOpen, onToggleInspector, vlmStatus, userName, userEntries, connected }: AppHeaderProps) {
  const t = useT()
  const sessionEntries: MenuEntry[] = sessions.length
    ? sessions.map(s => ({ id: s.id, label: s.name, checked: s.id === activeSession, disabled: !s.hasCloud, onSelect: () => onSelectSession(s.id), hint: s.loaded ? t('header.loaded') : undefined }))
    : [{ id: 'none', label: connected ? t('header.noSessions') : t('header.notConnected'), disabled: true, onSelect: () => {} }]
  const scanEntries: MenuEntry[] = scans.map(s => ({
    id: s.key,
    label: s.kind === 'fused' ? s.label : `${s.date} ${s.label}`,
    checked: s.key === activeScan,
    hint: s.isReference ? t('header.reference') : s.kind === 'fused' ? t('header.fused') : undefined,
    onSelect: () => onSelectScan(s.key),
  }))
  const activeScanLabel = scans.find(s => s.key === activeScan)
  const vlmLabel = vlmStatus === 'up' ? t('assistant.modelUp') : vlmStatus === 'loading' ? t('assistant.modelLoading') : vlmStatus === 'busy' ? t('assistant.modelBusy') : t('assistant.modelDown')

  return (
    <header className="stac-header">
      <div className="stac-header__brand">
        <img src="/favicon.png" alt="" className="stac-header__logo" />
        <span className="stac-header__name">{t('app.name')}</span>
      </div>
      <nav className="stac-header__menus" aria-label={t('header.menuBar')}>
        {menus.map(m => (
          <MenuButton key={m.id} label={m.label} entries={m.entries} ariaLabel={m.label} disabled={m.disabled} variant="bar" />
        ))}
      </nav>
      <div className="stac-header__context">
        <MenuButton variant="select" chevron ariaLabel={t('header.sessionMenu')} minWidth="md"
          label={activeSession ?? t('header.chooseSession')} entries={sessionEntries} />
        {activeSession && scans.length > 0 && (
          <>
            <ChevronRight className="stac-header__crumb" aria-hidden />
            <MenuButton variant="select" chevron ariaLabel={t('header.scanMenu')} minWidth="md"
              label={activeScanLabel ? (activeScanLabel.kind === 'fused' ? activeScanLabel.label : `${activeScanLabel.date} ${activeScanLabel.label}`) : t('header.chooseScan')}
              entries={scanEntries} />
          </>
        )}
      </div>
      <div className="stac-header__spacer" />
      <button type="button" className="stac-header__search" onClick={onOpenPalette} aria-label={t('palette.open')}>
        <Search aria-hidden />
        <span className="stac-header__search-label">{t('palette.placeholder')}</span>
        <Kbd combo="Mod+K" />
      </button>
      <IconButton label={`${t('assistant.title')} — ${vlmLabel}`} icon={<Sparkles aria-hidden />} tone="measure" active={inspectorOpen}
        className={vlmStatus === 'up' ? '' : 'stac-header__vlm--off'} onClick={onToggleInspector} shortcut="Mod+J" />
      <MenuButton variant="bar" ariaLabel={t('header.userMenu')} align="right" icon={<User aria-hidden />} label={userName} entries={userEntries} />
    </header>
  )
}
