/**
 * BottomDock — collapsible strip under the viewport (§5): Console · Jobs ·
 * Timeline · Report. Height is dragged from its top edge; App passes the
 * content per tab.
 */
import type { ReactNode } from 'react'
import { PanelBottomClose } from 'lucide-react'
import { Tabs, type TabItem } from '../components/ui/Tabs'
import { IconButton } from '../components/ui/IconButton'
import { startDrag, useLayout, type DockTab } from './LayoutContext'
import { useT } from '../i18n'

export function BottomDock({ tabs, panels, actions }: { tabs: TabItem<DockTab>[]; panels: Record<DockTab, ReactNode>; actions?: ReactNode }) {
  const { state, setDockTab, setDockHeight, setDockOpen } = useLayout()
  const t = useT()
  return (
    <section className="stac-dock" aria-label={t('layout.dock')}>
      <div className="stac-resize stac-resize--top" role="separator" aria-orientation="horizontal" aria-label={t('layout.resizeDock')}
        onMouseDown={e => startDrag(e, 'y', state.dockHeight, -1, setDockHeight)} />
      <div className="stac-dock__head">
        <Tabs items={tabs} value={state.dockTab} onChange={setDockTab} ariaLabel={t('layout.dockTabs')} size="sm" />
        <div className="stac-dock__actions">
          {actions}
          <IconButton label={t('layout.hideDock')} shortcut="Mod+`" icon={<PanelBottomClose aria-hidden />} onClick={() => setDockOpen(false)} size="sm" />
        </div>
      </div>
      <div className="stac-dock__body">{panels[state.dockTab]}</div>
    </section>
  )
}
