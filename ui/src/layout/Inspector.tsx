/**
 * Inspector — the right region (§5): tabs Properties, Acta / Quality ,
 * Assistant, own scroll per tab, drag handle on its left edge. The content
 * of each tab is passed by App (`panels[tab]`); the tab strip is the only
 * UI this component owns.
 */
import type { ReactNode } from 'react'
import { PanelRightClose } from 'lucide-react'
import { Tabs, type TabItem } from '../components/ui/Tabs'
import { IconButton } from '../components/ui/IconButton'
import { startDrag, useLayout, type InspectorTab } from './LayoutContext'
import { useT } from '../i18n'

export function Inspector({ tabs, panels }: { tabs: TabItem<InspectorTab>[]; panels: Record<InspectorTab, ReactNode> }) {
  const { state, setInspectorTab, setInspectorWidth, setInspectorOpen } = useLayout()
  const t = useT()
  return (
    <aside className="stac-inspector" aria-label={t('layout.inspector')}>
      <div className="stac-resize stac-resize--left" role="separator" aria-orientation="vertical" aria-label={t('layout.resizePanel')}
        onMouseDown={e => startDrag(e, 'x', state.inspectorWidth, -1, setInspectorWidth)} />
      <div className="stac-inspector__head">
        <Tabs items={tabs} value={state.inspectorTab} onChange={setInspectorTab} ariaLabel={t('layout.inspectorTabs')} size="sm" fill className="stac-inspector__tabs" />
        <IconButton label={t('layout.hideInspector')} shortcut="Mod+J" icon={<PanelRightClose aria-hidden />} onClick={() => setInspectorOpen(false)} size="sm" />
      </div>
      <div className="stac-inspector__body">{panels[state.inspectorTab]}</div>
    </aside>
  )
}
