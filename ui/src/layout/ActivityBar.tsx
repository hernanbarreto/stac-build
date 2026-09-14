/**
 * ActivityBar — 48 px column of 20 px icons that toggle the left panel tabs;
 * bottom group for team / console / settings. Active item = 2 px orange
 * left border (selection), badge = count.
 */
import type { ReactNode } from 'react'
import { IconButton } from '../components/ui/IconButton'

export interface ActivityItem {
  id: string
  icon: ReactNode
  label: string
  shortcut?: string
  badge?: number | string
  active?: boolean
  disabled?: boolean
  onClick: () => void
}

export function ActivityBar({ items, bottom, ariaLabel }: { items: ActivityItem[]; bottom: ActivityItem[]; ariaLabel: string }) {
  const render = (it: ActivityItem) => (
    <div key={it.id} className={`stac-activity__slot ${it.active ? 'stac-activity__slot--active' : ''}`.trim()}>
      <IconButton size="lg" label={it.label} shortcut={it.shortcut} icon={it.icon} badge={it.badge} active={it.active} disabled={it.disabled} onClick={it.onClick} tooltipSide="right" className="stac-activity__btn" />
    </div>
  )
  return (
    <nav className="stac-activity" aria-label={ariaLabel}>
      <div className="stac-activity__group">{items.map(render)}</div>
      <div className="stac-activity__group stac-activity__group--bottom">{bottom.map(render)}</div>
    </nav>
  )
}
