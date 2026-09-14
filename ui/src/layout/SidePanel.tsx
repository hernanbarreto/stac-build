/**
 * SidePanel — the left region: hosts one Panel (sessions / instances /
 * scans / BIM / team) and a drag handle on its right edge. Width lives in
 * LayoutContext and is written as `--left-w` on the app grid by App.
 */
import type { ReactNode } from 'react'
import { startDrag, useLayout } from './LayoutContext'
import { useT } from '../i18n'

export function SidePanel({ children }: { children: ReactNode }) {
  const { state, setLeftWidth } = useLayout()
  const t = useT()
  return (
    <div className="stac-side">
      {children}
      <div className="stac-resize stac-resize--right" role="separator" aria-orientation="vertical" aria-label={t('layout.resizePanel')}
        onMouseDown={e => startDrag(e, 'x', state.leftWidth, 1, setLeftWidth)} />
    </div>
  )
}
