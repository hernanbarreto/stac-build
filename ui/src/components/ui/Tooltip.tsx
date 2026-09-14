/**
 * Tooltip — 150 ms delay, --fs-0, no arrow (prompt_ui.txt §6).
 * Renders into a portal so it is never clipped by a scrolling panel; the
 * position is passed as CSS variables (`--tip-x`, `--tip-y`), the rule
 * lives in Tooltip.css. Shows on hover AND keyboard focus.
 */
import { cloneElement, useCallback, useEffect, useRef, useState, type ReactElement, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { Kbd } from './Kbd'

export type TooltipSide = 'top' | 'bottom' | 'left' | 'right'

interface TooltipProps {
  label: ReactNode
  shortcut?: string
  side?: TooltipSide
  children: ReactElement
  disabled?: boolean
}

const DELAY_MS = 150
const GAP = 8

export function Tooltip({ label, shortcut, side = 'bottom', children, disabled }: TooltipProps) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const anchorRef = useRef<HTMLElement | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const show = useCallback(() => {
    if (disabled || !label) return
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => {
      const el = anchorRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const x = side === 'left' ? r.left - GAP : side === 'right' ? r.right + GAP : r.left + r.width / 2
      const y = side === 'top' ? r.top - GAP : side === 'bottom' ? r.bottom + GAP : r.top + r.height / 2
      setPos({ x, y })
    }, DELAY_MS)
  }, [disabled, label, side])

  const hide = useCallback(() => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = null
    setPos(null)
  }, [])

  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  useEffect(() => {
    if (!pos) return
    const onScroll = () => hide()
    window.addEventListener('scroll', onScroll, true)
    window.addEventListener('resize', onScroll)
    return () => { window.removeEventListener('scroll', onScroll, true); window.removeEventListener('resize', onScroll) }
  }, [pos, hide])

  const childProps = children.props as Record<string, unknown>
  const child = cloneElement(children, {
    ref: (node: HTMLElement | null) => {
      anchorRef.current = node
      const orig = (children as unknown as { ref?: unknown }).ref
      if (typeof orig === 'function') orig(node)
      else if (orig && typeof orig === 'object') (orig as { current: HTMLElement | null }).current = node
    },
    onMouseEnter: (e: React.MouseEvent) => { (childProps.onMouseEnter as ((e: React.MouseEvent) => void) | undefined)?.(e); show() },
    onMouseLeave: (e: React.MouseEvent) => { (childProps.onMouseLeave as ((e: React.MouseEvent) => void) | undefined)?.(e); hide() },
    onFocus: (e: React.FocusEvent) => { (childProps.onFocus as ((e: React.FocusEvent) => void) | undefined)?.(e); show() },
    onBlur: (e: React.FocusEvent) => { (childProps.onBlur as ((e: React.FocusEvent) => void) | undefined)?.(e); hide() },
    onMouseDown: (e: React.MouseEvent) => { (childProps.onMouseDown as ((e: React.MouseEvent) => void) | undefined)?.(e); hide() },
  } as Record<string, unknown>)

  return (
    <>
      {child}
      {pos && createPortal(
        <div
          role="tooltip"
          className={`stac-tooltip stac-tooltip--${side}`}
          style={{ '--tip-x': `${pos.x}px`, '--tip-y': `${pos.y}px` } as React.CSSProperties}
        >
          <span className="stac-tooltip__label">{label}</span>
          {shortcut && <Kbd combo={shortcut} />}
        </div>,
        document.body,
      )}
    </>
  )
}
