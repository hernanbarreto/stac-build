/**
 * EmptyState — icon, one line saying what this is, one button with the first
 * step (§6, §9). Written as an invitation, never as an apology.
 */
import type { ReactNode } from 'react'

interface EmptyStateProps {
  icon?: ReactNode
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  compact?: boolean
  className?: string
}

export function EmptyState({ icon, title, description, action, compact = false, className = '' }: EmptyStateProps) {
  return (
    <div className={`stac-empty ${compact ? 'stac-empty--compact' : ''} ${className}`.trim()} role="status">
      {icon && <div className="stac-empty__icon" aria-hidden>{icon}</div>}
      <div className="stac-empty__title">{title}</div>
      {description && <div className="stac-empty__desc">{description}</div>}
      {action && <div className="stac-empty__action">{action}</div>}
    </div>
  )
}
