/**
 * Skeleton — loading placeholder: a stack of muted bars with a slow opacity
 * pulse (a progress indicator, the one allowed non-user animation).
 */
export function Skeleton({ lines = 3, className = '' }: { lines?: number; className?: string }) {
  return (
    <div className={`stac-skeleton ${className}`.trim()} aria-hidden>
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className={`stac-skeleton__bar ${i % 3 === 2 ? 'stac-skeleton__bar--short' : ''}`.trim()} />
      ))}
    </div>
  )
}
