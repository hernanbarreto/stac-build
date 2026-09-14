/**
 * Kbd — keyboard shortcut in mono --fs-0. `combo="Ctrl+K"` renders one key
 * cap per segment; "Mod" becomes ⌘ on macOS and Ctrl elsewhere — written as
 * the word, never as a glyph icon.
 */
const IS_MAC = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

export function formatShortcut(combo: string): string[] {
  return combo.split('+').map(k => (k === 'Mod' ? (IS_MAC ? 'Cmd' : 'Ctrl') : k))
}

export function Kbd({ combo, className = '' }: { combo: string; className?: string }) {
  return (
    <span className={`stac-kbd ${className}`.trim()} aria-label={combo}>
      {formatShortcut(combo).map((k, i) => (
        <kbd key={i} className="stac-kbd__key">{k}</kbd>
      ))}
    </span>
  )
}
