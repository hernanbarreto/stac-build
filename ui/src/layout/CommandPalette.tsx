/**
 * CommandPalette — Ctrl+K (§5): existing menu / toolbar actions, navigation
 * to sessions and instances, shortcuts shown as <Kbd>. Subsequence match
 * over label + keywords, grouped, keyboard driven. No new actions live
 * here: App builds the command list from the handlers it already has.
 */
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { CornerDownLeft, Search } from 'lucide-react'
import { Kbd } from '../components/ui/Kbd'
import { useT } from '../i18n'

export interface Command {
  id: string
  label: string
  group: string
  icon?: ReactNode
  shortcut?: string
  keywords?: string
  disabled?: boolean
  hint?: string
  run: () => void
}

function score(query: string, text: string): number {
  if (!query) return 1
  const q = query.toLowerCase(), s = text.toLowerCase()
  if (s.includes(q)) return 100 - s.indexOf(q)
  let qi = 0
  for (let i = 0; i < s.length && qi < q.length; i++) if (s[i] === q[qi]) qi++
  return qi === q.length ? 10 : 0
}

export function CommandPalette({ open, onClose, commands }: { open: boolean; onClose: () => void; commands: Command[] }) {
  const t = useT()
  const [query, setQuery] = useState('')
  const [idx, setIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const results = useMemo(() => {
    const scored = commands
      .map(c => ({ c, s: Math.max(score(query, c.label), score(query, `${c.group} ${c.keywords ?? ''}`) * 0.5) }))
      .filter(r => r.s > 0)
      .sort((a, b) => b.s - a.s || a.c.group.localeCompare(b.c.group))
    return scored.slice(0, 60).map(r => r.c)
  }, [commands, query])

  useEffect(() => { if (open) { setQuery(''); setIdx(0); setTimeout(() => inputRef.current?.focus(), 0) } }, [open])
  useEffect(() => { setIdx(0) }, [query])
  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[data-active="true"]')?.scrollIntoView({ block: 'nearest' })
  }, [idx])

  if (!open) return null
  const run = (c: Command) => { if (c.disabled) return; onClose(); c.run() }
  let lastGroup = ''
  return createPortal(
    <div className="stac-palette-layer" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}>
      <div role="dialog" aria-label={t('palette.title')} className="stac-palette">
        <div className="stac-palette__search">
          <Search aria-hidden />
          <input ref={inputRef} className="stac-palette__input" value={query} placeholder={t('palette.placeholder')} aria-label={t('palette.placeholder')}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Escape') onClose()
              else if (e.key === 'ArrowDown') { e.preventDefault(); setIdx(i => Math.min(results.length - 1, i + 1)) }
              else if (e.key === 'ArrowUp') { e.preventDefault(); setIdx(i => Math.max(0, i - 1)) }
              else if (e.key === 'Enter' && results[idx]) run(results[idx])
            }} />
          <Kbd combo="Esc" />
        </div>
        <div className="stac-palette__list" role="listbox" ref={listRef}>
          {results.length === 0 && <div className="stac-palette__empty">{t('palette.noResults')}</div>}
          {results.map((c, i) => {
            const showGroup = c.group !== lastGroup
            lastGroup = c.group
            return (
              <div key={c.id}>
                {showGroup && <div className="stac-palette__group">{c.group}</div>}
                <button type="button" role="option" aria-selected={i === idx} data-active={i === idx} disabled={c.disabled}
                  className={`stac-palette__item ${i === idx ? 'stac-palette__item--active' : ''}`.trim()}
                  onMouseEnter={() => setIdx(i)} onClick={() => run(c)}>
                  <span className="stac-palette__icon" aria-hidden>{c.icon}</span>
                  <span className="stac-palette__label">{c.label}</span>
                  {c.hint && <span className="stac-palette__hint">{c.hint}</span>}
                  {c.shortcut && <Kbd combo={c.shortcut} />}
                  {i === idx && <CornerDownLeft className="stac-palette__enter" aria-hidden />}
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>,
    document.body,
  )
}
