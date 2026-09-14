/**
 * ObjectLibraryDialog — insert a mesh from the collection or from any
 * session into the scene as a reference (never touches the source GLB).
 */
import { Package } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Section, Stack } from '../components/ui/Panel'
import { EmptyState } from '../components/ui/EmptyState'
import { useFmt, useT } from '../i18n'

export interface LibraryItem { source: string; session?: string; name: string; url: string; size_mb: number }

export function ObjectLibraryDialog({ open, items, onClose, onPick }: { open: boolean; items: LibraryItem[]; onClose: () => void; onPick: (it: LibraryItem) => void }) {
  const t = useT()
  const fmt = useFmt()
  const groups = ['collection', ...Array.from(new Set(items.filter(o => o.source === 'project').map(o => o.session ?? '')))]
  return (
    <Dialog open={open} title={t('library.title')} onClose={onClose} icon={<Package aria-hidden />}>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t('library.description')}</p>
        {items.length === 0 && <EmptyState compact icon={<Package aria-hidden />} title={t('library.emptyTitle')} description={t('library.emptyDesc')} />}
        {groups.map(g => {
          const list = g === 'collection' ? items.filter(o => o.source === 'collection') : items.filter(o => o.session === g)
          if (!list.length) return null
          return (
            <Section key={g} title={g === 'collection' ? t('library.collection') : g} flush>
              <ul className="stac-library__list">
                {list.map(it => (
                  <li key={it.url}>
                    <button type="button" className="stac-library__item" onClick={() => onPick(it)}>
                      <span className="stac-library__name">{it.name}</span>
                      <span className="stac-library__size stac-mono">{fmt.bytesMb(it.size_mb)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </Section>
          )
        })}
      </Stack>
    </Dialog>
  )
}
