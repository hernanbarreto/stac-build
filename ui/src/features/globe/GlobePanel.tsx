/**
 * THE GLOBE'S SECTION OF THE LEFT PANEL (USER 2026-11-24: "en el panel izquierdo debe haber una opción con icono de
 * mundo … cuando seleccionamos el icono del mundo, muestra cesium con todos los pines donde están los proyectos").
 * The globe itself is drawn over the viewer's area (GlobeView); here, the works: the ones placed (go to it on the
 * globe, open it, place it again, take its place away) and the ones not placed yet (place it: a click on the globe).
 */
import { useState } from 'react'
import { Building2, Crosshair, Earth, FolderOpen, MapPin, MapPinOff, RefreshCw, Trash2 } from 'lucide-react'
import { Button } from '../../components/ui/Button'
import { EmptyState } from '../../components/ui/EmptyState'
import { IconButton } from '../../components/ui/IconButton'
import { Panel } from '../../components/ui/Panel'
import { useT } from '../../i18n'
import { formatPlace, type GeoProject } from './geo'
import './globePanel.css'

export interface GlobePanelProps {
  projects: GeoProject[]
  loading: boolean
  error: string | null
  placing: string | null
  /** a signed-in user: placing a work records who did */
  canEdit: boolean
  onShow: (sid: string) => void
  onOpen: (sid: string) => void
  onPlace: (sid: string) => void
  onRemove: (sid: string) => void
  onRefresh: () => void
  /** the works whose IFC is shown on the globe (loaded when first asked: heavy) */
  ifcOn: string[]
  onToggleIfc: (sid: string) => void
}

export function GlobePanel({ projects, loading, error, placing, canEdit, onShow, onOpen, onPlace, onRemove, onRefresh, ifcOn, onToggleIfc }: GlobePanelProps) {
  const t = useT()
  const [confirming, setConfirming] = useState<string | null>(null)
  const placed = projects.filter(p => p.georef)
  const missing = projects.filter(p => !p.georef)
  return (
    <Panel title={t('globe.title')} subtitle={t('globe.subtitle', { placed: placed.length, total: projects.length })}
      loading={loading && !projects.length}
      actions={<IconButton label={t('globe.refresh')} size="sm" icon={<RefreshCw aria-hidden />} onClick={onRefresh} />}
      isEmpty={!loading && !projects.length && !error}
      empty={<EmptyState icon={<Earth aria-hidden />} title={t('globe.emptyTitle')} description={t('globe.emptyDesc')} />}>
      {error && <div className="stac-globe-list__error" role="alert">{error}</div>}
      {!canEdit && <div className="stac-globe-list__hint">{t('globe.signIn')}</div>}

      <h3 className="stac-globe-list__group"><MapPin aria-hidden /> {t('globe.placed')} <span>{placed.length}</span></h3>
      {!placed.length && <div className="stac-globe-list__hint">{t('globe.nonePlaced')}</div>}
      <ul className="stac-globe-list">
        {placed.map(p => (
          <li key={p.session_id} className={`stac-globe-list__row${placing === p.session_id ? ' is-placing' : ''}`}>
            <button type="button" className="stac-globe-list__name" onClick={() => onShow(p.session_id)} title={t('globe.show')}>
              <MapPin aria-hidden /> <span>{p.session_id}</span>
            </button>
            <div className="stac-globe-list__meta">
              {formatPlace(p.georef!)} · {t(`globe.source_${p.georef!.source}`)}
            </div>
            {confirming === p.session_id ? (
              <div className="stac-globe-list__actions">
                <span>{t('globe.removeAsk')}</span>
                <Button size="sm" variant="danger" onClick={() => { setConfirming(null); onRemove(p.session_id) }}>{t('globe.removeYes')}</Button>
                <Button size="sm" variant="ghost" onClick={() => setConfirming(null)}>{t('globe.cancel')}</Button>
              </div>
            ) : (
              <div className="stac-globe-list__actions">
                <Button size="sm" variant="secondary" icon={<FolderOpen aria-hidden />} onClick={() => onOpen(p.session_id)}>{t('globe.open')}</Button>
                {p.ifcs.length > 0 && (
                  <Button size="sm" variant={ifcOn.includes(p.session_id) ? 'primary' : 'ghost'} icon={<Building2 aria-hidden />}
                    aria-pressed={ifcOn.includes(p.session_id)} onClick={() => onToggleIfc(p.session_id)}>{t('globe.ifcOn')}</Button>
                )}
                <Button size="sm" variant="ghost" icon={<Crosshair aria-hidden />} disabled={!canEdit} onClick={() => onPlace(p.session_id)}>{t('globe.placeAgain')}</Button>
                <IconButton size="sm" label={t('globe.remove')} icon={<Trash2 aria-hidden />} disabled={!canEdit}
                  onClick={() => setConfirming(p.session_id)} />
              </div>
            )}
          </li>
        ))}
      </ul>

      <h3 className="stac-globe-list__group"><MapPinOff aria-hidden /> {t('globe.notPlaced')} <span>{missing.length}</span></h3>
      <ul className="stac-globe-list">
        {missing.map(p => (
          <li key={p.session_id} className={`stac-globe-list__row${placing === p.session_id ? ' is-placing' : ''}`}>
            <div className="stac-globe-list__name stac-globe-list__name--plain"><span>{p.session_id}</span></div>
            <div className="stac-globe-list__actions">
              <Button size="sm" variant="primary" icon={<Crosshair aria-hidden />} disabled={!canEdit} onClick={() => onPlace(p.session_id)}>{t('globe.place')}</Button>
            </div>
          </li>
        ))}
      </ul>
    </Panel>
  )
}
