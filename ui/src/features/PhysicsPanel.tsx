/**
 * PhysicsPanel — the gravity sandbox (USER 2026-10-01): drop spheres that bounce on the point cloud (the floor is the cloud's own, steps included)
 * and on every visible mesh (library objects such as the reference human, ShapeR objects, Mesh
 * results); with no surface under it a sphere falls and is gone. Dropped 1.5 m above the view centre.
 */
import { Circle, Trash2, X } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Slider } from '../components/ui/Field'
import { useT } from '../i18n'

export function PhysicsPanel(p: {
  diameterCm: number; onDiameterCm: (v: number) => void
  bounce: number; onBounce: (v: number) => void
  onDrop: () => void; onClear: () => void; onClose: () => void
}) {
  const t = useT()
  return (
    <div className="stac-physics" role="region" aria-label={t('physics.title')}>
      <div className="stac-physics__head">
        <strong>{t('physics.title')}</strong>
        <Button variant="ghost" size="sm" aria-label={t('common.close')} icon={<X aria-hidden />} onClick={p.onClose} />
      </div>
      <p className="stac-physics__hint">{t('physics.hint')}</p>
      <Slider label={t('physics.diameter')} value={p.diameterCm} onChange={p.onDiameterCm} min={2} max={100} step={1} unit="cm" />
      <Slider label={t('physics.bounce')} value={p.bounce} onChange={p.onBounce} min={0} max={1} step={0.05} digits={2} />
      <div className="stac-physics__actions">
        <Button variant="primary" size="sm" icon={<Circle aria-hidden />} onClick={p.onDrop}>{t('physics.drop')}</Button>
        <Button variant="secondary" size="sm" icon={<Trash2 aria-hidden />} onClick={p.onClear}>{t('physics.clear')}</Button>
      </div>
    </div>
  )
}
