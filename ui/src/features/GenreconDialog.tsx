/**
 * GenreconDialog — the GenRecon mesh on demand (USER 2026-11-05: "GenRecon es un botón más, como el de autosegmentar
 * … el usuario va a tener que seleccionar ahí el valor del porcentaje que va a filtrar"): the viewer's confidence filter
 * for THIS session (it starts at the slider's value) — GenRecon receives the points the viewer shows at that value,
 * the keyframes vote on them and the mesh is generated over the whole of what stays.
 */
import { useEffect, useState } from 'react'
import { Box, Play } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Field } from '../components/ui/Field'
import { Stack } from '../components/ui/Panel'
import { useT } from '../i18n'

interface GenreconDialogProps {
  open: boolean
  sessionId: string | null
  scanLabel: string | null
  /** the viewer's confidence slider for this session (0..1) */
  defaultConfidence: number
  onClose: () => void
  onRun: (confidence: number) => Promise<void> | void
}

export function GenreconDialog({ open, sessionId, scanLabel, defaultConfidence, onClose, onRun }: GenreconDialogProps) {
  const t = useT()
  const [pct, setPct] = useState(0)
  const [running, setRunning] = useState(false)

  useEffect(() => {
    if (!open) return
    setPct(Math.round((defaultConfidence || 0) * 100))
    setRunning(false)
  }, [open, defaultConfidence])

  const run = async () => {
    setRunning(true)
    try { await onRun(Math.min(99, Math.max(0, pct)) / 100) } finally { setRunning(false) }
  }

  return (
    <Dialog open={open} title={t('genrecon.title')} subtitle={[sessionId, scanLabel].filter(Boolean).join(' · ') || undefined}
      icon={<Box aria-hidden />} size="md" onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>{t('common.cancel')}</Button>
          <Button variant="primary" icon={<Play aria-hidden />} disabled={running || !sessionId} onClick={run} data-autofocus>{t('genrecon.run')}</Button>
        </>
      }>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t('genrecon.description')}</p>
        <Field label={t('genrecon.confidence', { pct })} hint={t('genrecon.confidenceHint')}>
          <div className="stac-autoseg__row">
            <input type="range" min={0} max={99} step={1} value={pct} onChange={e => setPct(Number(e.target.value))} />
            <input className="stac-input__el stac-mono" type="number" min={0} max={99} step={1} value={pct} style={{ width: '5em' }}
              onChange={e => setPct(Math.min(99, Math.max(0, Math.round(Number(e.target.value) || 0))))} />
            <span className="stac-muted">%</span>
          </div>
        </Field>
      </Stack>
    </Dialog>
  )
}
