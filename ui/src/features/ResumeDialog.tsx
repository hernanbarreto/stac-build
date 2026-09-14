/**
 * ResumeDialog — incomplete segmentation propagation found on opening the
 * manager: Resume (re-seed + batched propagate) or Cancel (delete the
 * incomplete instances). Stays open with live progress until the task ends.
 */
import { AlertTriangle, Play, Trash2 } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Progress } from '../components/ui/Progress'
import { Stack } from '../components/ui/Panel'
import { useFmt, useT } from '../i18n'

interface ResumeDialogProps {
  incomplete: Array<{ instance_id: number; label: string; frames: number }>
  complete: number
  busy: boolean
  progress: { pct: number; detail: string; elapsed_s: number } | null
  onResume: () => void
  onCancelDelete: () => void
}

export function ResumeDialog({ incomplete, complete, busy, progress, onResume, onCancelDelete }: ResumeDialogProps) {
  const t = useT()
  const fmt = useFmt()
  return (
    <Dialog open title={t('resume.title')} tone="warn" icon={<AlertTriangle aria-hidden />} busy={busy}
      footer={
        <>
          <Button variant="danger" icon={<Trash2 aria-hidden />} disabled={busy} onClick={onCancelDelete}>{t('resume.cancelDelete')}</Button>
          <Button variant="primary" icon={<Play aria-hidden />} loading={busy} onClick={onResume} data-autofocus>{busy ? t('resume.resuming') : t('resume.resume')}</Button>
        </>
      }>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t.plural('resume.summary', incomplete.length, { complete: fmt.integer(complete) })}</p>
        <ul className="stac-resume__list">
          {incomplete.map(i => <li key={i.instance_id}>{i.label} <span className="stac-mono stac-resume__frames">{t.plural('sessions.frames', i.frames)}</span></li>)}
        </ul>
        <p className="stac-dialog__message">{t('resume.explain')}</p>
        {busy && (
          <Progress value={progress?.pct ?? 0} tone="brand" size="md" label={progress?.detail || t('resume.starting')} />
        )}
        {busy && progress && <p className="stac-dialog__message">{t('resume.elapsed', { min: fmt.integer(Math.round(progress.elapsed_s / 60)) })}</p>}
      </Stack>
    </Dialog>
  )
}
