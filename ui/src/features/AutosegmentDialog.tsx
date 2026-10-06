/**
 * AutosegmentDialog — the segmentation chain on demand (USER 2026-10-05):
 * the VLM prompt (scene understanding → the SAM3 vocabulary) and the SAM3
 * prompts the session has, both editable and saved in the session, then a
 * checkbox per remaining stage — VLM, SAM3 + mask projection, certification,
 * object descriptions. Run sends `run_pipeline` with `autosegment` over the
 * viewer socket: the chosen stages run on the cloud on disk, nothing is wiped.
 */
import { useEffect, useState } from 'react'
import { Play, RotateCcw, Wand2 } from 'lucide-react'
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Checkbox, Field } from '../components/ui/Field'
import { Banner } from '../components/ui/Banner'
import { Stack } from '../components/ui/Panel'
import { useT } from '../i18n'

export interface AutosegmentState {
  vlm_prompt: string
  vlm_prompt_default: string
  vlm_prompt_overridden: boolean
  sam3_prompts: string[]
  has_vlm_analysis: boolean
  has_cloud: boolean
  has_segmentation: boolean
  busy: boolean
}

export interface AutosegmentStages { vlm: boolean; sam3: boolean; certify: boolean; captions: boolean }

interface AutosegmentDialogProps {
  open: boolean
  sessionId: string | null
  state: AutosegmentState | null
  loading: boolean
  onClose: () => void
  /** save the prompts (the session keeps them), then run the chosen stages */
  onRun: (vlmPrompt: string, sam3Prompts: string[], stages: AutosegmentStages) => Promise<void> | void
}

export function AutosegmentDialog({ open, sessionId, state, loading, onClose, onRun }: AutosegmentDialogProps) {
  const t = useT()
  const [vlmPrompt, setVlmPrompt] = useState('')
  const [sam3Text, setSam3Text] = useState('')
  const [stages, setStages] = useState<AutosegmentStages>({ vlm: true, sam3: true, certify: true, captions: true })
  const [running, setRunning] = useState(false)

  useEffect(() => {
    if (!open || !state) return
    setVlmPrompt(state.vlm_prompt)
    setSam3Text(state.sam3_prompts.join('\n'))
    setStages({ vlm: true, sam3: true, certify: true, captions: true })
    setRunning(false)
  }, [open, state])

  const sam3Prompts = sam3Text.split('\n').map(s => s.trim()).filter(Boolean)
  const nothing = !stages.vlm && !stages.sam3 && !stages.certify
  const sam3NeedsPrompts = stages.sam3 && !stages.vlm && sam3Prompts.length === 0
  // USER 2026-10-06: an order never waits for the window to be free — a running pipeline on this
  // session (or any other) only means the order QUEUES behind it
  const canRun = !!state && !nothing && !sam3NeedsPrompts && !running && !loading

  const run = async () => {
    setRunning(true)
    try { await onRun(vlmPrompt, sam3Prompts, stages) } finally { setRunning(false) }
  }

  return (
    <Dialog open={open} title={t('autosegment.title')} subtitle={sessionId ?? undefined} icon={<Wand2 aria-hidden />} size="lg" onClose={onClose}
      footer={
        <>
          <Button onClick={onClose}>{t('common.cancel')}</Button>
          <Button variant="primary" icon={<Play aria-hidden />} disabled={!canRun} onClick={run} data-autofocus>{t('autosegment.run')}</Button>
        </>
      }>
      <Stack gap={3}>
        <p className="stac-dialog__message">{t('autosegment.description')}</p>
        {state && !state.has_cloud && <Banner tone="warn" compact>{t('autosegment.noCloud')}</Banner>}
        {state && state.busy && <Banner tone="info" compact>{t('autosegment.busy')}</Banner>}
        <Field label={t('autosegment.vlmPrompt')} hint={t('autosegment.vlmPromptHint')}>
          <textarea className="stac-input__el stac-autoseg__textarea" rows={10} value={vlmPrompt} spellCheck={false}
            onChange={e => setVlmPrompt(e.target.value)} disabled={loading} />
          <div className="stac-autoseg__row">
            {state?.vlm_prompt_overridden || (state && vlmPrompt.trim() !== state.vlm_prompt_default.trim())
              ? <Button size="sm" icon={<RotateCcw aria-hidden />} onClick={() => state && setVlmPrompt(state.vlm_prompt_default)}>{t('autosegment.restoreDefault')}</Button>
              : <span className="stac-muted">{t('autosegment.defaultPrompt')}</span>}
          </div>
        </Field>
        <Field label={t('autosegment.sam3Prompts', { n: sam3Prompts.length })} hint={t('autosegment.sam3PromptsHint')}>
          <textarea className="stac-input__el stac-autoseg__textarea stac-mono" rows={6} value={sam3Text} spellCheck={false}
            placeholder={t('autosegment.sam3Placeholder')} onChange={e => setSam3Text(e.target.value)} disabled={loading} />
        </Field>
        <Stack gap={1}>
          <div className="stac-section__title">{t('autosegment.stages')}</div>
          <Checkbox checked={stages.vlm} onChange={v => setStages(s => ({ ...s, vlm: v }))} label={t('autosegment.stageVlm')} description={t('autosegment.stageVlmHint')} />
          <Checkbox checked={stages.sam3} onChange={v => setStages(s => ({ ...s, sam3: v }))} label={t('autosegment.stageSam3')} description={t('autosegment.stageSam3Hint')} />
          <Checkbox checked={stages.certify} onChange={v => setStages(s => ({ ...s, certify: v }))} label={t('autosegment.stageCertify')} description={t('autosegment.stageCertifyHint')} />
          <Checkbox checked={stages.captions} disabled={!stages.certify} onChange={v => setStages(s => ({ ...s, captions: v }))} label={t('autosegment.stageCaptions')} description={t('autosegment.stageCaptionsHint')} />
        </Stack>
        {sam3NeedsPrompts && <Banner tone="warn" compact>{t('autosegment.sam3NeedsPrompts')}</Banner>}
        {nothing && <Banner tone="warn" compact>{t('autosegment.nothing')}</Banner>}
      </Stack>
    </Dialog>
  )
}
