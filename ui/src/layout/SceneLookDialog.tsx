/**
 * SceneLookDialog — View, Background and grid (USER 2026-11-09): the background (Default / Horizon / Plain
 * with its colour) and the grid (cell size, a major line every N cells, its colour, free whatever the
 * background). The choice is the session's, saved in the user's preferences file of that session.
 */
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Field, SegmentedControl, Select } from '../components/ui/Field'
import { Stack } from '../components/ui/Panel'
import { useFmt, useI18n } from '../i18n'
import { tokenColor, VP } from '../components/viewport/palette'
import { DEFAULT_LOOK, GRID_CELLS_M, GRID_MAJOR_EVERY, gridLength, type BackgroundKind, type SceneLook } from '../components/viewport/sceneLook'
import './SceneLookDialog.css'

export function SceneLookDialog({ open, onClose, look, onChange }: {
  open: boolean
  onClose: () => void
  look: SceneLook
  onChange: (look: SceneLook) => void
}) {
  const { t } = useI18n()
  const fmt = useFmt()
  const bg = look.background, grid = look.grid
  const setBg = (patch: Partial<SceneLook['background']>) => onChange({ ...look, background: { ...bg, ...patch } })
  const setGrid = (patch: Partial<SceneLook['grid']>) => onChange({ ...look, grid: { ...grid, ...patch } })
  return (
    <Dialog open={open} title={t('sceneLook.title')} onClose={onClose} size="sm"
      footer={<>
        <Button variant="ghost" onClick={() => onChange(DEFAULT_LOOK)}>{t('sceneLook.reset')}</Button>
        <Button variant="primary" onClick={onClose} data-autofocus>{t('common.close')}</Button>
      </>}>
      <Stack gap={4}>
        <Field label={t('sceneLook.background')} hint={t('sceneLook.bgHint')}>
          <SegmentedControl<BackgroundKind> ariaLabel={t('sceneLook.background')} value={bg.kind}
            onChange={kind => setBg({ kind })}
            options={[{ value: 'default', label: t('sceneLook.bgDefault') },
                      { value: 'horizon', label: t('sceneLook.bgHorizon') },
                      { value: 'plain', label: t('sceneLook.bgPlain') }]} />
        </Field>
        {bg.kind === 'plain' && (
          <Field label={t('sceneLook.bgColor')}>
            <input type="color" className="stac-scene-look__color" aria-label={t('sceneLook.bgColor')}
              value={bg.color ?? tokenColor(VP.clear)} onChange={e => setBg({ color: e.target.value })} />
          </Field>
        )}
        <Field label={t('sceneLook.cell')}>
          <Select<string> aria-label={t('sceneLook.cell')} value={String(grid.cellM)}
            onChange={v => setGrid({ cellM: Number(v) })}
            options={GRID_CELLS_M.map(c => ({ value: String(c), label: gridLength(c, fmt.number) }))} />
        </Field>
        <Field label={t('sceneLook.major')} hint={t('sceneLook.majorHint')}>
          <Select<string> aria-label={t('sceneLook.major')} value={String(grid.majorEvery)}
            onChange={v => setGrid({ majorEvery: Number(v) })}
            options={GRID_MAJOR_EVERY.map(n => ({
              value: String(n), label: t('sceneLook.majorCells', { n: String(n), length: gridLength(n * grid.cellM, fmt.number) }),
            }))} />
        </Field>
        <Field label={t('sceneLook.gridColor')} hint={t('sceneLook.gridColorHint')}>
          <input type="color" className="stac-scene-look__color" aria-label={t('sceneLook.gridColor')}
            value={grid.color ?? tokenColor(VP.gridMajor)} onChange={e => setGrid({ color: e.target.value })} />
        </Field>
        <span className="stac-scene-look__note">{t('sceneLook.saved')}</span>
      </Stack>
    </Dialog>
  )
}
