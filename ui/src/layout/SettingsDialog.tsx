/**
 * SettingsDialog — density (compact / comfortable) and language (es-MX / en),
 * both persisted (§4.5, §8). Opened from File -> Settings, the activity bar
 * gear and the command palette.
 */
import { Dialog } from '../components/ui/Dialog'
import { Button } from '../components/ui/Button'
import { Field, SegmentedControl } from '../components/ui/Field'
import { Stack } from '../components/ui/Panel'
import { useLayout, type Density } from './LayoutContext'
import { useI18n, type Lang } from '../i18n'

export function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { state, setDensity } = useLayout()
  const { lang, setLang, t } = useI18n()
  return (
    <Dialog open={open} title={t('settings.title')} onClose={onClose} size="sm"
      footer={<Button variant="primary" onClick={onClose} data-autofocus>{t('common.close')}</Button>}>
      <Stack gap={4}>
        <Field label={t('settings.density')} hint={t('settings.densityHint')}>
          <SegmentedControl<Density> ariaLabel={t('settings.density')} value={state.density} onChange={setDensity}
            options={[{ value: 'compact', label: t('settings.compact') }, { value: 'comfortable', label: t('settings.comfortable') }]} />
        </Field>
        <Field label={t('settings.language')} hint={t('settings.languageHint')}>
          <SegmentedControl<Lang> ariaLabel={t('settings.language')} value={lang} onChange={setLang}
            options={[{ value: 'es', label: t('settings.spanish') }, { value: 'en', label: t('settings.english') }]} />
        </Field>
      </Stack>
    </Dialog>
  )
}
