/**
 * ConfirmDialog — thin adapter over the catalog's ConfirmDialog (ui/Dialog):
 * keeps the `useConfirmDialog()` imperative hook every feature already uses
 * (confirm / confirmDanger / alert -> Promise<boolean>). Destructive dialogs
 * focus Cancel first and use the danger button (prompt_ui.txt §6).
 */
import { useCallback, useState } from 'react'
import { ConfirmDialog as CatalogConfirmDialog } from './ui/Dialog'

export type DialogVariant = 'confirm' | 'alert' | 'danger'

interface DialogState {
  open: boolean
  title?: string
  message: string
  variant: DialogVariant
  confirmLabel?: string
  cancelLabel?: string
  resolve: ((result: boolean) => void) | null
}

export function useConfirmDialog() {
  const [state, setState] = useState<DialogState>({ open: false, message: '', variant: 'confirm', resolve: null })

  const showDialog = useCallback((opts: { message: string; title?: string; variant?: DialogVariant; confirmLabel?: string; cancelLabel?: string }): Promise<boolean> =>
    new Promise(resolve => {
      setState({ open: true, message: opts.message, title: opts.title, variant: opts.variant ?? 'confirm', confirmLabel: opts.confirmLabel, cancelLabel: opts.cancelLabel, resolve })
    }), [])

  const confirm = useCallback((message: string, title?: string) => showDialog({ message, title, variant: 'confirm' }), [showDialog])
  const confirmDanger = useCallback((message: string, title?: string) => showDialog({ message, title, variant: 'danger' }), [showDialog])
  const alert = useCallback((message: string, title?: string) => showDialog({ message, title, variant: 'alert' }), [showDialog])

  const handleConfirm = useCallback(() => { state.resolve?.(true); setState(s => ({ ...s, open: false, resolve: null })) }, [state.resolve])
  const handleCancel = useCallback(() => { state.resolve?.(false); setState(s => ({ ...s, open: false, resolve: null })) }, [state.resolve])

  const dialogElement = (
    <CatalogConfirmDialog open={state.open} title={state.title} message={state.message} variant={state.variant}
      confirmLabel={state.confirmLabel} cancelLabel={state.cancelLabel} onConfirm={handleConfirm} onCancel={handleCancel} />
  )

  return { confirm, confirmDanger, alert, dialogElement }
}
