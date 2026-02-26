import React, { useEffect, useRef, useCallback } from 'react'
import './ConfirmDialog.css'

export type DialogVariant = 'confirm' | 'alert' | 'danger'

interface ConfirmDialogProps {
    open: boolean
    title?: string
    message: string
    variant?: DialogVariant
    confirmLabel?: string
    cancelLabel?: string
    onConfirm: () => void
    onCancel: () => void
}

export function ConfirmDialog({
    open,
    title,
    message,
    variant = 'confirm',
    confirmLabel,
    cancelLabel = 'Cancel',
    onConfirm,
    onCancel,
}: ConfirmDialogProps) {
    const overlayRef = useRef<HTMLDivElement>(null)
    const confirmBtnRef = useRef<HTMLButtonElement>(null)

    // Default confirm label based on variant
    const resolvedConfirmLabel = confirmLabel ?? (variant === 'alert' ? 'OK' : variant === 'danger' ? 'Delete' : 'Confirm')

    // Auto-focus confirm button
    useEffect(() => {
        if (open) confirmBtnRef.current?.focus()
    }, [open])

    // Close on Escape
    useEffect(() => {
        if (!open) return
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onCancel()
            if (e.key === 'Enter') onConfirm()
        }
        window.addEventListener('keydown', handler)
        return () => window.removeEventListener('keydown', handler)
    }, [open, onCancel, onConfirm])

    // Close on backdrop click
    const handleBackdropClick = useCallback(
        (e: React.MouseEvent) => {
            if (e.target === overlayRef.current) onCancel()
        },
        [onCancel]
    )

    if (!open) return null

    const icon = variant === 'danger'
        ? '⚠️'
        : variant === 'alert'
            ? 'ℹ️'
            : '❓'

    return (
        <div className="cd-overlay" ref={overlayRef} onClick={handleBackdropClick}>
            <div className={`cd-dialog cd-${variant}`}>
                <div className="cd-header">
                    <img src="/logo.png" alt="STAC" className="cd-logo" />
                    <span className="cd-app-name">STAC Build</span>
                </div>

                <div className="cd-body">
                    <span className="cd-icon">{icon}</span>
                    <div className="cd-content">
                        {title && <div className="cd-title">{title}</div>}
                        <div className="cd-message">{message}</div>
                    </div>
                </div>

                <div className="cd-actions">
                    {variant !== 'alert' && (
                        <button className="cd-btn cd-btn-cancel" onClick={onCancel}>
                            {cancelLabel}
                        </button>
                    )}
                    <button
                        className={`cd-btn cd-btn-${variant === 'danger' ? 'danger' : 'confirm'}`}
                        onClick={onConfirm}
                        ref={confirmBtnRef}
                    >
                        {resolvedConfirmLabel}
                    </button>
                </div>
            </div>
        </div>
    )
}

/* ── Hook for imperative usage ── */
import { useState } from 'react'

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
    const [state, setState] = useState<DialogState>({
        open: false,
        message: '',
        variant: 'confirm',
        resolve: null,
    })

    const showDialog = useCallback(
        (opts: {
            message: string
            title?: string
            variant?: DialogVariant
            confirmLabel?: string
            cancelLabel?: string
        }): Promise<boolean> => {
            return new Promise((resolve) => {
                setState({
                    open: true,
                    message: opts.message,
                    title: opts.title,
                    variant: opts.variant ?? 'confirm',
                    confirmLabel: opts.confirmLabel,
                    cancelLabel: opts.cancelLabel,
                    resolve,
                })
            })
        },
        []
    )

    const confirm = useCallback(
        (message: string, title?: string) =>
            showDialog({ message, title, variant: 'confirm' }),
        [showDialog]
    )

    const confirmDanger = useCallback(
        (message: string, title?: string) =>
            showDialog({ message, title, variant: 'danger' }),
        [showDialog]
    )

    const alert = useCallback(
        (message: string, title?: string) =>
            showDialog({ message, title, variant: 'alert' }),
        [showDialog]
    )

    const handleConfirm = useCallback(() => {
        state.resolve?.(true)
        setState((s) => ({ ...s, open: false, resolve: null }))
    }, [state.resolve])

    const handleCancel = useCallback(() => {
        state.resolve?.(false)
        setState((s) => ({ ...s, open: false, resolve: null }))
    }, [state.resolve])

    const dialogElement = (
        <ConfirmDialog
            open={state.open}
            title={state.title}
            message={state.message}
            variant={state.variant}
            confirmLabel={state.confirmLabel}
            cancelLabel={state.cancelLabel}
            onConfirm={handleConfirm}
            onCancel={handleCancel}
        />
    )

    return { confirm, confirmDanger, alert, dialogElement }
}
