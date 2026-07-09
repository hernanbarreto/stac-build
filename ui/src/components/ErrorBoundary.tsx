/**
 * STAC Build — top-level React error boundary.
 * Without it, any uncaught render error unmounts the whole tree and the app
 * turns into a silent white page. This shows what happened + a reload button.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
    children: ReactNode
}

interface State {
    error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
    state: State = { error: null }

    static getDerivedStateFromError(error: Error): State {
        return { error }
    }

    componentDidCatch(error: Error, info: ErrorInfo) {
        console.error('[ErrorBoundary] uncaught render error:', error, info.componentStack)
    }

    render() {
        if (!this.state.error) return this.props.children
        return (
            <div style={{
                position: 'fixed', inset: 0, background: '#0d1117', color: '#e6edf3',
                display: 'flex', flexDirection: 'column', alignItems: 'center',
                justifyContent: 'center', gap: 16, padding: 32, textAlign: 'center',
                fontFamily: 'system-ui, sans-serif',
            }}>
                <div style={{ fontSize: 40 }}>⚠️</div>
                <div style={{ fontSize: 18, fontWeight: 700 }}>Something went wrong</div>
                <div style={{
                    fontSize: 13, color: '#8b949e', maxWidth: 560, whiteSpace: 'pre-wrap',
                    fontFamily: 'ui-monospace, monospace',
                }}>
                    {this.state.error.message}
                </div>
                <div style={{ fontSize: 13, color: '#8b949e', maxWidth: 480 }}>
                    If this happened while loading a large point cloud, lower the
                    Detail (point budget) slider after reloading.
                </div>
                <button
                    onClick={() => window.location.reload()}
                    style={{
                        padding: '10px 24px', border: 'none', borderRadius: 8,
                        background: '#2f81f7', color: '#fff', fontSize: 14,
                        fontWeight: 600, cursor: 'pointer',
                    }}
                >↻ Reload</button>
            </div>
        )
    }
}
