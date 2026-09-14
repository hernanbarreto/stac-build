/**
 * STAC Build — top-level React error boundary.
 * Without it, any uncaught render error unmounts the whole tree and the app
 * turns into a silent white page. This shows what happened + a reload button.
 * Lives outside the i18n provider, so it reads the module-level translator.
 */
import { Component, type ErrorInfo, type ReactNode } from 'react'
import { getT } from '../i18n'

interface Props { children: ReactNode }
interface State { error: Error | null }

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
    const t = getT()
    return (
      <div className="stac-fatal" role="alert">
        <div className="stac-fatal__title">{t('fatal.title')}</div>
        <pre className="stac-fatal__message stac-selectable">{this.state.error.message}</pre>
        <p className="stac-fatal__hint">{t('fatal.hint')}</p>
        <button type="button" className="stac-btn stac-btn--primary stac-btn--md" onClick={() => window.location.reload()}>
          <span className="stac-btn__label">{t('common.retry')}</span>
        </button>
      </div>
    )
  }
}
