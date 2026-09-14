/**
 * STAC Build — Entry Point
 * Hernán Barreto — Ingerop IN3
 *
 * Provider order: i18n (language) -> layout (regions, density) -> toasts ->
 * auth -> App. The density attribute is applied to <html> before the first
 * paint so the token scale is right from the start.
 */
import ReactDOM from 'react-dom/client'
import './index.css'
import { AuthProvider } from './context/AuthContext'
import { ErrorBoundary } from './components/ErrorBoundary'
import { I18nProvider } from './i18n'
import { LayoutProvider } from './layout/LayoutContext'
import { ToastProvider } from './components/ui/Toast'
import App from './App.tsx'

document.documentElement.dataset.theme = 'dark'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <ErrorBoundary>
    <I18nProvider>
      <LayoutProvider>
        <ToastProvider>
          <AuthProvider>
            <App />
          </AuthProvider>
        </ToastProvider>
      </LayoutProvider>
    </I18nProvider>
  </ErrorBoundary>,
)
