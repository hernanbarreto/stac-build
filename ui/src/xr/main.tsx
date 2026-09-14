// STAC-Builder — mobile XR viewer entry (ui/xr.html -> /app/xr.html).
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import React from 'react'
import ReactDOM from 'react-dom/client'
import XRApp from './XRApp'
import './xr.css'

// phones/tablets: comfortable density, dark theme (tokens.css scales)
document.documentElement.dataset.density = 'comfortable'
document.documentElement.dataset.theme = 'dark'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <XRApp />
  </React.StrictMode>,
)
