// STAC-Builder — mobile XR viewer entry (ui/xr.html → /app/xr.html).
// Hernán Barreto - Ingerop IN3 Session IV - STAC

import React from 'react'
import ReactDOM from 'react-dom/client'
import XRApp from './XRApp'
import './xr.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <XRApp />
  </React.StrictMode>,
)
