import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import { App } from './App'
import './index.css'
import { GtaMode } from './pages/GtaMode'
import { Ingest } from './pages/Ingest'
import { JobProgress } from './pages/JobProgress'
import { LiveWatch } from './pages/LiveWatch'
import { PickParts } from './pages/PickParts'
import { Review } from './pages/Review'
import { Settings } from './pages/Settings'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Ingest /> },
      { path: 'jobs/:jobId', element: <JobProgress /> },
      { path: 'jobs/:jobId/clips', element: <Review /> },
      { path: 'jobs/:jobId/pick', element: <PickParts /> },
      { path: 'live', element: <LiveWatch /> },
      { path: 'gta', element: <GtaMode /> },
      { path: 'settings', element: <Settings /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
