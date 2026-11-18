import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import { App } from './App'
import { AutopilotEditor } from './autopilot/AutopilotEditor'
import './index.css'
import { AutopilotDetail, AutopilotList } from './pages/Autopilot'
import { Explore } from './pages/Explore'
import { GtaMode } from './pages/GtaMode'
import { Health } from './pages/Health'
import { Ingest } from './pages/Ingest'
import { JobProgress } from './pages/JobProgress'
import { LiveWatch } from './pages/LiveWatch'
import { PickParts } from './pages/PickParts'
import { Review } from './pages/Review'
import { Schedule } from './pages/Schedule'
import { Settings } from './pages/Settings'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Ingest /> },
      { path: 'jobs/:jobId', element: <JobProgress /> },
      { path: 'jobs/:jobId/clips', element: <Review /> },
      { path: 'explore', element: <Explore /> },
      { path: 'jobs/:jobId/pick', element: <PickParts /> },
      { path: 'live', element: <LiveWatch /> },
      { path: 'gta', element: <GtaMode /> },
      { path: 'settings', element: <Settings /> },
      { path: 'schedule', element: <Schedule /> },
      { path: 'autopilot', element: <AutopilotList /> },
      { path: 'autopilot/new', element: <AutopilotEditor /> },
      { path: 'autopilot/:autopilotId', element: <AutopilotDetail /> },
      { path: 'autopilot/:autopilotId/edit', element: <AutopilotEditor /> },
      { path: 'health', element: <Health /> },
    ],
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
