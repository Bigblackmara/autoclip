/**
 * Every page route, as children of the app shell (App.tsx). To add a page:
 * add one line here and one entry in nav.tsx (NAV_ITEMS).
 */

import { matchRoutes, type RouteObject } from 'react-router-dom'

import { AutopilotEditor } from './autopilot/AutopilotEditor'
import { AutopilotDetail, AutopilotList } from './pages/Autopilot'
import { Explore } from './pages/Explore'
import { GtaMode } from './pages/GtaMode'
import { Health } from './pages/Health'
import { Ingest } from './pages/Ingest'
import { JobProgress } from './pages/JobProgress'
import { LiveWatch } from './pages/LiveWatch'
import { NotFound } from './pages/NotFound'
import { PickParts } from './pages/PickParts'
import { Posting } from './pages/Posting'
import { Projects } from './pages/Projects'
import { Review } from './pages/Review'
import { Schedule } from './pages/Schedule'
import { Settings } from './pages/Settings'

export const pageRoutes: RouteObject[] = [
  { index: true, element: <Ingest /> },
  { path: 'projects', element: <Projects /> },
  { path: 'jobs/:jobId', element: <JobProgress /> },
  { path: 'jobs/:jobId/clips', element: <Review /> },
  { path: 'jobs/:jobId/pick', element: <PickParts /> },
  { path: 'explore', element: <Explore /> },
  { path: 'live', element: <LiveWatch /> },
  { path: 'gta', element: <GtaMode /> },
  { path: 'posting', element: <Posting /> },
  { path: 'schedule', element: <Schedule /> },
  { path: 'autopilot', element: <AutopilotList /> },
  { path: 'autopilot/new', element: <AutopilotEditor /> },
  { path: 'autopilot/:autopilotId', element: <AutopilotDetail /> },
  { path: 'autopilot/:autopilotId/edit', element: <AutopilotEditor /> },
  { path: 'settings', element: <Settings /> },
  { path: 'health', element: <Health /> },
  // Keep last: anything unknown gets a friendly "not found".
  { path: '*', element: <NotFound /> },
]

/** True when `path` (query ignored) is served by a real page, not the catch-all. */
export function isRouteRegistered(path: string): boolean {
  const matches = matchRoutes(pageRoutes, path.split('?')[0])
  const last = matches?.[matches.length - 1]
  return Boolean(last && last.route.path !== '*')
}
