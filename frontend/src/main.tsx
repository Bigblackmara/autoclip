import React from 'react'
import ReactDOM from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'

import { App } from './App'
import './index.css'
import { pageRoutes } from './routes'

// Page routes live in routes.tsx; navigation entries in nav.tsx.
const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: pageRoutes,
  },
])

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
