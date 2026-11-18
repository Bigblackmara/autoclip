import { useEffect, useMemo, useState } from 'react'

import { api, type Job, type Settings } from '../api'
import { ErrorNote } from '../components/ErrorNote'
import { JobList } from '../components/JobList'
import { ButtonLink, Card, EmptyState, FolderIcon, Input, PageHeader, PlusIcon, SearchIcon, Segmented, Skeleton } from '../ui'

type Filter = 'all' | 'done' | 'working' | 'failed'

const FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'done', label: 'Ready' },
  { value: 'working', label: 'In progress' },
  { value: 'failed', label: 'Failed' },
] as const

/** Every job, newest first: open one to review its clips or follow its progress. */
export function Projects() {
  const [jobs, setJobs] = useState<Job[] | null>(null)
  const [settings, setSettings] = useState<Settings | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')

  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    const load = async () => {
      try {
        const next = await api.listJobs(200)
        if (stopped) return
        setJobs(next)
        setError(null)
        if (next.some((job) => job.status === 'queued' || job.status === 'running')) {
          timer = window.setTimeout(load, 3000)
        }
      } catch (err) {
        if (!stopped) setError(err as Error)
      }
    }
    void load()
    api.getSettings().then(setSettings).catch(() => undefined)
    return () => {
      stopped = true
      window.clearTimeout(timer)
    }
  }, [])

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (jobs ?? []).filter((job) => {
      if (filter === 'done' && job.status !== 'done') return false
      if (filter === 'working' && job.status !== 'running' && job.status !== 'queued') return false
      if (filter === 'failed' && job.status !== 'failed' && job.status !== 'cancelled') return false
      return !q || (job.source?.title ?? '').toLowerCase().includes(q)
    })
  }, [jobs, filter, query])

  return (
    <div>
      <PageHeader
        title="Projects"
        description="Every video you've clipped. Open one to review, trim and export its clips."
        actions={
          <ButtonLink to="/" variant="primary" icon={PlusIcon}>
            New clips
          </ButtonLink>
        }
      />

      {error && (
        <div className="mb-6">
          <ErrorNote error={error} />
        </div>
      )}

      {jobs === null && !error ? (
        <div className="card divide-y divide-line" aria-label="Loading" role="status">
          {Array.from({ length: 5 }, (_, i) => (
            <div key={i} className="flex items-center gap-3 px-5 py-4">
              <Skeleton className="size-10 rounded-lg" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-3.5 w-1/2" />
                <Skeleton className="h-3 w-1/4" />
              </div>
            </div>
          ))}
        </div>
      ) : jobs && jobs.length === 0 ? (
        <EmptyState
          icon={FolderIcon}
          title="No projects yet"
          description="Paste a link or upload a video and your first project shows up here."
          action={
            <ButtonLink to="/" variant="primary" icon={PlusIcon}>
              Make your first clips
            </ButtonLink>
          }
        />
      ) : (
        jobs && (
          <Card padding="none" className="overflow-hidden">
            <div className="flex flex-col gap-3 border-b border-line p-4 sm:flex-row sm:items-center sm:justify-between sm:px-5">
              <Segmented label="Show" options={FILTERS} value={filter} onChange={setFilter} size="sm" />
              <div className="relative sm:w-72">
                <SearchIcon size={16} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-fg-subtle" />
                <Input
                  type="search"
                  aria-label="Search projects"
                  placeholder="Search by title"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </div>
            {shown.length === 0 ? (
              <p className="px-5 py-10 text-center text-sm text-fg-muted">Nothing matches. Try another filter.</p>
            ) : (
              <JobList jobs={shown} retentionDays={settings?.housekeeping?.retention_days ?? 0} />
            )}
          </Card>
        )
      )}
    </div>
  )
}
