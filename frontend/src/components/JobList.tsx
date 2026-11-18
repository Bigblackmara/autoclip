import { Link } from 'react-router-dom'

import { formatDuration, type Job } from '../api'
import { Badge, ChevronRightIcon, ScissorsIcon, type Tone } from '../ui'

/** "Deletes in 3 days" for a job, given the retention setting (0 = off). */
export function deletesIn(job: Job, retentionDays: number): string | null {
  if (retentionDays <= 0 || job.status === 'queued' || job.status === 'running') return null
  const created = Date.parse(job.created_at)
  if (Number.isNaN(created)) return null
  const left = Math.ceil((created + retentionDays * 86_400_000 - Date.now()) / 86_400_000)
  if (left <= 0) return 'Deleting soon'
  return `Deletes in ${left} day${left === 1 ? '' : 's'}`
}

const STATUS: Record<Job['status'], { tone: Tone; label: string }> = {
  done: { tone: 'success', label: 'Ready' },
  failed: { tone: 'danger', label: 'Failed' },
  running: { tone: 'accent', label: 'Working' },
  queued: { tone: 'neutral', label: 'Waiting' },
  cancelled: { tone: 'neutral', label: 'Cancelled' },
}

export function JobStatusBadge({ job }: { job: Job }) {
  const look = STATUS[job.status] ?? { tone: 'neutral' as Tone, label: job.status }
  const label = job.status === 'running' ? `${look.label} · ${Math.round(job.progress * 100)}%` : look.label
  return (
    <Badge tone={look.tone} dot pulse={job.status === 'running'}>
      {label}
    </Badge>
  )
}

function when(iso: string): string {
  const t = Date.parse(iso)
  if (Number.isNaN(t)) return ''
  const mins = Math.round((Date.now() - t) / 60_000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  const hours = Math.round(mins / 60)
  if (hours < 24) return `${hours} h ago`
  const days = Math.round(hours / 24)
  if (days < 30) return `${days} day${days === 1 ? '' : 's'} ago`
  return new Date(t).toLocaleDateString()
}

/**
 * Jobs as a list of rows: title, length, when, status. A finished job opens
 * its clips; anything else opens its progress page.
 */
export function JobList({ jobs, retentionDays }: { jobs: Job[]; retentionDays: number }) {
  return (
    <ul className="divide-y divide-line">
      {jobs.map((job) => {
        const deletes = deletesIn(job, retentionDays)
        return (
          <li key={job.id} className="group relative flex items-center gap-3 px-4 py-3 transition-colors hover:bg-subtle sm:px-5">
            <span className="hidden size-10 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent-text sm:grid">
              <ScissorsIcon size={18} />
            </span>
            <div className="min-w-0 flex-1">
              <Link
                to={job.status === 'done' ? `/jobs/${job.id}/clips` : `/jobs/${job.id}`}
                className="block truncate text-sm font-semibold text-fg after:absolute after:inset-0 after:content-['']"
              >
                {job.source?.title || 'Untitled'}
              </Link>
              <p className="numeric mt-0.5 truncate text-xs text-fg-muted">
                {[
                  job.source ? formatDuration(job.source.duration_s) : null,
                  when(job.created_at),
                  job.provider,
                  deletes,
                ]
                  .filter(Boolean)
                  .join(' · ')}
              </p>
            </div>
            {job.status === 'done' && (
              <Link
                to={`/jobs/${job.id}/pick`}
                className="relative z-10 hidden rounded-md px-2 py-1 text-xs font-medium text-fg-muted transition-colors hover:bg-surface hover:text-accent-text md:inline"
              >
                Pick parts
              </Link>
            )}
            <JobStatusBadge job={job} />
            <ChevronRightIcon size={16} className="text-fg-subtle" />
          </li>
        )
      })}
    </ul>
  )
}
