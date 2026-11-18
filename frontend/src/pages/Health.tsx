import { HealthPanel } from '../components/HealthPanel'
import { PageHeader } from '../ui'

/** The Health panel as a page (same checks as `autoclip doctor`). */
export function Health() {
  return (
    <div className="max-w-4xl">
      <PageHeader
        title="Health check"
        description={
          <>
            Everything AutoClip needs to work, each with a plain fix. The terminal version is{' '}
            <code className="rounded bg-subtle px-1.5 py-0.5 font-mono text-sm text-fg">autoclip doctor</code>.
          </>
        }
      />
      <HealthPanel />
    </div>
  )
}
