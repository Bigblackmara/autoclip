import { HealthPanel } from '../components/HealthPanel'

/** The Health panel as a page (same checks as `autoclip doctor`). */
export function Health() {
  return (
    <div className="rise max-w-4xl space-y-8 pt-10">
      <header>
        <p className="eyebrow">System</p>
        <h1 className="font-display text-5xl text-ink-100">Health</h1>
        <p className="mt-2 text-sm text-ink-400">
          Everything AutoClip needs, each with a plain fix. The terminal version is{' '}
          <code className="text-ink-200">autoclip doctor</code>.
        </p>
      </header>
      <HealthPanel />
    </div>
  )
}
