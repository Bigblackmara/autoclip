/**
 * Live Watch shell page (placeholder — F1 replaces the body in Wave 2).
 *
 * The route and this file are seeded now so parallel frontend workers never
 * need to edit main.tsx (a shared file). F1 owns: dashboard layout, stream
 * grid, add-stream form; F2 owns the clips gallery (child route components
 * imported only from F1's files); F3 owns the settings section.
 */

import { ClipsPanel } from '../live/ClipsPanel'
import { LiveSettingsSection } from '../live/LiveSettingsSection'
import { StreamsPanel } from '../live/StreamsPanel'

export function LiveWatch() {
  return (
    <section className="mx-auto max-w-[1600px] px-6 pb-16 pt-8 lg:px-10">
      <header className="mb-8">
        <p className="eyebrow text-sodium-500">Live</p>
        <h1 className="font-display text-3xl text-ink-100">Live Watch</h1>
        <p className="mt-2 max-w-2xl text-ink-300">
          Watch several livestreams at once, detect viral moments as they
          happen, and cut them into caption-burned vertical clips
          automatically.
        </p>
        <p className="mt-4 max-w-3xl rounded-lg border border-sodium-500/30 bg-sodium-500/5 p-3 text-sm text-ink-300">
          <strong className="text-sodium-400">
            You are responsible for having the rights or permission to clip
            and repost each streamer's content
          </strong>
          , and for following each platform's terms of service.
        </p>
      </header>

      <div className="flex flex-col gap-12">
        <StreamsPanel />
        <ClipsPanel />
        <LiveSettingsSection />
      </div>
    </section>
  )
}
