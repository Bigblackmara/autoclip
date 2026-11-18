import { useState } from 'react'

import {
  errorText,
  PLATFORM_NAME,
  publishingApi,
  STATUS_LABEL,
  when,
  type Platform,
  type PostableClip,
} from '../publishing/api'

/** A local "YYYY-MM-DDTHH:MM" for a datetime-local input, an hour from now. */
function defaultLocalTime(): string {
  const d = new Date(Date.now() + 3_600_000)
  d.setMinutes(0, 0, 0)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/**
 * "Post" for one exported clip: YouTube Short and/or Instagram Reel, now or
 * at a chosen time. A platform the clip is already on (or booked for) is
 * locked: AutoClip never posts a clip twice.
 */
export function PostClipPanel({
  clip,
  onPosted,
}: {
  clip: PostableClip
  onPosted?: () => void
}) {
  const free = (p: Platform) => !clip.active[p]
  const [platforms, setPlatforms] = useState<Platform[]>(
    (['youtube', 'instagram'] as Platform[]).filter((p) => free(p)).slice(0, 1),
  )
  const [mode, setMode] = useState<'now' | 'later'>('now')
  const [at, setAt] = useState(defaultLocalTime)
  const [privacy, setPrivacy] = useState<'public' | 'unlisted' | 'private'>('public')
  const [title, setTitle] = useState(clip.title)
  const [caption, setCaption] = useState(clip.text)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ tone: 'ok' | 'bad'; text: string } | null>(null)

  const toggle = (p: Platform) =>
    setPlatforms((cur) => (cur.includes(p) ? cur.filter((x) => x !== p) : [...cur, p]))

  async function submit() {
    setBusy(true)
    setMessage(null)
    try {
      const scheduled = mode === 'later' ? new Date(at).toISOString() : null
      const out = await publishingApi.createPosts({
        clip_id: clip.clip_id,
        platforms,
        scheduled_at: scheduled,
        privacy,
        title,
        caption,
      })
      const where = out.posts.map((p) => PLATFORM_NAME[p.platform]).join(' and ')
      const text =
        mode === 'later' ? `Scheduled for ${where}, ${when(scheduled)}.` : `Posting to ${where} now.`
      setMessage({
        tone: out.warnings.length ? 'bad' : 'ok',
        text: out.warnings.length ? `${text} Before it can go out: ${out.warnings.join(' ')}` : text,
      })
      onPosted?.()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-4 border-l border-ink-700 pl-4">
      <div className="flex flex-wrap gap-4">
        {(['youtube', 'instagram'] as Platform[]).map((p) => {
          const active = clip.active[p]
          return (
            <label
              key={p}
              className={`flex items-center gap-2 text-sm ${active ? 'text-ink-500' : 'text-ink-200'}`}
            >
              <input
                type="checkbox"
                disabled={!!active}
                checked={platforms.includes(p)}
                onChange={() => toggle(p)}
              />
              {p === 'youtube' ? 'YouTube Short' : 'Instagram Reel'}
              {active && (
                <span className="text-xs">
                  ({STATUS_LABEL[active.status] ?? active.status}
                  {active.scheduled_at ? `, ${when(active.scheduled_at)}` : ''})
                </span>
              )}
            </label>
          )
        })}
      </div>

      <div className="flex flex-wrap items-center gap-4 text-sm text-ink-200">
        <label className="flex items-center gap-2">
          <input type="radio" checked={mode === 'now'} onChange={() => setMode('now')} />
          Now
        </label>
        <label className="flex items-center gap-2">
          <input type="radio" checked={mode === 'later'} onChange={() => setMode('later')} />
          At
        </label>
        {mode === 'later' && (
          <input
            type="datetime-local"
            className="field max-w-56"
            value={at}
            onChange={(e) => setAt(e.target.value)}
          />
        )}
        {mode === 'now' && platforms.includes('youtube') && (
          <select
            className="field max-w-40"
            value={privacy}
            onChange={(e) => setPrivacy(e.target.value as typeof privacy)}
          >
            <option value="public">Public</option>
            <option value="unlisted">Unlisted</option>
            <option value="private">Private</option>
          </select>
        )}
      </div>
      {mode === 'later' && (
        <p className="text-xs text-ink-500">
          Sent to the platform shortly before that time: YouTube holds it private until then and
          publishes it itself. Until it is sent you can cancel it on the Schedule page.
        </p>
      )}

      <label className="block">
        <span className="eyebrow">Title</span>
        <input
          className="field"
          maxLength={100}
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
      </label>
      <label className="block">
        <span className="eyebrow">Caption</span>
        <textarea
          className="field min-h-20"
          maxLength={2200}
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
        />
      </label>
      {clip.hashtags.length > 0 && (
        <p className="text-xs text-ink-400">{clip.hashtags.join(' ')}</p>
      )}

      <div className="flex items-center gap-4">
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy || platforms.length === 0}
          onClick={submit}
        >
          {busy ? 'Posting…' : mode === 'now' ? 'Post now' : 'Schedule'}
        </button>
        {message && (
          <p
            className={`text-sm ${message.tone === 'ok' ? 'text-signal-good' : 'text-signal-bad'}`}
            role="status"
          >
            {message.text}
          </p>
        )}
      </div>
    </div>
  )
}
