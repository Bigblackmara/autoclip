import { useState } from 'react'

import { errorText, PLATFORM_NAME, publishingApi, when, type Platform, type PostTarget } from '../publishing/api'
import { PostStatusBadge } from '../publishing/ui'
import { Button, Callout, Field, Input, Segmented, Select, SendIcon, Textarea, cx, useToast } from '../ui'

/** A local "YYYY-MM-DDTHH:MM" for a datetime-local input, an hour from now. */
function defaultLocalTime(): string {
  const d = new Date(Date.now() + 3_600_000)
  d.setMinutes(0, 0, 0)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

const PLATFORMS: Platform[] = ['youtube', 'instagram']

/**
 * "Post" for one exported clip: YouTube Short and/or Instagram Reel, now or
 * at a chosen time. A platform the clip is already on (or booked for) is
 * locked: AutoClip never posts a clip twice.
 */
export function PostClipPanel({ clip, onPosted }: { clip: PostTarget; onPosted?: () => void }) {
  const free = (p: Platform) => !clip.active[p]
  const [platforms, setPlatforms] = useState<Platform[]>(PLATFORMS.filter((p) => free(p)).slice(0, 1))
  const [mode, setMode] = useState<'now' | 'later'>('now')
  const [at, setAt] = useState(defaultLocalTime)
  const [privacy, setPrivacy] = useState<'public' | 'unlisted' | 'private'>('public')
  const [title, setTitle] = useState(clip.title)
  const [caption, setCaption] = useState(clip.text)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ tone: 'ok' | 'bad'; text: string } | null>(null)
  const toast = useToast()

  const toggle = (p: Platform) => setPlatforms((cur) => (cur.includes(p) ? cur.filter((x) => x !== p) : [...cur, p]))

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
      const text = mode === 'later' ? `Scheduled for ${where}, ${when(scheduled)}.` : `Posting to ${where} now.`
      setMessage({
        tone: out.warnings.length ? 'bad' : 'ok',
        text: out.warnings.length ? `${text} Before it can go out: ${out.warnings.join(' ')}` : text,
      })
      if (!out.warnings.length) toast.success(mode === 'later' ? 'Post scheduled' : 'Posting now', text)
      onPosted?.()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-5">
      <div>
        <p className="mb-2 text-sm font-medium text-fg">Post to</p>
        <div className="grid gap-2 sm:grid-cols-2">
          {PLATFORMS.map((p) => {
            const active = clip.active[p]
            const on = platforms.includes(p)
            return (
              <label
                key={p}
                className={cx(
                  'flex cursor-pointer items-start gap-3 rounded-xl border px-3.5 py-3 transition-colors',
                  active
                    ? 'cursor-not-allowed border-line bg-subtle'
                    : on
                      ? 'border-accent bg-accent-soft ring-1 ring-accent'
                      : 'border-line hover:border-line-strong hover:bg-subtle',
                )}
              >
                <input
                  type="checkbox"
                  className="mt-0.5 size-4"
                  disabled={!!active}
                  checked={on}
                  onChange={() => toggle(p)}
                />
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-fg">
                    {p === 'youtube' ? 'YouTube Short' : 'Instagram Reel'}
                  </span>
                  {active ? (
                    <span className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-fg-muted">
                      <PostStatusBadge status={active.status} />
                      {active.scheduled_at && when(active.scheduled_at)}
                    </span>
                  ) : (
                    <span className="block text-xs text-fg-muted">Not posted yet</span>
                  )}
                </span>
              </label>
            )
          })}
        </div>
        <p className="mt-1.5 text-xs text-fg-muted">A clip is never posted twice to the same place.</p>
      </div>

      <div className="flex flex-wrap items-end gap-4">
        <div>
          <p className="mb-1.5 text-sm font-medium text-fg">When</p>
          <Segmented
            label="When to post"
            options={[
              { value: 'now', label: 'Now' },
              { value: 'later', label: 'Pick a time' },
            ]}
            value={mode}
            onChange={setMode}
          />
        </div>
        {mode === 'later' && (
          <Field label="Date and time" className="w-56">
            <Input type="datetime-local" value={at} onChange={(e) => setAt(e.target.value)} />
          </Field>
        )}
        {mode === 'now' && platforms.includes('youtube') && (
          <Field label="YouTube visibility" className="w-44">
            <Select
              value={privacy}
              onChange={(e) => setPrivacy(e.target.value as typeof privacy)}
              options={[
                { value: 'public', label: 'Public' },
                { value: 'unlisted', label: 'Unlisted' },
                { value: 'private', label: 'Private' },
              ]}
            />
          </Field>
        )}
      </div>
      {mode === 'later' && (
        <p className="-mt-2 text-xs leading-relaxed text-fg-muted">
          Sent to the platform shortly before that time: YouTube holds it private until then and publishes it
          itself. Until it is sent you can cancel it on the Schedule page.
        </p>
      )}

      <Field label="Title" hint={`${title.length}/100`}>
        <Input maxLength={100} value={title} onChange={(e) => setTitle(e.target.value)} />
      </Field>
      <Field label="Caption" hint={clip.hashtags.length > 0 ? `Hashtags added: ${clip.hashtags.join(' ')}` : undefined}>
        <Textarea className="min-h-20" maxLength={2200} value={caption} onChange={(e) => setCaption(e.target.value)} />
      </Field>

      {message && (
        <Callout tone={message.tone === 'ok' ? 'success' : 'warn'} role="status" onDismiss={() => setMessage(null)}>
          {message.text}
        </Callout>
      )}

      <Button variant="primary" icon={SendIcon} loading={busy} disabled={platforms.length === 0} onClick={submit}>
        {busy ? 'Posting…' : mode === 'now' ? 'Post now' : 'Schedule post'}
      </Button>
    </div>
  )
}
