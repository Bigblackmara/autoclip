/**
 * GTA Mode → Posting: connect YouTube (shared with Live Watch) and Instagram
 * (through Zernio), set the posting schedule, and watch what goes out when.
 *
 * Each slot gets, per enabled platform, the best-scored QA-passed clip not yet
 * posted there (normally the same clip on both). YouTube: uploaded as private
 * shortly before the slot with a scheduled publish time, so YouTube makes it
 * public exactly at the slot. Instagram: scheduled at Zernio for the same time.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { ApiError } from '../api'
import { liveApi, type LiveAttestation, type LivePlatformStatus } from '../live/api'
import {
  MAX_UPLOAD_ATTEMPTS,
  PLATFORMS,
  PLATFORM_NAME,
  PLATFORM_SHORT,
  gtaApi,
  type GtaClip,
  slotLabel,
  type GtaPosting,
  type InstagramStatus,
  type PlanEntry,
  type Platform,
  type PostingOverview,
} from './api'

const POLL_MS = 4000
const CONSOLE_URL = 'https://console.cloud.google.com/apis/credentials'
const ZERNIO_URL = 'https://zernio.com'

function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.hint ? `${err.message}\n${err.hint}` : err.message
  return err instanceof Error ? err.message : String(err)
}

const STATUS_TEXT: Record<PlanEntry['status'], string> = {
  open: 'Open — no finished clip yet',
  missed: 'Missed',
  planned: 'Next in line',
  uploading: 'Uploading…',
  scheduled: 'Scheduled (goes public at this time)',
  checking: 'Checking with Zernio…',
  posted: 'Live on Instagram',
  failed: 'Failed',
  interrupted: 'Interrupted',
  skipped: 'Skipped',
}

const STATUS_TONE: Record<PlanEntry['status'], string> = {
  open: 'text-ink-500',
  missed: 'text-ink-500',
  planned: 'text-ink-300',
  uploading: 'text-sodium-400',
  scheduled: 'text-signal-good',
  checking: 'text-sodium-400',
  posted: 'text-signal-good',
  failed: 'text-signal-bad',
  interrupted: 'text-signal-bad',
  skipped: 'text-ink-500',
}

/** Status wording for one platform (YouTube and Instagram/Zernio differ). */
function statusText(platform: Platform, status: PlanEntry['status']): string {
  if (platform === 'youtube') {
    if (status === 'scheduled') return 'On YouTube (goes public at this time)'
    if (status === 'interrupted') return 'Interrupted — check YouTube Studio'
  } else {
    if (status === 'scheduled') return 'Scheduled at Zernio (goes live at this time)'
    if (status === 'uploading') return 'Sending to Zernio…'
    if (status === 'interrupted') return 'Interrupted — check Zernio / Instagram'
  }
  return STATUS_TEXT[status]
}

function PlatformTag({ platform }: { platform: Platform }) {
  return (
    <span
      title={PLATFORM_NAME[platform]}
      className="border border-ink-700 px-1 py-px font-mono text-[10px] tracking-wide text-ink-300"
    >
      {PLATFORM_SHORT[platform]}
    </span>
  )
}

export function YouTubeTab() {
  const [params, setParams] = useSearchParams()
  const [status, setStatus] = useState<LivePlatformStatus | null>(null)
  const [attestation, setAttestation] = useState<LiveAttestation | null>(null)
  const [instagram, setInstagram] = useState<InstagramStatus | null>(null)
  const [overview, setOverview] = useState<PostingOverview | null>(null)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')

  const loadConnection = useCallback(() => {
    liveApi
      .authStatus('youtube')
      .then(setStatus)
      .catch((err) => setError(errorText(err)))
    liveApi
      .getAttestation()
      .then(setAttestation)
      .catch((err) => setError(errorText(err)))
    gtaApi
      .instagram()
      .then(setInstagram)
      .catch((err) => setError(errorText(err)))
  }, [])

  const loadOverview = useCallback(() => {
    gtaApi
      .posting()
      .then((value) => {
        setOverview(value)
        setError('')
      })
      .catch((err) => setError(errorText(err)))
  }, [])

  useEffect(() => {
    loadConnection()
    loadOverview()
    const id = window.setInterval(loadOverview, POLL_MS)
    return () => window.clearInterval(id)
  }, [loadConnection, loadOverview])

  // Back from Google's consent page (/gta?connected=1).
  useEffect(() => {
    if (params.get('connected') !== '1') return
    setNote('YouTube connected.')
    void gtaApi.runPosting().catch(() => undefined)
    setParams(
      (p) => {
        const next = new URLSearchParams(p)
        next.delete('connected')
        next.set('tab', 'youtube')
        return next
      },
      { replace: true },
    )
  }, [params, setParams])

  const refreshAll = () => {
    loadConnection()
    loadOverview()
    void gtaApi.runPosting().catch(() => undefined)
  }

  return (
    <div className="grid max-w-4xl gap-6">
      {error && <p className="whitespace-pre-line text-sm text-signal-bad">{error}</p>}
      {note && <p className="text-sm text-signal-good">{note}</p>}
      <ConnectSection status={status} attestation={attestation} onChange={refreshAll} />
      <InstagramSection
        info={instagram}
        attestation={attestation}
        account={overview?.posting.instagram_account ?? ''}
        onInfo={setInstagram}
        onChange={refreshAll}
      />
      {overview && <ScheduleSection overview={overview} onChange={refreshAll} />}
      {overview && <ScheduledSection overview={overview} />}
      <ReadySection onChange={loadOverview} />
      {overview && <PlanSection overview={overview} />}
      {overview && <HistorySection overview={overview} onChange={loadOverview} />}
    </div>
  )
}

// --------------------------------------------------------------------------
// Ready to post: every finished clip (VOD or live) as soon as it passes QA
// --------------------------------------------------------------------------

const ON_YOUTUBE = new Set(['uploading', 'scheduled', 'posted', 'checking'])

/** A Date as the value a datetime-local input wants (local time, minutes). */
function localInputValue(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function untilText(iso: string | null): string {
  if (!iso) return ''
  const ms = new Date(iso).getTime() - Date.now()
  if (Number.isNaN(ms)) return ''
  if (ms <= 0) return 'going public now'
  const m = Math.round(ms / 60000)
  return m < 60 ? `in ${m} min` : `in ${Math.floor(m / 60)} h ${m % 60} min`
}

function whenText(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : ''
}

/** Every post waiting to go public, soonest first, with its exact time. */
function ScheduledSection({ overview }: { overview: PostingOverview }) {
  const [, tick] = useState(0)
  useEffect(() => {
    const id = window.setInterval(() => tick((n) => n + 1), 30_000)
    return () => window.clearInterval(id)
  }, [])
  const waiting = overview.uploads
    .filter((u) => u.status === 'scheduled' || u.status === 'uploading')
    .sort((a, b) => (a.slot_at ?? '').localeCompare(b.slot_at ?? ''))
  const recent = overview.uploads
    .filter((u) => u.status === 'posted')
    .sort((a, b) => (b.slot_at ?? '').localeCompare(a.slot_at ?? ''))
    .slice(0, 5)
  return (
    <section className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <h2 className="text-base font-semibold text-fg">Scheduled ({waiting.length})</h2>
      {waiting.length === 0 ? (
        <p className="mt-2 text-sm text-fg-muted">Nothing scheduled.</p>
      ) : (
        <ul className="mt-2 divide-y divide-line">
          {waiting.map((u) => (
            <li key={u.id} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 py-3 text-sm">
              <span className="w-44 font-medium text-fg">{whenText(u.slot_at)}</span>
              <span className="text-xs font-semibold text-accent">{untilText(u.slot_at)}</span>
              <PlatformTag platform={u.platform} />
              <span className="min-w-0 flex-1 truncate text-fg">{u.title || u.clip_title}</span>
              {u.url && (
                <a href={u.url} target="_blank" rel="noreferrer" className="text-xs text-accent underline">
                  Open ↗
                </a>
              )}
            </li>
          ))}
        </ul>
      )}
      {recent.length > 0 && (
        <>
          <h3 className="mt-4 text-sm font-semibold text-fg">Posted</h3>
          <ul className="mt-1 divide-y divide-line">
            {recent.map((u) => (
              <li key={u.id} className="flex flex-wrap items-baseline gap-x-4 py-2 text-sm">
                <span className="w-44 text-fg-muted">{whenText(u.slot_at)}</span>
                <PlatformTag platform={u.platform} />
                <span className="min-w-0 flex-1 truncate text-fg">{u.title || u.clip_title}</span>
                {u.url && (
                  <a href={u.url} target="_blank" rel="noreferrer" className="text-xs text-accent underline">
                    Open ↗
                  </a>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}

function ReadyRow({ clip, onPosted }: { clip: GtaClip; onPosted: () => void }) {
  const [title, setTitle] = useState(clip.hook_title || '')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [failed, setFailed] = useState(false)
  const [when, setWhen] = useState(() => localInputValue(new Date(Date.now() + 60 * 60 * 1000)))
  const seconds = Math.round(clip.end_s - clip.start_s)
  const post = async (schedule: boolean) => {
    setBusy(true)
    setMessage(schedule ? 'Uploading and scheduling…' : 'Uploading to YouTube…')
    setFailed(false)
    try {
      const at = schedule ? new Date(when).toISOString() : undefined
      const result = await gtaApi.postNow(clip.id, title.trim(), at)
      const good = schedule ? result.outcome === 'scheduled' : result.outcome === 'posted'
      setFailed(!good)
      setMessage(
        good && !schedule && result.upload?.url ? `Live: ${result.upload.url}` : result.message,
      )
      onPosted()
    } catch (err) {
      setFailed(true)
      setMessage(errorText(err))
    } finally {
      setBusy(false)
    }
  }
  return (
    <li className="flex flex-wrap items-start gap-4 py-4">
      <video
        src={`/api/gta/clips/${encodeURIComponent(clip.id)}/video`}
        className="h-40 w-[5.6rem] shrink-0 rounded-lg bg-black object-cover"
        controls
        preload="metadata"
      />
      <div className="min-w-[14rem] flex-1 space-y-2">
        <label className="block text-xs font-medium text-fg-muted" htmlFor={`title-${clip.id}`}>
          YouTube title
        </label>
        <input
          id={`title-${clip.id}`}
          value={title}
          maxLength={100}
          onChange={(e) => setTitle(e.target.value)}
          className="w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-fg"
        />
        <p className="text-xs text-fg-muted">
          {seconds} s · {clip.moment_type || 'moment'}
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={busy || !title.trim()}
            onClick={() => void post(false)}
            className="rounded-lg bg-accent px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50"
          >
            {busy ? 'Working…' : 'Post now (public)'}
          </button>
          <span className="text-xs text-fg-muted">or</span>
          <input
            type="datetime-local"
            value={when}
            onChange={(e) => setWhen(e.target.value)}
            className="rounded-lg border border-line bg-surface px-2 py-1 text-sm text-fg"
            aria-label="Schedule time"
          />
          <button
            type="button"
            disabled={busy || !title.trim() || !when}
            onClick={() => void post(true)}
            className="rounded-lg border border-line px-3 py-1.5 text-sm font-semibold text-fg disabled:opacity-50"
          >
            Schedule
          </button>
          {message && (
            <span className={`text-xs ${failed ? 'text-danger' : 'text-success'} break-all`}>{message}</span>
          )}
        </div>
      </div>
    </li>
  )
}

function ReadySection({ onChange }: { onChange: () => void }) {
  const [clips, setClips] = useState<GtaClip[]>([])
  const load = useCallback(() => {
    gtaApi
      .clips()
      .then((all) =>
        setClips(
          all.filter(
            (c) =>
              c.status === 'ready' &&
              c.has_video &&
              c.qa?.passed === true &&
              !ON_YOUTUBE.has(c.uploads?.youtube?.status ?? ''),
          ),
        ),
      )
      .catch(() => undefined)
  }, [])
  useEffect(() => {
    load()
    const id = window.setInterval(load, POLL_MS)
    return () => window.clearInterval(id)
  }, [load])
  return (
    <section className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
      <h2 className="text-base font-semibold text-fg">Ready to post ({clips.length})</h2>
      <p className="mt-1 text-sm text-fg-muted">
        Every clip appears here the moment it passes QA, live clips included. Edit the title and post it now, or
        leave it for the schedule.
      </p>
      {clips.length === 0 ? (
        <p className="mt-4 text-sm text-fg-muted">No finished clips waiting.</p>
      ) : (
        <ul className="mt-2 divide-y divide-line">
          {clips.map((c) => (
            <ReadyRow
              key={c.id}
              clip={c}
              onPosted={() => {
                load()
                onChange()
              }}
            />
          ))}
        </ul>
      )}
    </section>
  )
}

// --------------------------------------------------------------------------
// Connection (rights notice → OAuth app → Connect)
// --------------------------------------------------------------------------

function Step({
  n,
  done,
  title,
  children,
}: {
  n: number
  done: boolean
  title: string
  children: React.ReactNode
}) {
  return (
    <li className="border-l-2 border-ink-800 pl-4">
      <p className="text-sm font-semibold text-ink-100">
        <span className={done ? 'text-signal-good' : 'text-sodium-500'}>{done ? '✓' : n}</span> {title}
      </p>
      <div className="mt-2">{children}</div>
    </li>
  )
}

/** Step 1 of both connections: the shared Live Watch rights notice. */
function RightsStep({
  accepted,
  busy,
  platform,
  onAccept,
}: {
  accepted: boolean
  busy: boolean
  platform: string
  onAccept: () => void
}) {
  const [agreed, setAgreed] = useState(false)
  return (
    <Step n={1} done={accepted} title="Rights notice">
      {accepted ? (
        <p className="text-xs text-ink-500">Accepted.</p>
      ) : (
        <>
          <label className="flex items-start gap-3 text-sm text-ink-200">
            <input
              type="checkbox"
              checked={agreed}
              onChange={(e) => setAgreed(e.target.checked)}
              className="mt-0.5 size-4 accent-sodium-500"
            />
            I have the rights or permission to clip and repost the streams I add, and I follow {platform}'s
            terms.
          </label>
          <button className="btn btn-primary mt-3" disabled={!agreed || busy} onClick={onAccept}>
            Accept and unlock uploads
          </button>
        </>
      )}
    </Step>
  )
}

function ConnectSection({
  status,
  attestation,
  onChange,
}: {
  status: LivePlatformStatus | null
  attestation: LiveAttestation | null
  onChange: () => void
}) {
  const [clientId, setClientId] = useState('')
  const [secret, setSecret] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [copied, setCopied] = useState(false)
  const [changing, setChanging] = useState(false)
  const [notice, setNotice] = useState('')
  const redirectRef = useRef<HTMLElement>(null)

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true)
    setMessage('')
    try {
      await fn()
      onChange()
    } catch (err) {
      setMessage(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  const accepted = attestation?.accepted === true
  const hasCredentials = status?.has_credentials === true
  const connected = status?.connected === true

  const connect = () =>
    act(async () => {
      const { redirect_url: target } = await gtaApi.connectYoutube()
      if (!target) throw new Error('The server returned no consent URL.')
      window.location.assign(target)
    })

  const copyRedirect = async () => {
    if (!status?.redirect_uri) return
    try {
      await navigator.clipboard.writeText(status.redirect_uri)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      const element = redirectRef.current
      if (element) window.getSelection()?.selectAllChildren(element)
    }
  }

  return (
    <section className="card min-w-0 p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-fg">YouTube channel</h2>
        <span className={`text-xs ${connected ? 'text-signal-good' : 'text-ink-500'}`}>
          {status === null
            ? 'checking…'
            : connected
              ? `connected${status.account ? ` · ${status.account}` : ''}`
              : 'not connected'}
        </span>
      </div>
      <p className="mt-2 max-w-prose text-xs text-ink-500">
        The same connection Live Watch uses. Upload permission only: AutoClip can post to the channel but
        can't read or change anything else on it.
      </p>
      <ol className="mt-5 grid gap-6">
        <RightsStep
          accepted={accepted}
          busy={busy}
          platform="YouTube"
          onAccept={() => void act(() => liveApi.acceptAttestation())}
        />

        <Step n={2} done={hasCredentials} title="Your Google OAuth app">
          {notice && <p className="mb-2 text-xs text-sodium-400">{notice}</p>}
          {hasCredentials && !changing ? (
            <div className="flex flex-wrap items-center gap-3">
              <p className="text-xs text-ink-500">Client ID and secret are saved in your OS keyring.</p>
              <button
                type="button"
                className="btn btn-quiet"
                disabled={busy}
                onClick={() => {
                  setNotice('')
                  setChanging(true)
                }}
              >
                Change
              </button>
            </div>
          ) : (
            <form
              className="grid gap-3"
              onSubmit={(e) => {
                e.preventDefault()
                if (!clientId.trim() || !secret.trim()) return
                void act(async () => {
                  const saved = await liveApi.saveCredentials('youtube', clientId.trim(), secret.trim())
                  setClientId('')
                  setSecret('')
                  setChanging(false)
                  setNotice(
                    saved.disconnected
                      ? 'New OAuth app saved. The old connection was removed — press Connect YouTube below.'
                      : 'Saved.',
                  )
                })
              }}
            >
              <p className="max-w-prose text-xs leading-relaxed text-ink-400">
                In the{' '}
                <a href={CONSOLE_URL} target="_blank" rel="noreferrer" className="text-ink-200 underline">
                  Google Cloud Console ↗
                </a>
                : enable <strong>YouTube Data API v3</strong>, set up the OAuth consent screen (add your
                Google account as a test user), then create an OAuth client ID of type{' '}
                <strong>Web application</strong> with this exact redirect URI:
              </p>
              {status?.redirect_uri && (
                <div className="flex flex-wrap items-center gap-2">
                  <code ref={redirectRef} className="min-w-0 rounded-md bg-ink-850 px-2 py-1 font-mono text-xs break-all text-ink-200">
                    {status.redirect_uri}
                  </code>
                  <button type="button" className="btn btn-quiet" onClick={copyRedirect}>
                    {copied ? 'Copied ✓' : 'Copy'}
                  </button>
                </div>
              )}
              <label className="text-xs text-ink-400">
                Client ID
                <input className="field" value={clientId} onChange={(e) => setClientId(e.target.value)} />
              </label>
              <label className="text-xs text-ink-400">
                Client secret
                <input
                  className="field"
                  type="password"
                  autoComplete="off"
                  value={secret}
                  onChange={(e) => setSecret(e.target.value)}
                />
              </label>
              <div className="flex flex-wrap gap-3">
                <button
                  className="btn btn-primary"
                  disabled={busy || !clientId.trim() || !secret.trim()}
                >
                  {hasCredentials ? 'Replace in keyring' : 'Save to keyring'}
                </button>
                {changing && (
                  <button
                    type="button"
                    className="btn btn-quiet"
                    onClick={() => {
                      setChanging(false)
                      setClientId('')
                      setSecret('')
                    }}
                  >
                    Cancel
                  </button>
                )}
              </div>
            </form>
          )}
        </Step>

        <Step n={3} done={connected} title="Connect your channel">
          <p className="mb-3 text-xs text-ink-500">
            Opens Google's consent page in this tab; you sign in there yourself and come back here.
          </p>
          <button
            className={connected ? 'btn btn-ghost' : 'btn btn-primary'}
            disabled={busy || !hasCredentials}
            onClick={connect}
          >
            {connected ? 'Reconnect' : 'Connect YouTube'}
          </button>
        </Step>
      </ol>
      {message && <p className="mt-4 whitespace-pre-line text-sm text-signal-bad">{message}</p>}
    </section>
  )
}

// --------------------------------------------------------------------------
// Instagram via Zernio (rights notice → API key → test)
// --------------------------------------------------------------------------

function InstagramSection({
  info,
  attestation,
  account,
  onInfo,
  onChange,
}: {
  info: InstagramStatus | null
  attestation: LiveAttestation | null
  /** Saved schedule's Instagram account (may be empty). */
  account: string
  onInfo: (value: InstagramStatus) => void
  onChange: () => void
}) {
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)

  const act = async (fn: () => Promise<InstagramStatus | void>, done?: string) => {
    setBusy(true)
    setMessage(null)
    try {
      const value = await fn()
      if (value) onInfo(value)
      if (done) setMessage({ tone: 'good', text: done })
      onChange()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  const test = async () => {
    setBusy(true)
    setMessage(null)
    try {
      const result = await gtaApi.testInstagram(account)
      setMessage({ tone: result.ok ? 'good' : 'bad', text: result.message })
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    } finally {
      setBusy(false)
    }
  }

  const accepted = attestation?.accepted === true
  const keySaved = info?.key_saved === true

  return (
    <section className="card min-w-0 p-5 sm:p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-fg">Instagram (via Zernio)</h2>
        <span className={`text-xs ${keySaved ? 'text-signal-good' : 'text-ink-500'}`}>
          {info === null ? 'checking…' : keySaved ? 'API key saved' : 'no API key'}
        </span>
      </div>
      <p className="mt-2 max-w-prose text-xs text-ink-500">
        AutoClip hands each reel to{' '}
        <a href={ZERNIO_URL} target="_blank" rel="noreferrer" className="text-ink-300 underline">
          Zernio ↗
        </a>
        , a posting service, which publishes it on your Instagram at the slot time.
      </p>
      <ol className="mt-5 grid gap-6">
        <RightsStep
          accepted={accepted}
          busy={busy}
          platform="Instagram"
          onAccept={() => void act(async () => void (await liveApi.acceptAttestation()))}
        />

        <Step n={2} done={keySaved} title="Zernio API key">
          <p className="max-w-prose text-xs leading-relaxed text-ink-400">
            Make a free account at{' '}
            <a href={ZERNIO_URL} target="_blank" rel="noreferrer" className="text-ink-200 underline">
              zernio.com ↗
            </a>
            , connect your Instagram (Business or Creator account) there, then copy an API key from{' '}
            <strong>Dashboard → API Keys</strong>.
          </p>
          {keySaved ? (
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <p className="text-xs text-ink-500">The key is saved in your OS keyring.</p>
              <button
                className="btn btn-quiet"
                disabled={busy}
                onClick={() => {
                  if (!window.confirm('Remove the Zernio API key? Instagram posting stops until you add one.')) return
                  void act(() => gtaApi.saveInstagramKey(''), 'Key removed.')
                }}
              >
                Remove key
              </button>
            </div>
          ) : (
            <form
              className="mt-3 grid gap-3"
              onSubmit={(e) => {
                e.preventDefault()
                if (!key.trim()) return
                void act(async () => {
                  const value = await gtaApi.saveInstagramKey(key.trim())
                  setKey('')
                  return value
                }, 'Key saved.')
              }}
            >
              <label className="text-xs text-ink-400">
                API key
                <input
                  className="field"
                  type="password"
                  autoComplete="off"
                  value={key}
                  onChange={(e) => setKey(e.target.value)}
                />
              </label>
              <div className="flex flex-wrap items-center gap-3">
                <button className="btn btn-primary" disabled={busy || !key.trim()}>
                  Save to keyring
                </button>
                {info?.reelrepost_key_available && (
                  <button
                    type="button"
                    className="btn btn-quiet"
                    disabled={busy}
                    onClick={() => void act(() => gtaApi.importReelRepostKey(), 'Copied the key from ReelRepost.')}
                  >
                    Use ReelRepost's key
                  </button>
                )}
              </div>
            </form>
          )}
        </Step>

        <Step n={3} done={false} title="Test the connection">
          <p className="mb-3 text-xs text-ink-500">
            Checks the key and finds the Instagram account
            {account ? (
              <>
                {' '}
                <span className="font-mono text-ink-300">@{account.replace(/^@/, '')}</span> (from the schedule)
              </>
            ) : (
              ' (the first one connected at Zernio, or set one in the schedule below)'
            )}
            .
          </p>
          <button className="btn btn-ghost" disabled={busy || !keySaved} onClick={test}>
            Test connection
          </button>
        </Step>
      </ol>
      {message && (
        <p
          className={`mt-4 whitespace-pre-line text-sm ${
            message.tone === 'good' ? 'text-signal-good' : 'text-signal-bad'
          }`}
        >
          {message.text}
        </p>
      )}
    </section>
  )
}

// --------------------------------------------------------------------------
// Schedule
// --------------------------------------------------------------------------

function ScheduleSection({ overview, onChange }: { overview: PostingOverview; onChange: () => void }) {
  const saved = overview.posting
  const savedKey = JSON.stringify(saved)
  const [draft, setDraft] = useState<GtaPosting>(saved)
  const [slotsText, setSlotsText] = useState(saved.slots.join(', '))
  const [dirty, setDirty] = useState(false)
  const [message, setMessage] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)

  // Follow the server while the form is untouched (keyed on content, not identity:
  // the overview is re-fetched every few seconds).
  useEffect(() => {
    if (dirty) return
    const fresh = JSON.parse(savedKey) as GtaPosting
    setDraft(fresh)
    setSlotsText(fresh.slots.join(', '))
  }, [savedKey, dirty])

  const edit = (patch: Partial<GtaPosting>) => {
    setDirty(true)
    setDraft((d) => ({ ...d, ...patch }))
  }

  const save = async () => {
    const slots = slotsText
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean)
    const instagram_account = draft.instagram_account.trim().replace(/^@/, '')
    try {
      await gtaApi.saveSettings({ posting: { ...draft, slots, instagram_account } })
      setDirty(false)
      setMessage({ tone: 'good', text: draft.enabled ? 'Saved — posting is on.' : 'Saved.' })
      onChange()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    }
  }

  const enabledPlatforms = PLATFORMS.filter((p) => saved[p])
  return (
    <section className="card min-w-0 p-5 sm:p-6">
      <h2 className="text-base font-semibold text-fg">Posting schedule</h2>
      <div className="mt-5 grid gap-5 sm:grid-cols-2">
        <label className="flex items-center gap-3 text-sm text-ink-100 sm:col-span-2">
          <input
            type="checkbox"
            className="size-4 accent-sodium-500"
            checked={draft.enabled}
            onChange={(e) => edit({ enabled: e.target.checked })}
          />
          Post finished clips automatically
        </label>
        <div className="flex flex-wrap gap-x-8 gap-y-3 sm:col-span-2">
          <label className="flex items-center gap-3 text-sm text-ink-200">
            <input
              type="checkbox"
              className="size-4 accent-sodium-500"
              checked={draft.youtube}
              onChange={(e) => edit({ youtube: e.target.checked })}
            />
            Post to YouTube
          </label>
          <label className="flex items-center gap-3 text-sm text-ink-200">
            <input
              type="checkbox"
              className="size-4 accent-sodium-500"
              checked={draft.instagram}
              onChange={(e) => edit({ instagram: e.target.checked })}
            />
            Post to Instagram
          </label>
        </div>
        <label className="text-xs text-ink-400 sm:col-span-2">
          Go-public times, your local time (24 h, comma separated)
          <input
            className="field"
            value={slotsText}
            placeholder="12:00, 17:00, 21:00"
            onChange={(e) => {
              setDirty(true)
              setSlotsText(e.target.value)
            }}
          />
        </label>
        <label className="text-xs text-ink-400">
          Send this long before each time (minutes)
          <input
            className="field"
            type="number"
            min={15}
            max={1440}
            step={15}
            value={draft.lead_minutes}
            onChange={(e) => edit({ lead_minutes: Number(e.target.value) })}
          />
        </label>
        <label className="text-xs text-ink-400">
          Instagram account (optional, @username)
          <input
            className="field"
            value={draft.instagram_account}
            placeholder="@yourname"
            autoComplete="off"
            onChange={(e) => edit({ instagram_account: e.target.value })}
          />
        </label>
        <label className="text-xs text-ink-400">
          YouTube category
          <select
            className="field"
            value={draft.category_id}
            onChange={(e) => edit({ category_id: e.target.value })}
          >
            {Object.entries(overview.categories).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-3 self-end text-sm text-ink-200">
          <input
            type="checkbox"
            className="size-4 accent-sodium-500"
            checked={draft.shorts_tag}
            onChange={(e) => edit({ shorts_tag: e.target.checked })}
          />
          YouTube: add #Shorts to the description
        </label>
      </div>
      <div className="mt-5 flex flex-wrap items-center gap-4">
        <button className="btn btn-primary" onClick={save} disabled={!dirty}>
          Save schedule
        </button>
        {message && (
          <span className={`text-sm ${message.tone === 'good' ? 'text-signal-good' : 'text-signal-bad'}`}>
            {message.text}
          </span>
        )}
      </div>
      <div className="mt-4 grid gap-1 text-sm text-ink-300">
        {!overview.running ? (
          <p>
            <span className="text-ink-500">Status: </span>
            AutoClip’s background worker is off, so nothing posts.
          </p>
        ) : enabledPlatforms.length === 0 ? (
          <p>
            <span className="text-ink-500">Status: </span>
            No platform selected.
          </p>
        ) : (
          enabledPlatforms.map((p) => {
            const state = overview.states[p]
            return (
              <p key={p}>
                <span className="text-ink-500">{PLATFORM_NAME[p]}: </span>
                {state?.message ?? ''}
                {state?.paused_until && (
                  <span className="text-ink-500"> (until {slotLabel(state.paused_until)})</span>
                )}
              </p>
            )
          })
        )}
      </div>
      <p className="mt-2 max-w-prose text-xs text-ink-500">
        Each time gets, per platform, the best-scored clip that passed QA and isn't posted there yet (normally
        the same clip on both). YouTube: it's uploaded as private and YouTube makes it public exactly at that
        time. Instagram: the reel is scheduled at Zernio for the same time. Both go out even if AutoClip is
        closed by then. A time that passes while AutoClip is off is skipped, never posted late. Exclude a clip
        with “Don't post” on its card.
      </p>
    </section>
  )
}

// --------------------------------------------------------------------------
// Upcoming + history
// --------------------------------------------------------------------------

function PlanSection({ overview }: { overview: PostingOverview }) {
  // One row per slot (entries arrive sorted by slot, then platform).
  const slots: { slot_at: string; entries: PlanEntry[] }[] = []
  for (const entry of overview.plan) {
    const last = slots[slots.length - 1]
    if (last && last.slot_at === entry.slot_at) last.entries.push(entry)
    else slots.push({ slot_at: entry.slot_at, entries: [entry] })
  }
  return (
    <section className="card min-w-0 p-5 sm:p-6">
      <h2 className="text-base font-semibold text-fg">Coming up</h2>
      {slots.length === 0 ? (
        <p className="mt-3 text-sm text-ink-500">No posting times set, or no platform selected.</p>
      ) : (
        <ul className="mt-3 divide-y divide-ink-800">
          {slots.map((slot) => (
            <li key={slot.slot_at} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 py-2 text-sm">
              <span className="w-40 font-mono text-xs text-ink-300">{slotLabel(slot.slot_at)}</span>
              <div className="grid min-w-0 flex-1 gap-1">
                {slot.entries.map((entry) => (
                  <div key={entry.platform} className="flex min-w-0 flex-wrap items-baseline gap-x-3">
                    <PlatformTag platform={entry.platform} />
                    <span className={`w-64 text-xs ${STATUS_TONE[entry.status]}`}>
                      {statusText(entry.platform, entry.status)}
                    </span>
                    <span className="min-w-0 flex-1 truncate text-ink-100">{entry.title}</span>
                  </div>
                ))}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

const RETRYABLE: ReadonlySet<PlanEntry['status']> = new Set(['failed', 'interrupted', 'checking'])

function HistorySection({ overview, onChange }: { overview: PostingOverview; onChange: () => void }) {
  const [error, setError] = useState('')
  const uploads = overview.uploads.filter((u) => u.status !== 'skipped')
  const retry = async (id: string, status: PlanEntry['status']) => {
    if (
      status === 'checking' &&
      !window.confirm('Only retry if the reel is NOT on Instagram/Zernio already. Retry?')
    )
      return
    setError('')
    try {
      await gtaApi.retryUpload(id)
      onChange()
    } catch (err) {
      setError(errorText(err))
    }
  }
  return (
    <section className="card min-w-0 p-5 sm:p-6">
      <h2 className="text-base font-semibold text-fg">Uploads</h2>
      {error && <p className="mt-3 text-sm text-signal-bad">{error}</p>}
      {uploads.length === 0 ? (
        <p className="mt-3 text-sm text-ink-500">Nothing uploaded yet.</p>
      ) : (
        <ul className="mt-3 divide-y divide-ink-800">
          {uploads.map((u) => (
            <li key={u.id} className="grid gap-1 py-3 text-sm">
              <div className="flex flex-wrap items-baseline gap-x-4">
                <span className="w-40 font-mono text-xs text-ink-300">{slotLabel(u.slot_at)}</span>
                <PlatformTag platform={u.platform} />
                <span className={`text-xs ${STATUS_TONE[u.status]}`}>
                  {statusText(u.platform, u.status)}
                  {u.status === 'failed' && ` (${u.attempts}/${MAX_UPLOAD_ATTEMPTS})`}
                </span>
                <span className="min-w-0 flex-1 truncate text-ink-100">{u.title || u.clip_title}</span>
                {u.url && (
                  <a href={u.url} target="_blank" rel="noreferrer" className="text-xs text-sodium-400 underline">
                    Open on {PLATFORM_NAME[u.platform]} ↗
                  </a>
                )}
                {RETRYABLE.has(u.status) && (
                  <button className="btn btn-quiet" onClick={() => retry(u.id, u.status)}>
                    Retry
                  </button>
                )}
              </div>
              {u.error && <p className="text-xs text-signal-bad">{u.error}</p>}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
