/**
 * GTA Mode → YouTube: connect the channel (shared with Live Watch), set the
 * posting schedule, and watch what goes out when.
 *
 * Each slot gets the best-scored clip that passed QA. It is uploaded as
 * private shortly before the slot with a scheduled publish time, so YouTube
 * makes it public exactly at the slot.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { ApiError } from '../api'
import { liveApi, type LiveAttestation, type LivePlatformStatus } from '../live/api'
import {
  MAX_UPLOAD_ATTEMPTS,
  gtaApi,
  slotLabel,
  type GtaPosting,
  type PlanEntry,
  type PostingOverview,
} from './api'

const POLL_MS = 4000
const CONSOLE_URL = 'https://console.cloud.google.com/apis/credentials'

function errorText(err: unknown): string {
  if (err instanceof ApiError) return err.hint ? `${err.message}\n${err.hint}` : err.message
  return err instanceof Error ? err.message : String(err)
}

const STATUS_TEXT: Record<PlanEntry['status'], string> = {
  open: 'Open — no finished clip yet',
  missed: 'Missed',
  planned: 'Next in line',
  uploading: 'Uploading…',
  scheduled: 'On YouTube (goes public at this time)',
  failed: 'Failed',
  interrupted: 'Interrupted — check YouTube Studio',
  skipped: 'Skipped',
}

const STATUS_TONE: Record<PlanEntry['status'], string> = {
  open: 'text-ink-500',
  missed: 'text-ink-500',
  planned: 'text-ink-300',
  uploading: 'text-sodium-400',
  scheduled: 'text-signal-good',
  failed: 'text-signal-bad',
  interrupted: 'text-signal-bad',
  skipped: 'text-ink-500',
}

export function YouTubeTab() {
  const [params, setParams] = useSearchParams()
  const [status, setStatus] = useState<LivePlatformStatus | null>(null)
  const [attestation, setAttestation] = useState<LiveAttestation | null>(null)
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
    <div className="mt-8 grid max-w-4xl gap-12">
      {error && <p className="whitespace-pre-line text-sm text-signal-bad">{error}</p>}
      {note && <p className="text-sm text-signal-good">{note}</p>}
      <ConnectSection status={status} attestation={attestation} onChange={refreshAll} />
      {overview && <ScheduleSection overview={overview} onChange={refreshAll} />}
      {overview && <PlanSection overview={overview} />}
      {overview && <HistorySection overview={overview} onChange={loadOverview} />}
    </div>
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

function ConnectSection({
  status,
  attestation,
  onChange,
}: {
  status: LivePlatformStatus | null
  attestation: LiveAttestation | null
  onChange: () => void
}) {
  const [agreed, setAgreed] = useState(false)
  const [clientId, setClientId] = useState('')
  const [secret, setSecret] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [copied, setCopied] = useState(false)
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
    <section>
      <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-ink-800 pb-2">
        <h2 className="eyebrow">YouTube channel</h2>
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
                I have the rights or permission to clip and repost the streams I add, and I follow
                YouTube's terms.
              </label>
              <button
                className="btn btn-primary mt-3"
                disabled={!agreed || busy}
                onClick={() => act(() => liveApi.acceptAttestation())}
              >
                Accept and unlock uploads
              </button>
            </>
          )}
        </Step>

        <Step n={2} done={hasCredentials} title="Your Google OAuth app">
          {hasCredentials ? (
            <p className="text-xs text-ink-500">Client ID and secret are saved in your OS keyring.</p>
          ) : (
            <form
              className="grid gap-3"
              onSubmit={(e) => {
                e.preventDefault()
                if (!clientId.trim() || !secret.trim()) return
                void act(async () => {
                  await liveApi.saveCredentials('youtube', clientId.trim(), secret.trim())
                  setClientId('')
                  setSecret('')
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
                  <code ref={redirectRef} className="bg-ink-850 px-2 py-1 font-mono text-xs text-ink-200">
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
              <button
                className="btn btn-primary justify-self-start"
                disabled={busy || !clientId.trim() || !secret.trim()}
              >
                Save to keyring
              </button>
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
    try {
      await gtaApi.saveSettings({ posting: { ...draft, slots } })
      setDirty(false)
      setMessage({ tone: 'good', text: draft.enabled ? 'Saved — posting is on.' : 'Saved.' })
      onChange()
    } catch (err) {
      setMessage({ tone: 'bad', text: errorText(err) })
    }
  }

  const state = overview.state
  return (
    <section>
      <h2 className="eyebrow border-b border-ink-800 pb-2">Posting schedule</h2>
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
          Upload this long before each time (minutes)
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
        <label className="flex items-center gap-3 text-sm text-ink-200 sm:col-span-2">
          <input
            type="checkbox"
            className="size-4 accent-sodium-500"
            checked={draft.shorts_tag}
            onChange={(e) => edit({ shorts_tag: e.target.checked })}
          />
          Add #Shorts to the description
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
      <p className="mt-4 text-sm text-ink-300">
        <span className="text-ink-500">Status: </span>
        {state.running ? state.message : 'AutoClip’s background worker is off, so nothing posts.'}
        {state.paused_until && <span className="text-ink-500"> (until {slotLabel(state.paused_until)})</span>}
      </p>
      <p className="mt-2 max-w-prose text-xs text-ink-500">
        Each time gets the best-scored clip that passed QA. It's uploaded as private and YouTube makes it
        public exactly at that time, even if AutoClip is closed by then. A time that passes while AutoClip is
        off is skipped, never posted late. Exclude a clip with “Don't post” on its card.
      </p>
    </section>
  )
}

// --------------------------------------------------------------------------
// Upcoming + history
// --------------------------------------------------------------------------

function PlanSection({ overview }: { overview: PostingOverview }) {
  return (
    <section>
      <h2 className="eyebrow border-b border-ink-800 pb-2">Coming up</h2>
      {overview.plan.length === 0 ? (
        <p className="mt-3 text-sm text-ink-500">No posting times set.</p>
      ) : (
        <ul className="mt-3 divide-y divide-ink-800">
          {overview.plan.map((entry) => (
            <li key={entry.slot_at} className="flex flex-wrap items-baseline gap-x-4 gap-y-1 py-2 text-sm">
              <span className="w-40 font-mono text-xs text-ink-300">{slotLabel(entry.slot_at)}</span>
              <span className={`w-64 text-xs ${STATUS_TONE[entry.status]}`}>{STATUS_TEXT[entry.status]}</span>
              <span className="min-w-0 flex-1 truncate text-ink-100">{entry.title}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function HistorySection({ overview, onChange }: { overview: PostingOverview; onChange: () => void }) {
  const [error, setError] = useState('')
  const uploads = overview.uploads.filter((u) => u.status !== 'skipped')
  const retry = async (id: string) => {
    setError('')
    try {
      await gtaApi.retryUpload(id)
      onChange()
    } catch (err) {
      setError(errorText(err))
    }
  }
  return (
    <section>
      <h2 className="eyebrow border-b border-ink-800 pb-2">Uploads</h2>
      {error && <p className="mt-3 text-sm text-signal-bad">{error}</p>}
      {uploads.length === 0 ? (
        <p className="mt-3 text-sm text-ink-500">Nothing uploaded yet.</p>
      ) : (
        <ul className="mt-3 divide-y divide-ink-800">
          {uploads.map((u) => (
            <li key={u.id} className="grid gap-1 py-3 text-sm">
              <div className="flex flex-wrap items-baseline gap-x-4">
                <span className="w-40 font-mono text-xs text-ink-300">{slotLabel(u.slot_at)}</span>
                <span className={`text-xs ${STATUS_TONE[u.status]}`}>
                  {STATUS_TEXT[u.status]}
                  {u.status === 'failed' && ` (${u.attempts}/${MAX_UPLOAD_ATTEMPTS})`}
                </span>
                <span className="min-w-0 flex-1 truncate text-ink-100">{u.clip_title}</span>
                {u.url && (
                  <a href={u.url} target="_blank" rel="noreferrer" className="text-xs text-sodium-400 underline">
                    Open on YouTube ↗
                  </a>
                )}
                {(u.status === 'failed' || u.status === 'interrupted') && (
                  <button className="btn btn-quiet" onClick={() => retry(u.id)}>
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
