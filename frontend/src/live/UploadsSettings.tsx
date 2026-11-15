/**
 * Settings → Uploads: everything needed to go from nothing to auto-posting,
 * without leaving the app.
 *
 * Per platform it walks three steps: store your own OAuth app (write-only
 * client id/secret, plus the exact redirect URI to register), connect the
 * account (same tab — the consent page brings you back to /live?connected=1),
 * then posting preferences. The rights notice and its dated acceptance sit
 * above both, because uploads refuse to run without it.
 */

import { useCallback, useEffect, useId, useRef, useState } from 'react'

import {
  asError,
  describeUploadError,
  errorText,
  liveApi,
  platformLabel,
  type LiveAttestation,
  type LivePlatform,
  type LivePlatformStatus,
  type LiveSettingsData,
} from './api'
import { formatDate } from './format'
import { Spinner, announce } from './ui'

type UploadSettings = LiveSettingsData['uploads']

const PENDING_CONNECT_KEY = 'autoclip.live.pendingConnect'
const PENDING_CONNECT_TTL_MS = 30 * 60 * 1000

/** Remember which platform's consent page we left for, so the return can say so. */
function rememberPendingConnect(platform: LivePlatform) {
  try {
    window.sessionStorage.setItem(PENDING_CONNECT_KEY, JSON.stringify({ platform, at: Date.now() }))
  } catch {
    /* Storage blocked: the return toast just won't name the platform. */
  }
}

/** Read and clear the marker left by Connect; null when absent or stale. */
export function takePendingConnect(): LivePlatform | null {
  try {
    const raw = window.sessionStorage.getItem(PENDING_CONNECT_KEY)
    window.sessionStorage.removeItem(PENDING_CONNECT_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { platform?: unknown; at?: unknown }
    const fresh = typeof parsed.at === 'number' && Date.now() - parsed.at < PENDING_CONNECT_TTL_MS
    if (!fresh) return null
    return parsed.platform === 'youtube' || parsed.platform === 'instagram' ? parsed.platform : null
  } catch {
    return null
  }
}

const GUIDE: Record<
  LivePlatform,
  { console: string; consoleName: string; setup: string; connectNote: string; prefixLabel: string; prefixHint: string; placeholder: string; autoHint: string }
> = {
  youtube: {
    console: 'https://console.cloud.google.com/apis/credentials',
    consoleName: 'Google Cloud Console',
    setup:
      'Create an OAuth client ID of type “Web application” in a project with the YouTube Data API v3 enabled.',
    connectNote: 'Opens Google’s consent page in this tab and brings you back here.',
    prefixLabel: 'Title prefix',
    prefixHint: 'Put in front of every uploaded title.',
    placeholder: 'e.g. [LIVE] ',
    autoHint: 'Uploads to the connected channel as soon as a clip finishes rendering.',
  },
  instagram: {
    console: 'https://developers.facebook.com/apps/',
    consoleName: 'Meta for Developers',
    setup:
      'Create a Meta app with Facebook Login and Instagram publishing. The Instagram account must be a professional account linked to a Facebook Page.',
    connectNote: 'Opens Facebook’s consent page in this tab and brings you back here.',
    prefixLabel: 'Caption prefix',
    prefixHint: 'Put in front of every uploaded caption.',
    placeholder: 'e.g. 🔴 ',
    autoHint: 'Publishes a Reel as soon as a clip finishes rendering.',
  },
}

export function UploadsSettings({
  uploads,
  onChange,
  draftDirty,
  refreshSignal,
}: {
  /** The draft's upload preferences (staged with the rest of the settings). */
  uploads: UploadSettings
  onChange: (next: UploadSettings) => void
  /** Connecting leaves the page, so it waits until the draft is saved. */
  draftDirty: boolean
  /** Bumped after returning from a consent page. */
  refreshSignal: number
}) {
  const [attestation, setAttestation] = useState<LiveAttestation | null>(null)
  const [attestationError, setAttestationError] = useState<string | null>(null)
  const [agreed, setAgreed] = useState(false)
  const [accepting, setAccepting] = useState(false)
  const agreeId = useId()

  useEffect(() => {
    let cancelled = false
    liveApi
      .getAttestation()
      .then((value) => {
        if (cancelled) return
        setAttestation(value)
        setAttestationError(null)
      })
      .catch((error: unknown) => {
        if (!cancelled) setAttestationError(`Couldn’t read whether the notice was accepted: ${errorText(error)}`)
      })
    return () => {
      cancelled = true
    }
  }, [refreshSignal])

  const accept = async () => {
    if (!agreed || accepting) return
    setAccepting(true)
    setAttestationError(null)
    try {
      const value = await liveApi.acceptAttestation()
      // Read it back: a keyring that silently drops writes would otherwise
      // show "accepted" here while every upload is still refused.
      const stored = await liveApi.getAttestation().catch(() => null)
      if (stored && !stored.accepted) {
        setAttestationError(
          'The server took the acceptance but couldn’t keep it — this machine’s OS keyring may be unavailable.',
        )
        return
      }
      setAttestation({
        accepted: true,
        accepted_at: stored?.accepted_at ?? value.accepted_at ?? new Date().toISOString(),
      })
      announce('Rights notice accepted. Uploads are unlocked.')
    } catch (error) {
      setAttestationError(`Couldn’t record the acceptance: ${errorText(error)}`)
    } finally {
      setAccepting(false)
    }
  }

  const accepted = attestation?.accepted === true

  return (
    <>
      <div className="border-l-2 border-sodium-600 pl-4">
        <p className="max-w-prose text-sm leading-relaxed text-ink-200">
          <strong className="font-semibold text-sodium-400">
            You are responsible for having the rights or permission to clip and repost each
            streamer’s content
          </strong>
          , and for following each platform’s terms of service.
        </p>
        <p className="mt-2 max-w-prose text-xs leading-relaxed text-ink-500">
          Uploads stay locked until you accept this once. The acceptance is stored as a dated
          flag in your OS keyring, not in a config file.
        </p>

        {accepted ? (
          <p className="mt-4 flex items-center gap-2 text-sm text-signal-good">
            <span aria-hidden>✓</span>
            Accepted on {formatDate(attestation?.accepted_at ?? null)}
          </p>
        ) : (
          <div className="mt-4">
            <label htmlFor={agreeId} className="flex items-start gap-3 text-sm text-ink-200">
              <input
                id={agreeId}
                type="checkbox"
                checked={agreed}
                disabled={attestation === null && !attestationError}
                onChange={(event) => setAgreed(event.target.checked)}
                className="mt-0.5 size-4 accent-sodium-500"
              />
              I have the rights or permission to clip and repost the streams I add.
            </label>
            <button
              onClick={() => void accept()}
              disabled={!agreed || accepting}
              className="btn btn-primary mt-3"
            >
              {accepting ? 'Recording…' : 'Accept and unlock uploads'}
            </button>
          </div>
        )}
        {attestationError && (
          <p role="alert" className="mt-3 max-w-prose text-xs leading-snug text-signal-bad">
            {attestationError}
          </p>
        )}
      </div>

      <div className="mt-12 grid items-start gap-x-12 gap-y-14 lg:grid-cols-2">
        <PlatformSetup
          platform="youtube"
          autoUpload={uploads.youtube.auto_upload}
          prefix={uploads.youtube.title_prefix}
          onAutoUpload={(value) => onChange({ ...uploads, youtube: { ...uploads.youtube, auto_upload: value } })}
          onPrefix={(value) => onChange({ ...uploads, youtube: { ...uploads.youtube, title_prefix: value } })}
          accepted={attestation === null ? null : accepted}
          draftDirty={draftDirty}
          refreshSignal={refreshSignal}
        />
        <PlatformSetup
          platform="instagram"
          autoUpload={uploads.instagram.auto_upload}
          prefix={uploads.instagram.caption_prefix}
          onAutoUpload={(value) =>
            onChange({ ...uploads, instagram: { ...uploads.instagram, auto_upload: value } })
          }
          onPrefix={(value) => onChange({ ...uploads, instagram: { ...uploads.instagram, caption_prefix: value } })}
          accepted={attestation === null ? null : accepted}
          draftDirty={draftDirty}
          refreshSignal={refreshSignal}
        />
      </div>
    </>
  )
}

function PlatformSetup({
  platform,
  autoUpload,
  prefix,
  onAutoUpload,
  onPrefix,
  accepted,
  draftDirty,
  refreshSignal,
}: {
  platform: LivePlatform
  autoUpload: boolean
  prefix: string
  onAutoUpload: (value: boolean) => void
  onPrefix: (value: string) => void
  /** null while unknown. */
  accepted: boolean | null
  draftDirty: boolean
  refreshSignal: number
}) {
  const name = platformLabel(platform)
  const guide = GUIDE[platform]
  const [status, setStatus] = useState<LivePlatformStatus | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [checking, setChecking] = useState(true)
  const [clientId, setClientId] = useState('')
  const [clientSecret, setClientSecret] = useState('')
  const [savingCredentials, setSavingCredentials] = useState(false)
  const [credentialsNote, setCredentialsNote] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)
  const [connecting, setConnecting] = useState(false)
  const [connectError, setConnectError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const redirectRef = useRef<HTMLElement>(null)
  const ids = {
    heading: useId(),
    clientId: useId(),
    clientSecret: useId(),
    redirect: useId(),
    redirectNote: useId(),
    connectNote: useId(),
    prefix: useId(),
    prefixNote: useId(),
    auto: useId(),
  }

  const load = useCallback(async () => {
    setChecking(true)
    try {
      setStatus(await liveApi.authStatus(platform))
      setStatusError(null)
    } catch (error) {
      setStatusError(errorText(error))
    } finally {
      setChecking(false)
    }
  }, [platform])

  useEffect(() => {
    void load()
  }, [load, refreshSignal])

  const saveCredentials = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!clientId.trim() || !clientSecret.trim() || savingCredentials) return
    setSavingCredentials(true)
    setCredentialsNote(null)
    try {
      await liveApi.saveCredentials(platform, clientId.trim(), clientSecret.trim())
      setClientId('')
      setClientSecret('')
      // Read the status back: it's the proof the keyring actually kept them.
      const fresh = await liveApi.authStatus(platform).catch(() => null)
      if (fresh) {
        setStatus(fresh)
        setStatusError(null)
      }
      if (fresh?.has_credentials === false) {
        setCredentialsNote({
          tone: 'bad',
          text: 'The server took the credentials but couldn’t keep them — this machine’s OS keyring may be unavailable.',
        })
      } else {
        setCredentialsNote({ tone: 'good', text: 'Saved to your OS keyring.' })
        announce(`${name} app credentials saved.`)
      }
    } catch (error) {
      setCredentialsNote({ tone: 'bad', text: describeUploadError(asError(error).message, platform).message })
    } finally {
      setSavingCredentials(false)
    }
  }

  const connect = async () => {
    if (connecting) return
    setConnecting(true)
    setConnectError(null)
    try {
      const { redirect_url: target } = await liveApi.startAuth(platform)
      if (!target) throw new Error('The server returned no consent URL.')
      rememberPendingConnect(platform)
      // Same tab, on purpose: a new tab after an await gets popup-blocked, and
      // each extra tab holds its own live-feed connection.
      window.location.assign(target)
    } catch (error) {
      setConnectError(describeUploadError(asError(error).message, platform).message)
      setConnecting(false)
    }
  }

  const copyRedirect = async (uri: string) => {
    try {
      await navigator.clipboard.writeText(uri)
    } catch {
      // No clipboard permission: select it so Ctrl+C works.
      const element = redirectRef.current
      if (element) window.getSelection()?.selectAllChildren(element)
      return
    }
    setCopied(true)
    announce('Redirect URI copied.')
    window.setTimeout(() => setCopied(false), 2000)
  }

  const hasCredentials = status?.has_credentials
  const connected = status?.connected === true
  const redirectUri = status?.redirect_uri
  const connectBlocker =
    hasCredentials === false
      ? 'Save your OAuth app’s client ID and secret first.'
      : draftDirty
        ? 'Save or revert your settings changes first — connecting leaves this page.'
        : null
  const autoWarning = !autoUpload
    ? null
    : accepted === false
      ? 'Auto-upload is on, but uploads stay locked until you accept the rights notice above — finished clips are marked skipped.'
      : status && !connected
        ? `Auto-upload is on, but no ${name} account is connected — finished clips are marked skipped.`
        : null

  return (
    <section aria-labelledby={ids.heading} className="min-w-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-ink-800 pb-2">
        <h3 id={ids.heading} className="eyebrow">
          {name}
        </h3>
        <span className="text-xs" aria-live="polite">
          {checking && !status ? (
            <span className="inline-flex items-center gap-1.5 text-ink-500">
              <Spinner /> checking…
            </span>
          ) : statusError ? (
            <span className="text-signal-bad" title={statusError}>
              status unavailable
            </span>
          ) : connected ? (
            <span className="text-signal-good">
              connected{status?.account ? ` · ${status.account}` : ''}
            </span>
          ) : (
            <span className="text-ink-500">not connected</span>
          )}
        </span>
      </div>

      <ol className="mt-1">
        <Step number={1} done={hasCredentials === true} title="Your OAuth app">
          <p className="max-w-prose text-xs leading-relaxed text-ink-500">
            {guide.setup} Open the{' '}
            <a
              href={guide.console}
              target="_blank"
              rel="noreferrer"
              className="text-ink-300 underline underline-offset-4 hover:text-sodium-400"
            >
              {guide.consoleName} ↗
            </a>
            , then paste the client ID and secret here. They go to your OS keyring and are never
            shown again.
          </p>

          <form className="mt-4 space-y-4" onSubmit={(event) => void saveCredentials(event)} noValidate>
            <div>
              <label htmlFor={ids.clientId} className="eyebrow">
                Client ID
              </label>
              <input
                id={ids.clientId}
                className="field mt-1 font-mono text-sm"
                value={clientId}
                autoComplete="off"
                spellCheck={false}
                placeholder={hasCredentials ? 'stored — paste to replace' : 'paste the client ID'}
                onChange={(event) => setClientId(event.target.value)}
              />
            </div>
            <div>
              <label htmlFor={ids.clientSecret} className="eyebrow">
                Client secret
              </label>
              <input
                id={ids.clientSecret}
                type="password"
                className="field mt-1 font-mono text-sm"
                value={clientSecret}
                autoComplete="new-password"
                spellCheck={false}
                placeholder={hasCredentials ? '•••••••••••• stored' : 'paste the client secret'}
                onChange={(event) => setClientSecret(event.target.value)}
              />
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                className="btn btn-ghost"
                disabled={!clientId.trim() || !clientSecret.trim() || savingCredentials}
              >
                {savingCredentials ? 'Saving…' : hasCredentials ? 'Replace credentials' : 'Save credentials'}
              </button>
              {credentialsNote ? (
                <span
                  role={credentialsNote.tone === 'bad' ? 'alert' : undefined}
                  className={`text-xs ${credentialsNote.tone === 'good' ? 'text-signal-good' : 'text-signal-bad'}`}
                >
                  {credentialsNote.text}
                </span>
              ) : (
                hasCredentials && <span className="text-xs text-ink-500">Credentials are stored.</span>
              )}
            </div>
          </form>

          {redirectUri && (
            <div className="mt-6">
              <p id={ids.redirect} className="eyebrow">
                Redirect URI
              </p>
              {/* Shown whole, wrapping: the point is that it can be checked character for character. */}
              <div className="mt-2 flex items-start gap-3 border-b border-ink-700 pb-2">
                <code
                  ref={redirectRef}
                  aria-labelledby={ids.redirect}
                  aria-describedby={ids.redirectNote}
                  className="min-w-0 flex-1 break-all pt-1 font-mono text-xs leading-relaxed text-ink-200 select-all"
                >
                  {redirectUri}
                </code>
                <button
                  type="button"
                  onClick={() => void copyRedirect(redirectUri)}
                  className="btn btn-ghost shrink-0 px-3 py-1.5 text-xs"
                  aria-label={`Copy the ${name} redirect URI`}
                >
                  {copied ? 'Copied ✓' : 'Copy'}
                </button>
              </div>
              <p id={ids.redirectNote} className="mt-1.5 text-xs leading-snug text-ink-500">
                Add this exact address to your app’s authorized redirect URIs — character for
                character, or the consent page refuses to come back.
              </p>
            </div>
          )}
        </Step>

        <Step number={2} done={connected} title="Connect the account">
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => void connect()}
              disabled={connecting || connectBlocker !== null}
              aria-describedby={ids.connectNote}
              className={hasCredentials === true && !connected ? 'btn btn-primary' : 'btn btn-ghost'}
            >
              {connecting ? 'Opening…' : connected ? `Reconnect ${name}` : `Connect ${name}`}
            </button>
            <button
              type="button"
              onClick={() => void load()}
              disabled={checking}
              className="btn btn-quiet text-xs"
            >
              {checking && status ? 'Checking…' : 'Recheck'}
            </button>
          </div>
          <p
            id={ids.connectNote}
            className={`mt-2 text-xs leading-snug ${connectBlocker ? 'text-sodium-600' : 'text-ink-500'}`}
          >
            {connectBlocker ?? guide.connectNote}
          </p>
          {connectError && (
            <p role="alert" className="mt-2 max-w-prose text-xs leading-snug text-signal-bad">
              {connectError}
            </p>
          )}
          {statusError && (
            <p className="mt-2 max-w-prose text-xs leading-snug text-signal-bad">
              Couldn’t check the connection: {statusError}
            </p>
          )}
        </Step>

        <Step number={3} title="Posting">
          <label htmlFor={ids.auto} className="flex items-start gap-3 text-sm text-ink-200">
            <input
              id={ids.auto}
              type="checkbox"
              checked={autoUpload}
              onChange={(event) => onAutoUpload(event.target.checked)}
              className="mt-0.5 size-4 accent-sodium-500"
            />
            <span>
              Auto-upload finished clips
              <span className="mt-1 block text-xs leading-snug text-ink-500">{guide.autoHint}</span>
            </span>
          </label>
          {autoWarning && (
            <p className="mt-3 border-l-2 border-sodium-600 pl-3 text-xs leading-snug text-ink-300">
              {autoWarning}
            </p>
          )}
          <div className="mt-5">
            <label htmlFor={ids.prefix} className="eyebrow">
              {guide.prefixLabel}
            </label>
            <input
              id={ids.prefix}
              className="field mt-1 text-sm"
              value={prefix}
              maxLength={60}
              spellCheck={false}
              placeholder={guide.placeholder}
              aria-describedby={ids.prefixNote}
              onChange={(event) => onPrefix(event.target.value)}
            />
            <p id={ids.prefixNote} className="mt-1.5 text-xs leading-snug text-ink-500">
              {guide.prefixHint} Saved with the other settings.
            </p>
          </div>
        </Step>
      </ol>
    </section>
  )
}

/** One numbered setup step, styled like the job-progress checklist. */
function Step({
  number,
  done = false,
  title,
  children,
}: {
  number: number
  done?: boolean
  title: string
  children: React.ReactNode
}) {
  return (
    <li className="grid grid-cols-[2rem_minmax(0,1fr)] gap-x-3 border-b border-ink-850 py-5 last:border-b-0">
      <span
        aria-hidden
        className={`numeric pt-0.5 text-xs ${done ? 'text-signal-good' : 'text-ink-600'}`}
      >
        {done ? '✓' : String(number).padStart(2, '0')}
      </span>
      <div className="min-w-0">
        <p className="text-[0.9375rem] text-ink-100">
          {title}
          {done && <span className="sr-only"> (done)</span>}
        </p>
        <div className="mt-3">{children}</div>
      </div>
    </li>
  )
}
