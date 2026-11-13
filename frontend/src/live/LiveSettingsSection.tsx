/**
 * Live Watch settings section (F3).
 *
 * Self-contained by work order: pages/Settings.tsx's Field / NumberField /
 * Select are module-private and must not be edited, so typed equivalents are
 * inlined in this file and only liveApi + the shared ErrorNote are imported.
 *
 * How this screen behaves, and why it isn't the obvious version:
 *
 * - Editing stages a draft; nothing leaves the browser until Save. Each
 *   top-level section (enable / buffer / rate limits / detection / uploads /
 *   workers) is deep-compared against the last loaded snapshot, and the
 *   sticky bar lists exactly which sections carry unsaved changes. Save is
 *   disabled until at least one section is dirty, and Revert drops the draft.
 * - The PUT sends the COMPLETE merged settings object, not just the dirty
 *   sections. A server that validates a full LiveSettings body (the plain
 *   FastAPI `body: LiveSettings` reading of the /api/live/settings contract)
 *   would default-reset every field the user never touched if it received a
 *   partial body — pydantic fills missing sections with defaults. Sending
 *   the whole merged object is safe under merge-semantics servers too, so it
 *   is the shape that works everywhere. Section-level dirty flags drive the
 *   bar copy only.
 * - Numbers parse strictly: an empty or malformed draft is an inline error
 *   and is never silently coerced into a value, placeholders never stand in
 *   for real values, and fields the backend types as int reject decimals
 *   instead of truncating behind the user's back.
 *
 * Contract deviations the integrator must know about (also in the report):
 *
 * - The rights attestation is NOT sent through liveApi — live/api.ts is
 *   frozen for this worker. setAttestation() POSTs directly to
 *   `POST /api/live/uploads/attestation` with `{"accepted": true}`, the
 *   endpoint A10 must expose for A7's keyring gate (A7 refuses uploads until
 *   live_rights_attestation="accepted:<date>"). Errors are decoded from the
 *   body's `detail` the same way request() in api.ts does.
 * - The frozen backend docstring has no read endpoint for the attestation
 *   state, so the UI can only confirm an acceptance it posted itself, this
 *   session. If A10 adds GET status, wire it into the initial load.
 */

import { useEffect, useState } from 'react'

import { liveApi, type LiveSettingsData } from '../live/api'
import { ErrorNote } from '../components/ErrorNote'

type AuthStatus = { connected: boolean; account: string | null }
type Platform = 'youtube' | 'instagram'

/** Drives the sticky bar's "Unsaved: …" copy. Order matches the sections. */
const SECTION_TITLES: ReadonlyArray<{ key: keyof LiveSettingsData; title: string }> = [
  { key: 'enabled', title: 'Enable' },
  { key: 'rolling_buffer', title: 'Buffer' },
  { key: 'rate_limit', title: 'Rate limits' },
  { key: 'detection', title: 'Detection' },
  { key: 'uploads', title: 'Uploads' },
  { key: 'workers', title: 'Workers' },
]

/**
 * Strict number parsing. Returns the value, or the user-facing reason it
 * cannot become one. Never coerces: '' is not 0, '3.5' is not 3 in an integer
 * field, and out-of-range is rejected rather than clamped.
 */
function parseNumber(
  raw: string,
  spec: { min?: number; max?: number; integer?: boolean },
): number | string {
  const text = raw.trim()
  if (text === '') return 'Enter a number'
  const parsed = Number(text)
  if (!Number.isFinite(parsed)) return 'Not a number'
  if (spec.integer && !Number.isInteger(parsed)) return 'Whole numbers only'
  if (parsed < (spec.min ?? Number.NEGATIVE_INFINITY)) {
    return spec.max !== undefined ? `Between ${spec.min} and ${spec.max}` : `At least ${spec.min}`
  }
  if (parsed > (spec.max ?? Number.POSITIVE_INFINITY)) return `At most ${spec.max}`
  return parsed
}

/** Minimal deep-equal for the settings tree (plain JSON objects/arrays only). */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  const left = a as Record<string, unknown>
  const right = b as Record<string, unknown>
  const leftKeys = Object.keys(left)
  if (leftKeys.length !== Object.keys(right).length) return false
  return leftKeys.every((key) => key in right && deepEqual(left[key], right[key]))
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/**
 * Best-effort error message from a non-2xx Response, mirroring the `detail`
 * decoding request() does — so server hints reach the user here too.
 */
async function describeHttpError(response: Response): Promise<string> {
  let message = `${response.status} ${response.statusText}`
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') {
      message = body.detail
    } else if (body.detail && typeof body.detail === 'object') {
      const detail = body.detail as { message?: unknown; hint?: unknown }
      const parts = [
        typeof detail.message === 'string' ? detail.message : '',
        typeof detail.hint === 'string' ? detail.hint : '',
      ].filter(Boolean)
      if (parts.length > 0) message = parts.join(' — ')
    }
  } catch {
    /* Body had no JSON; the status line is the best we have. */
  }
  return message
}

export function LiveSettingsSection() {
  const [saved, setSaved] = useState<LiveSettingsData | null>(null)
  const [draft, setDraft] = useState<LiveSettingsData | null>(null)
  const [loadError, setLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  const [saving, setSaving] = useState(false)
  const [justSaved, setJustSaved] = useState(false)

  const [auth, setAuth] = useState<{ youtube: AuthStatus | null; instagram: AuthStatus | null }>({
    youtube: null,
    instagram: null,
  })
  const [authBusy, setAuthBusy] = useState<Platform | null>(null)
  const [authError, setAuthError] = useState<{ youtube: string | null; instagram: string | null }>({
    youtube: null,
    instagram: null,
  })

  const [attestationChecked, setAttestationChecked] = useState(false)
  const [attestationBusy, setAttestationBusy] = useState(false)
  const [attestationError, setAttestationError] = useState<string | null>(null)
  const [attestationRecorded, setAttestationRecorded] = useState(false)

  const load = () => {
    setLoadError(null)
    liveApi
      .getSettings()
      .then((data) => {
        setSaved(data)
        setDraft(data)
      })
      .catch((err: unknown) => setLoadError(err instanceof Error ? err : new Error(messageOf(err))))
  }

  const recordAuthError = (platform: Platform, message: string | null) =>
    setAuthError((prev) =>
      platform === 'youtube' ? { ...prev, youtube: message } : { ...prev, instagram: message },
    )

  const loadAuth = (platform: Platform) => {
    const call = platform === 'youtube' ? liveApi.youTubeAuthStatus() : liveApi.instagramAuthStatus()
    call
      .then((status) => {
        setAuth((prev) =>
          platform === 'youtube' ? { ...prev, youtube: status } : { ...prev, instagram: status },
        )
        recordAuthError(platform, null)
      })
      .catch((err: unknown) => recordAuthError(platform, `Status unavailable: ${messageOf(err)}`))
  }

  // OAuth consent happens in the tab window.open() gave us, and the callback
  // lands on the backend — so the reliable moment to re-read status is when
  // the user comes back to THIS tab.
  useEffect(() => {
    const onFocus = () => {
      loadAuth('youtube')
      loadAuth('instagram')
    }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [])

  useEffect(() => {
    load()
    loadAuth('youtube')
    loadAuth('instagram')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const connect = async (platform: Platform) => {
    if (authBusy) return
    setAuthBusy(platform)
    recordAuthError(platform, null)
    try {
      const start =
        platform === 'youtube'
          ? await liveApi.startYouTubeAuth()
          : await liveApi.startInstagramAuth()
      if (!start.redirect_url) throw new Error('The server returned no redirect URL')
      window.open(start.redirect_url, '_blank', 'noopener,noreferrer')
    } catch (err: unknown) {
      recordAuthError(platform, messageOf(err))
    } finally {
      setAuthBusy(null)
    }
  }

  /**
   * Rights attestation — the one call that bypasses liveApi (see the header
   * note: api.ts is frozen for F3). Contract: POST /api/live/uploads/attestation
   * with {"accepted": true}; A7 persists the keyring flag and unlocks uploads.
   */
  const setAttestation = async () => {
    if (!attestationChecked || attestationBusy) return
    setAttestationBusy(true)
    setAttestationError(null)
    try {
      const response = await fetch('/api/live/uploads/attestation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accepted: true }),
      })
      if (!response.ok) throw new Error(await describeHttpError(response))
      setAttestationRecorded(true)
    } catch (err: unknown) {
      setAttestationError(messageOf(err))
    } finally {
      setAttestationBusy(false)
    }
  }

  const isDirty = (key: keyof LiveSettingsData): boolean =>
    !!draft && !!saved && !deepEqual(draft[key], saved[key])

  const dirtyTitles = SECTION_TITLES.filter((s) => isDirty(s.key)).map((s) => s.title)
  const dirty = dirtyTitles.length > 0

  const save = async () => {
    if (!draft || !dirty || saving) return
    setSaving(true)
    setSaveError(null)
    try {
      // Full merged object on purpose — a full-body-validating server must not
      // see a partial body it would fill with defaults.
      const next = await liveApi.putSettings(draft)
      setSaved(next)
      setDraft(next)
      setJustSaved(true)
      setTimeout(() => setJustSaved(false), 1600)
    } catch (err: unknown) {
      setSaveError(err instanceof Error ? err : new Error(messageOf(err)))
    } finally {
      setSaving(false)
    }
  }

  const revert = () => {
    if (saved) setDraft(saved)
  }

  // Section updaters — explicit per section so the compiler checks every key
  // instead of trusting a generic computed-property spread.
  const setEnabled = (enabled: boolean) => setDraft((d) => (d ? { ...d, enabled } : d))
  const setBuffer = (patch: Partial<LiveSettingsData['rolling_buffer']>) =>
    setDraft((d) => (d ? { ...d, rolling_buffer: { ...d.rolling_buffer, ...patch } } : d))
  const setRate = (patch: Partial<LiveSettingsData['rate_limit']>) =>
    setDraft((d) => (d ? { ...d, rate_limit: { ...d.rate_limit, ...patch } } : d))
  const setAudio = (patch: Partial<LiveSettingsData['detection']['audio']>) =>
    setDraft((d) => (d ? { ...d, detection: { ...d.detection, audio: { ...d.detection.audio, ...patch } } } : d))
  const setChat = (patch: Partial<LiveSettingsData['detection']['chat']>) =>
    setDraft((d) => (d ? { ...d, detection: { ...d.detection, chat: { ...d.detection.chat, ...patch } } } : d))
  const setYouTube = (patch: Partial<LiveSettingsData['uploads']['youtube']>) =>
    setDraft((d) => (d ? { ...d, uploads: { ...d.uploads, youtube: { ...d.uploads.youtube, ...patch } } } : d))
  const setInstagram = (patch: Partial<LiveSettingsData['uploads']['instagram']>) =>
    setDraft((d) => (d ? { ...d, uploads: { ...d.uploads, instagram: { ...d.uploads.instagram, ...patch } } } : d))
  const setWorkers = (patch: Partial<LiveSettingsData['workers']>) =>
    setDraft((d) => (d ? { ...d, workers: { ...d.workers, ...patch } } : d))

  if (loadError) {
    return (
      <div className="max-w-4xl pt-8">
        <ErrorNote error={loadError} />
        <button onClick={load} className="btn btn-ghost mt-4">
          Retry
        </button>
      </div>
    )
  }

  if (!draft) return <p className="max-w-4xl pt-8 text-sm text-ink-500">Loading…</p>

  return (
    <div className="max-w-4xl pt-8">
      <div className="rise border-b border-ink-800 pb-5">
        <h2 className="font-display text-3xl leading-none text-ink-100">Live settings</h2>
        <p className="mt-2 text-xs text-ink-500">
          Staged edits — one Save commits every changed section to the watch workers.
        </p>
      </div>

      <Section title="Enable Live Watch">
        <Toggle
          label="Watch streams and clip moments automatically"
          hint="Records every added stream into a rolling buffer and reacts to detected moments — capture costs CPU and disk even when nothing is found."
          checked={draft.enabled}
          onChange={setEnabled}
        />
      </Section>

      <Section title="Rolling buffer" note="The disk-backed window every stream is recorded into.">
        <div className="grid gap-5 sm:grid-cols-2">
          <NumberField
            label="Segment length (s)"
            hint="Stream time per buffer segment on disk."
            value={draft.rolling_buffer.segment_s}
            min={0.1}
            onCommit={(v) => setBuffer({ segment_s: v })}
          />
          <NumberField
            label="Retain (min)"
            hint="How much of each stream is kept, waiting for a trigger."
            value={draft.rolling_buffer.retain_minutes}
            min={0.1}
            onCommit={(v) => setBuffer({ retain_minutes: v })}
          />
          <NumberField
            label="Max disk (MB)"
            hint="Hard cap for one stream's buffer."
            value={draft.rolling_buffer.max_disk_mb}
            integer
            min={1}
            onCommit={(v) => setBuffer({ max_disk_mb: v })}
          />
          <NumberField
            label="Pre-roll (s)"
            hint="Seconds kept before the trigger moment, so clips start early enough."
            value={draft.rolling_buffer.pre_roll_s}
            min={0}
            onCommit={(v) => setBuffer({ pre_roll_s: v })}
          />
          <NumberField
            label="Post-roll (s)"
            hint="Seconds kept after the moment ends, so clips don't cut the punchline."
            value={draft.rolling_buffer.post_roll_s}
            min={0}
            onCommit={(v) => setBuffer({ post_roll_s: v })}
          />
        </div>
      </Section>

      <Section title="Rate limits" note="Every judge call spends LLM tokens — these are the brakes.">
        <div className="grid gap-5 sm:grid-cols-2">
          <NumberField
            label="Judge calls / min"
            hint="Max candidate windows per stream per minute — each one costs tokens."
            value={draft.rate_limit.judge_per_minute}
            integer
            min={1}
            onCommit={(v) => setRate({ judge_per_minute: v })}
          />
          <NumberField
            label="Judge calls / hour"
            hint="Per-stream hourly cap — the main throttle on long streams."
            value={draft.rate_limit.judge_per_hour}
            integer
            min={1}
            onCommit={(v) => setRate({ judge_per_hour: v })}
          />
          <NumberField
            label="Judge calls / day"
            hint="Per-stream daily ceiling — the strict budget backstop."
            value={draft.rate_limit.judge_per_day}
            integer
            min={1}
            onCommit={(v) => setRate({ judge_per_day: v })}
          />
          <NumberField
            label="Min score (0-100)"
            hint="Verdicts scoring below this are dropped — higher keeps fewer, better clips."
            value={draft.rate_limit.judge_min_score}
            integer
            min={0}
            max={100}
            onCommit={(v) => setRate({ judge_min_score: v })}
          />
          <NumberField
            label="Cooldown (s)"
            hint="Quiet time after a clip before the same stream can spend tokens again."
            value={draft.rate_limit.cooldown_s}
            min={0}
            onCommit={(v) => setRate({ cooldown_s: v })}
          />
        </div>
      </Section>

      <Section
        title="Detection"
        note="Cheap always-on signals that choose which windows the LLM judge sees."
      >
        <h3 className="eyebrow">Audio</h3>
        <div className="mt-3 grid gap-5 sm:grid-cols-2">
          <NumberField
            label="Baseline window (s)"
            hint="Recent audio the loudness baseline is averaged over."
            value={draft.detection.audio.baseline_window_s}
            min={1}
            onCommit={(v) => setAudio({ baseline_window_s: v })}
          />
          <NumberField
            label="Spike z-score"
            hint="How many deviations louder than baseline counts as a spike."
            value={draft.detection.audio.spike_z}
            min={0.1}
            onCommit={(v) => setAudio({ spike_z: v })}
          />
          <NumberField
            label="RMS floor (0-1)"
            hint="Below this loudness a spike is ignored — quiet streams can't hype."
            value={draft.detection.audio.floor_rms}
            min={0}
            max={1}
            onCommit={(v) => setAudio({ floor_rms: v })}
          />
          <NumberField
            label="Max sustained (s)"
            hint="Longest run of spike seconds treated as one burst."
            value={draft.detection.audio.max_sustained_s}
            min={1}
            onCommit={(v) => setAudio({ max_sustained_s: v })}
          />
        </div>

        <h3 className="eyebrow mt-8">Chat</h3>
        <div className="mt-3">
          <Toggle
            label="Read Twitch chat"
            hint="Chat rate is the second detection signal — the only chat source in v1."
            checked={draft.detection.chat.enabled}
            onChange={(v) => setChat({ enabled: v })}
          />
        </div>
        <div className="mt-5 grid gap-5 sm:grid-cols-2">
          <NumberField
            label="Baseline window (s)"
            hint="Chat history the message-rate baseline uses."
            value={draft.detection.chat.baseline_window_s}
            min={1}
            onCommit={(v) => setChat({ baseline_window_s: v })}
          />
          <NumberField
            label="Rate z-score"
            hint="Message-rate deviation that counts as a chat spike."
            value={draft.detection.chat.rate_z}
            min={0.1}
            onCommit={(v) => setChat({ rate_z: v })}
          />
          <NumberField
            label="Floor msgs / min"
            hint="Below this rate chat counts as quiet, whatever the z-score says."
            value={draft.detection.chat.floor_msgs_per_min}
            min={0}
            onCommit={(v) => setChat({ floor_msgs_per_min: v })}
          />
          <NumberField
            label="Emotes per 100 msgs"
            hint="Emote / CAPS words per 100 messages flagged as hype."
            value={draft.detection.chat.emotes_threshold}
            min={0}
            onCommit={(v) => setChat({ emotes_threshold: v })}
          />
          <NumberField
            label="Hype train msgs / min"
            hint="Above this velocity, flag immediately regardless of other signals."
            value={draft.detection.chat.hypetrain_msgs_per_min}
            min={1}
            onCommit={(v) => setChat({ hypetrain_msgs_per_min: v })}
          />
        </div>
      </Section>

      <Section
        title="Uploads"
        note="Official platform APIs via OAuth — credentials stay in your OS keyring."
      >
        <p className="rounded-lg border border-sodium-500/30 bg-sodium-500/5 p-4 text-sm leading-relaxed text-ink-300">
          <strong className="text-sodium-400">
            You are responsible for having the rights or permission to clip and repost each
            streamer's content
          </strong>
          , and for following each platform's terms of service. Auto-upload refuses to run
          until you accept the attestation below.
        </p>

        <div className="mt-6 grid items-start gap-10 sm:grid-cols-2">
          <div className="space-y-5">
            <h3 className="eyebrow">YouTube</h3>
            <Toggle
              label="Auto-upload finished clips"
              hint="Uploads to the connected channel as soon as a clip renders."
              checked={draft.uploads.youtube.auto_upload}
              onChange={(v) => setYouTube({ auto_upload: v })}
            />
            <Field
              label="Title prefix"
              placeholder="e.g. [LIVE]"
              hint="Prepended to every auto-uploaded title."
              value={draft.uploads.youtube.title_prefix}
              onCommit={(v) => setYouTube({ title_prefix: v })}
            />
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => void connect('youtube')}
                disabled={authBusy !== null}
                className="btn btn-ghost"
              >
                {authBusy === 'youtube' ? 'Opening…' : auth.youtube?.connected ? 'Reconnect' : 'Connect'}
              </button>
              <StatusChip status={auth.youtube} />
            </div>
            {authError.youtube && (
              <p className="text-xs leading-snug text-signal-bad">{authError.youtube}</p>
            )}
          </div>

          <div className="space-y-5">
            <h3 className="eyebrow">Instagram</h3>
            <Toggle
              label="Auto-upload finished clips"
              hint="Publishes as a Reel as soon as a clip renders. Needs a professional account."
              checked={draft.uploads.instagram.auto_upload}
              onChange={(v) => setInstagram({ auto_upload: v })}
            />
            <Field
              label="Caption prefix"
              placeholder="e.g. 🔥"
              hint="Prepended to every auto-uploaded caption."
              value={draft.uploads.instagram.caption_prefix}
              onCommit={(v) => setInstagram({ caption_prefix: v })}
            />
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => void connect('instagram')}
                disabled={authBusy !== null}
                className="btn btn-ghost"
              >
                {authBusy === 'instagram' ? 'Opening…' : auth.instagram?.connected ? 'Reconnect' : 'Connect'}
              </button>
              <StatusChip status={auth.instagram} />
            </div>
            {authError.instagram && (
              <p className="text-xs leading-snug text-signal-bad">{authError.instagram}</p>
            )}
          </div>
        </div>

        <button
          onClick={() => {
            loadAuth('youtube')
            loadAuth('instagram')
          }}
          className="btn btn-quiet mt-4 text-xs"
        >
          Recheck connection status
        </button>

        <div className="mt-8 border-l-2 border-sodium-600 pl-4">
          <label className="flex items-start gap-3 text-sm text-ink-200">
            <input
              type="checkbox"
              checked={attestationChecked}
              onChange={(e) => setAttestationChecked(e.target.checked)}
              className="mt-0.5 size-4 accent-sodium-500"
            />
            <span>
              I have the rights to clip these streams
              <span className="mt-1 block text-xs leading-snug text-ink-500">
                Accepted once, stored in your OS keyring — uploads stay refused while it is unset.
              </span>
            </span>
          </label>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <button
              onClick={() => void setAttestation()}
              disabled={!attestationChecked || attestationBusy}
              className="btn btn-primary"
            >
              {attestationBusy ? 'Recording…' : 'Set attestation'}
            </button>
            {attestationRecorded && (
              <span className="text-xs text-signal-good">recorded — uploads unlocked</span>
            )}
            {attestationError && (
              <p className="max-w-prose text-xs leading-snug text-signal-bad">{attestationError}</p>
            )}
          </div>
        </div>
      </Section>

      <Section title="Workers" note="Resource ceilings for watching and rendering.">
        <div className="grid gap-5 sm:grid-cols-3">
          <NumberField
            label="Max concurrent streams"
            hint="Each watched stream runs its own ffmpeg capture."
            value={draft.workers.max_concurrent_streams}
            integer
            min={1}
            onCommit={(v) => setWorkers({ max_concurrent_streams: v })}
          />
          <NumberField
            label="Clip workers"
            hint="Heavy render jobs at once — more than one fights for the GPU."
            value={draft.workers.clip_queue_workers}
            integer
            min={1}
            onCommit={(v) => setWorkers({ clip_queue_workers: v })}
          />
          <NumberField
            label="Clip queue max"
            hint="Queued clips; overflow is dropped with a warning, not queued."
            value={draft.workers.clip_queue_max}
            integer
            min={1}
            onCommit={(v) => setWorkers({ clip_queue_max: v })}
          />
        </div>
        <div className="mt-5 grid gap-5 sm:grid-cols-2">
          <NumberField
            label="CPU budget (%)"
            hint="Percent of logical cores budgeted for live analysis passes."
            value={draft.workers.cpu_budget_percent}
            integer
            min={1}
            max={100}
            onCommit={(v) => setWorkers({ cpu_budget_percent: v })}
          />
          <NumberField
            label="RAM budget (GB)"
            hint="Advisory combined memory cap for the heavy stages."
            value={draft.workers.ram_budget_gb}
            integer
            min={1}
            onCommit={(v) => setWorkers({ ram_budget_gb: v })}
          />
        </div>
      </Section>

      {saveError && (
        <div className="mt-8">
          <ErrorNote error={saveError} onDismiss={() => setSaveError(null)} />
        </div>
      )}

      <div className="sticky bottom-0 mt-12 border-t border-ink-800 bg-ink-900 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="text-xs text-ink-400">
            {dirty ? (
              <>
                Unsaved:{' '}
                <span className="text-sodium-400">{dirtyTitles.join(', ')}</span>
              </>
            ) : (
              'No unsaved changes.'
            )}
          </p>
          <div className="flex items-center gap-3">
            <span
              className="text-xs text-signal-good transition-opacity duration-300"
              style={{ opacity: justSaved ? 1 : 0 }}
            >
              saved
            </span>
            <button onClick={revert} disabled={!dirty || saving} className="btn btn-ghost">
              Revert
            </button>
            <button onClick={() => void save()} disabled={!dirty || saving} className="btn btn-primary">
              {saving ? 'Saving…' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function Section({
  title,
  note,
  children,
}: {
  title: string
  note?: string
  children: React.ReactNode
}) {
  return (
    <section className="rise mt-14">
      <div className="border-b border-ink-800 pb-2">
        <h3 className="eyebrow">{title}</h3>
        {note && <p className="mt-1 text-xs text-ink-500">{note}</p>}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  )
}

function StatusChip({ status }: { status: AuthStatus | null }) {
  if (!status) return <span className="text-xs text-ink-600">status unknown</span>
  if (status.connected) {
    return (
      <span className="text-xs text-signal-good">
        connected{status.account ? ` · ${status.account}` : ''}
      </span>
    )
  }
  return <span className="text-xs text-ink-500">not connected</span>
}

/** Text input committing on blur, so the draft isn't touched per keystroke. */
function Field({
  label,
  hint,
  value,
  placeholder,
  onCommit,
}: {
  label: string
  hint?: string
  value: string
  placeholder?: string
  onCommit: (value: string) => void
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        className="field mt-1 text-sm"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => draft !== value && onCommit(draft)}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        spellCheck={false}
      />
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

/**
 * Number input with strict parsing: an empty or malformed draft is an inline
 * error and never becomes a value — invalid edits simply don't enter the
 * draft, so Save can't ship them.
 */
function NumberField({
  label,
  hint,
  value,
  min,
  max,
  integer,
  onCommit,
}: {
  label: string
  hint?: string
  value: number
  min?: number
  max?: number
  integer?: boolean
  onCommit: (value: number) => void
}) {
  const [draft, setDraft] = useState(String(value))
  const [touched, setTouched] = useState(false)

  useEffect(() => {
    setDraft(String(value))
    setTouched(false)
  }, [value])

  const parsed = parseNumber(draft, { min, max, integer })
  const error = typeof parsed === 'string' && (touched || draft.trim() !== '') ? parsed : null

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric mt-1 text-sm"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          // Flag the field even when the draft is empty, so a blank input is
          // visibly wrong instead of silently meaning zero.
          setTouched(true)
          if (typeof parsed === 'number' && parsed !== value) onCommit(parsed)
        }}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
      {error ? (
        <span className="mt-1.5 block text-xs leading-snug text-signal-bad">{error}</span>
      ) : (
        hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>
      )}
    </label>
  )
}

function Toggle({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string
  hint?: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex items-start gap-3 text-sm text-ink-200">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5 size-4 accent-sodium-500"
      />
      <span>
        {label}
        {hint && <span className="mt-1 block text-xs leading-snug text-ink-500">{hint}</span>}
      </span>
    </label>
  )
}
