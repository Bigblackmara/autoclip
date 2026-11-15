/**
 * Live Watch settings.
 *
 * Edits stage a draft; nothing leaves the browser until Save, and Save sends
 * the complete merged object (the server validates and replaces the whole
 * `live` section, then applies it to the running watchers at once).
 *
 * - Every number field validates as you type and reports to this parent, so
 *   Save is blocked while any field holds something that isn't a value. Fields
 *   are remounted on Save and Revert (`key={revision}`), which is what makes
 *   Revert reset a half-typed field instead of leaving its text behind.
 * - Detection and Workers are tuning knobs most people never touch; they sit
 *   behind an Advanced disclosure. Knobs this version doesn't enforce aren't
 *   offered as if they did.
 * - The draft's upload preferences live in Settings → Uploads with the rest of
 *   the upload setup (UploadsSettings).
 */

import { useCallback, useEffect, useId, useRef, useState } from 'react'

import { ErrorNote } from '../components/ErrorNote'
import { asError, errorText, liveApi, type LiveSettingsData } from './api'
import { plural } from './format'
import { UploadsSettings } from './UploadsSettings'
import { announce, scrollToElement } from './ui'

/** Drives the save bar's "Unsaved: …" copy. Order matches the page. */
const SECTION_TITLES: ReadonlyArray<{ key: keyof LiveSettingsData; title: string }> = [
  { key: 'enabled', title: 'Live Watch on/off' },
  { key: 'rolling_buffer', title: 'Buffer' },
  { key: 'rate_limit', title: 'Judge' },
  { key: 'uploads', title: 'Uploads' },
  { key: 'detection', title: 'Detection' },
  { key: 'workers', title: 'Workers' },
]

interface NumberSpec {
  min?: number
  max?: number
  integer?: boolean
}

/**
 * Strict parsing: the value, or the reason it can't be one. Never coerces —
 * '' is not 0, '3.5' is not 3 in a whole-number field, and out of range is
 * an error rather than a silent clamp.
 */
function parseNumber(raw: string, spec: NumberSpec): number | string {
  const text = raw.trim()
  if (text === '') return 'Enter a number.'
  const parsed = Number(text)
  if (!Number.isFinite(parsed)) return 'That isn’t a number.'
  if (spec.integer && !Number.isInteger(parsed)) return 'Whole numbers only.'
  if (spec.min !== undefined && parsed < spec.min) {
    return spec.max !== undefined ? `Between ${spec.min} and ${spec.max}.` : `At least ${spec.min}.`
  }
  if (spec.max !== undefined && parsed > spec.max) {
    return spec.min !== undefined ? `Between ${spec.min} and ${spec.max}.` : `At most ${spec.max}.`
  }
  return parsed
}

/** Minimal deep-equal for the settings tree (plain JSON objects only). */
function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true
  if (typeof a !== 'object' || typeof b !== 'object' || a === null || b === null) return false
  const left = a as Record<string, unknown>
  const right = b as Record<string, unknown>
  const keys = Object.keys(left)
  if (keys.length !== Object.keys(right).length) return false
  return keys.every((key) => key in right && deepEqual(left[key], right[key]))
}

function formatSeconds(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`
  const minutes = seconds / 60
  return `${Number.isInteger(minutes) ? minutes : minutes.toFixed(1)} min`
}

type Validity = (key: string, label: string, error: string | null) => void

export function LiveSettingsSection({
  onSettingsChange,
  onDirtyChange,
  focusEnableSignal,
  focusUploadsSignal,
  authRefreshSignal,
}: {
  /** The saved settings, on load and after every save. */
  onSettingsChange: (saved: LiveSettingsData) => void
  /** True while the draft differs from what's saved (or can't be saved). */
  onDirtyChange: (dirty: boolean) => void
  /** Bumped to scroll to and focus the on/off switch. */
  focusEnableSignal: number
  /** Bumped to scroll to the uploads setup. */
  focusUploadsSignal: number
  /** Bumped when upload status should be re-read (back from a consent page). */
  authRefreshSignal: number
}) {
  const [saved, setSaved] = useState<LiveSettingsData | null>(null)
  const [draft, setDraft] = useState<LiveSettingsData | null>(null)
  const [revision, setRevision] = useState(0)
  const [invalid, setInvalid] = useState<Record<string, { label: string; error: string }>>({})
  const [loadError, setLoadError] = useState<Error | null>(null)
  const [saveError, setSaveError] = useState<Error | null>(null)
  const [saving, setSaving] = useState(false)
  const [lastSave, setLastSave] = useState<{ applied: boolean; applyError?: string } | null>(null)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const enableRef = useRef<HTMLInputElement>(null)
  const uploadsRef = useRef<HTMLElement>(null)
  const pendingFocus = useRef<'enable' | 'uploads' | null>(null)
  const reportSettings = useRef(onSettingsChange)
  reportSettings.current = onSettingsChange
  const ids = { enable: useId(), advanced: useId() }

  const load = useCallback(() => {
    setLoadError(null)
    liveApi
      .getSettings()
      .then((data) => {
        setSaved(data)
        setDraft(data)
        setInvalid({})
        setRevision((current) => current + 1)
        reportSettings.current(data)
      })
      .catch((error: unknown) => setLoadError(asError(error)))
  }, [])

  useEffect(load, [load])

  const onValidity = useCallback<Validity>((key, label, error) => {
    setInvalid((current) => {
      if (error === null) {
        if (!(key in current)) return current
        const next = { ...current }
        delete next[key]
        return next
      }
      if (current[key]?.error === error) return current
      return { ...current, [key]: { label, error } }
    })
  }, [])

  const dirtyTitles =
    draft && saved
      ? SECTION_TITLES.filter(({ key }) => !deepEqual(draft[key], saved[key])).map(({ title }) => title)
      : []
  const dirty = dirtyTitles.length > 0
  const invalidList = Object.values(invalid)
  const blocked = invalidList.length > 0

  useEffect(() => {
    onDirtyChange(dirty || blocked)
  }, [dirty, blocked, onDirtyChange])

  useEffect(() => {
    if (dirty) setLastSave(null)
  }, [dirty])

  // "Turn on in Settings" and "Open upload settings" land here. The panel may
  // have just been un-hidden, or still be loading: try now, and again on load.
  const runPendingFocus = useCallback(() => {
    const target = pendingFocus.current
    if (!target) return
    const element = target === 'enable' ? enableRef.current : uploadsRef.current
    if (!element) return
    pendingFocus.current = null
    requestAnimationFrame(() => {
      scrollToElement(element, target === 'enable' ? 'center' : 'start')
      if (target === 'enable') element.focus({ preventScroll: true })
    })
  }, [])

  useEffect(() => {
    if (focusEnableSignal > 0) {
      pendingFocus.current = 'enable'
      runPendingFocus()
    }
  }, [focusEnableSignal, runPendingFocus])

  useEffect(() => {
    if (focusUploadsSignal > 0) {
      pendingFocus.current = 'uploads'
      runPendingFocus()
    }
  }, [focusUploadsSignal, runPendingFocus])

  useEffect(() => {
    if (draft) runPendingFocus()
  }, [draft, runPendingFocus])

  const save = async () => {
    if (!draft || !dirty || saving || blocked) return
    setSaving(true)
    setSaveError(null)
    try {
      // apply_error is response metadata, not a setting: keep it out of the draft.
      const { applied, apply_error: applyError, ...next } = await liveApi.putSettings(draft)
      setSaved(next)
      setDraft(next)
      setInvalid({})
      setRevision((current) => current + 1)
      setLastSave({ applied: applied === true, applyError: applyError || undefined })
      announce(
        applyError
          ? `Settings saved, but not applied: ${applyError}`
          : applied
            ? 'Settings saved and applied to the running watchers.'
            : 'Settings saved.',
      )
      onSettingsChange(next)
    } catch (error) {
      setSaveError(asError(error))
      announce(`Settings not saved: ${errorText(error)}`)
    } finally {
      setSaving(false)
    }
  }

  const revert = () => {
    if (!saved) return
    setDraft(saved)
    setInvalid({})
    setSaveError(null)
    setRevision((current) => current + 1)
    announce('Changes reverted.')
  }

  // Section updaters — explicit per section so the compiler checks every key.
  const setEnabled = (enabled: boolean) => setDraft((d) => (d ? { ...d, enabled } : d))
  const setBuffer = (patch: Partial<LiveSettingsData['rolling_buffer']>) =>
    setDraft((d) => (d ? { ...d, rolling_buffer: { ...d.rolling_buffer, ...patch } } : d))
  const setRate = (patch: Partial<LiveSettingsData['rate_limit']>) =>
    setDraft((d) => (d ? { ...d, rate_limit: { ...d.rate_limit, ...patch } } : d))
  const setAudio = (patch: Partial<LiveSettingsData['detection']['audio']>) =>
    setDraft((d) =>
      d ? { ...d, detection: { ...d.detection, audio: { ...d.detection.audio, ...patch } } } : d,
    )
  const setChat = (patch: Partial<LiveSettingsData['detection']['chat']>) =>
    setDraft((d) =>
      d ? { ...d, detection: { ...d.detection, chat: { ...d.detection.chat, ...patch } } } : d,
    )
  const setUploads = (uploads: LiveSettingsData['uploads']) => setDraft((d) => (d ? { ...d, uploads } : d))
  const setWorkers = (patch: Partial<LiveSettingsData['workers']>) =>
    setDraft((d) => (d ? { ...d, workers: { ...d.workers, ...patch } } : d))

  if (loadError) {
    return (
      <div className="max-w-4xl pt-10">
        <ErrorNote error={loadError} />
        <button onClick={load} className="btn btn-ghost mt-4">
          Try again
        </button>
      </div>
    )
  }

  if (!draft) return <p className="max-w-4xl pt-10 text-sm text-ink-500">Loading settings…</p>

  const buffer = draft.rolling_buffer
  const retentionS = buffer.retain_minutes * 60
  const rollS = buffer.pre_roll_s + buffer.post_roll_s
  const field = { onValidity }
  const advancedDirty = dirtyTitles.includes('Detection') || dirtyTitles.includes('Workers')

  return (
    <div className="max-w-4xl">
      <p className="mt-8 max-w-prose text-sm leading-relaxed text-ink-400">
        Changes are staged until you save. Saving applies them to the running watchers right
        away — no restart.
      </p>

      <Section title="Live Watch" note="The master switch for watching and clipping.">
        <Toggle
          id={ids.enable}
          inputRef={enableRef}
          label="Watch streams and clip moments automatically"
          hint="Each stream on your list is recorded into a rolling buffer. Capture costs CPU and disk even when nothing is found."
          checked={draft.enabled}
          onChange={setEnabled}
        />
      </Section>

      <Section
        title="Buffer & clip timing"
        note={`How much of each stream is kept on disk, and what a clip includes around its moment. Streams are recorded in fixed ${formatSeconds(buffer.segment_s)} segments.`}
      >
        <div key={revision} className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
          <NumberField
            {...field}
            name="retain_minutes"
            label="Keep (minutes)"
            hint="How much of each stream stays on disk, waiting for a moment."
            value={buffer.retain_minutes}
            spec={{ min: 0.1 }}
            onChange={(v) => setBuffer({ retain_minutes: v })}
          />
          <NumberField
            {...field}
            name="max_disk_mb"
            label="Disk cap per stream (MB)"
            hint="Hard ceiling for one stream’s buffer, whatever the minutes say."
            value={buffer.max_disk_mb}
            spec={{ min: 1, integer: true }}
            onChange={(v) => setBuffer({ max_disk_mb: v })}
          />
          <NumberField
            {...field}
            name="pre_roll_s"
            label="Pre-roll (s)"
            hint="Seconds kept before the moment, so the clip starts early enough."
            value={buffer.pre_roll_s}
            spec={{ min: 0 }}
            onChange={(v) => setBuffer({ pre_roll_s: v })}
          />
          <NumberField
            {...field}
            name="post_roll_s"
            label="Post-roll (s)"
            hint="Seconds kept after it, so the clip doesn’t cut the punchline."
            value={buffer.post_roll_s}
            spec={{ min: 0 }}
            onChange={(v) => setBuffer({ post_roll_s: v })}
          />
          <NumberField
            {...field}
            name="judge_context_s"
            label="Judge context (s)"
            hint="Lead-up the AI judge reads before a moment. More context, smarter picks, slightly slower verdicts."
            value={buffer.judge_context_s}
            spec={{ min: 0 }}
            onChange={(v) => setBuffer({ judge_context_s: v })}
          />
        </div>
        {(rollS > retentionS || buffer.judge_context_s > retentionS) && (
          <div className="mt-6 space-y-2 border-l-2 border-sodium-600 pl-4 text-sm leading-relaxed text-ink-300">
            {rollS > retentionS && (
              <p>
                Pre-roll plus post-roll ({formatSeconds(rollS)}) is longer than the buffer keeps (
                {formatSeconds(retentionS)}). The start of each clip would already be gone — keep
                more minutes or shorten the rolls.
              </p>
            )}
            {buffer.judge_context_s > retentionS && (
              <p className="text-ink-400">
                The judge can only read what the buffer holds, so its context is capped at{' '}
                {formatSeconds(retentionS)}.
              </p>
            )}
          </div>
        )}
      </Section>

      <Section title="Judge & budget" note="Every judged window is one LLM call on your key — these are the brakes.">
        <div key={revision} className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
          <NumberField
            {...field}
            name="judge_min_score"
            label="Minimum score (0–100)"
            hint="Moments the judge scores below this are dropped, with the score kept. Higher means fewer, better clips."
            value={draft.rate_limit.judge_min_score}
            spec={{ min: 0, max: 100, integer: true }}
            onChange={(v) => setRate({ judge_min_score: v })}
          />
          <NumberField
            {...field}
            name="cooldown_s"
            label="Cooldown (s)"
            hint="Quiet time after a clip before the same stream can trigger again."
            value={draft.rate_limit.cooldown_s}
            spec={{ min: 0 }}
            onChange={(v) => setRate({ cooldown_s: v })}
          />
          <NumberField
            {...field}
            name="judge_per_minute"
            label="Judge calls per minute"
            hint="Per stream."
            value={draft.rate_limit.judge_per_minute}
            spec={{ min: 1, integer: true }}
            onChange={(v) => setRate({ judge_per_minute: v })}
          />
          <NumberField
            {...field}
            name="judge_per_hour"
            label="Judge calls per hour"
            hint="Per stream — the main throttle on long streams."
            value={draft.rate_limit.judge_per_hour}
            spec={{ min: 1, integer: true }}
            onChange={(v) => setRate({ judge_per_hour: v })}
          />
          <NumberField
            {...field}
            name="judge_per_day"
            label="Judge calls per day"
            hint="Per stream — the hard ceiling on spend."
            value={draft.rate_limit.judge_per_day}
            spec={{ min: 1, integer: true }}
            onChange={(v) => setRate({ judge_per_day: v })}
          />
        </div>
      </Section>

      <Section
        title="Uploads"
        note="Optional. Clips go out through the official APIs with your own OAuth app; tokens and secrets stay in your OS keyring."
        sectionRef={uploadsRef}
      >
        <UploadsSettings
          uploads={draft.uploads}
          onChange={setUploads}
          draftDirty={dirty || blocked}
          refreshSignal={authRefreshSignal}
        />
      </Section>

      <section className="mt-16">
        <button
          type="button"
          onClick={() => setAdvancedOpen((open) => !open)}
          aria-expanded={advancedOpen}
          aria-controls={ids.advanced}
          className="btn btn-quiet -ml-1"
        >
          <span
            aria-hidden
            className="inline-block transition-transform duration-300"
            style={{ transform: advancedOpen ? 'rotate(90deg)' : 'none' }}
          >
            ›
          </span>
          {advancedOpen ? 'Hide advanced tuning' : 'Advanced tuning'}
          {!advancedOpen && advancedDirty && <span className="text-sodium-500">· unsaved</span>}
        </button>
        <p className="mt-1 text-xs text-ink-500">Detection thresholds and worker limits. The defaults suit most streams.</p>

        <div id={ids.advanced} hidden={!advancedOpen} className="rise">
          <Section title="Detection — audio" note="Loudness spikes measured against each stream’s own baseline.">
            <div key={revision} className="grid gap-x-8 gap-y-6 sm:grid-cols-2">
              <NumberField
                {...field}
                name="audio_baseline"
                label="Baseline window (s)"
                hint="Recent audio the loudness baseline is averaged over."
                value={draft.detection.audio.baseline_window_s}
                spec={{ min: 1 }}
                onChange={(v) => setAudio({ baseline_window_s: v })}
              />
              <NumberField
                {...field}
                name="spike_z"
                label="Spike z-score"
                hint="How many deviations above the baseline count as a spike."
                value={draft.detection.audio.spike_z}
                spec={{ min: 0.1 }}
                onChange={(v) => setAudio({ spike_z: v })}
              />
              <NumberField
                {...field}
                name="floor_rms"
                label="Loudness floor (0–1)"
                hint="Below this level a spike is ignored, so a quiet stream can’t fake hype."
                value={draft.detection.audio.floor_rms}
                spec={{ min: 0, max: 1 }}
                onChange={(v) => setAudio({ floor_rms: v })}
              />
              <NumberField
                {...field}
                name="max_sustained"
                label="Longest burst (s)"
                hint="The longest run of spike seconds still counted as one moment."
                value={draft.detection.audio.max_sustained_s}
                spec={{ min: 1 }}
                onChange={(v) => setAudio({ max_sustained_s: v })}
              />
            </div>
          </Section>

          <Section title="Detection — Twitch chat" note="Message rate and hype, read from the public chat — no account needed.">
            <Toggle
              label="Read Twitch chat"
              hint="The second detection signal. Twitch is the only chat source in this version."
              checked={draft.detection.chat.enabled}
              onChange={(v) => setChat({ enabled: v })}
            />
            <div key={revision} className="mt-6 grid gap-x-8 gap-y-6 sm:grid-cols-2">
              <NumberField
                {...field}
                name="chat_baseline"
                label="Baseline window (s)"
                hint="Chat history the message-rate baseline uses."
                value={draft.detection.chat.baseline_window_s}
                spec={{ min: 1 }}
                onChange={(v) => setChat({ baseline_window_s: v })}
              />
              <NumberField
                {...field}
                name="rate_z"
                label="Rate z-score"
                hint="Message-rate deviation that counts as a chat spike."
                value={draft.detection.chat.rate_z}
                spec={{ min: 0.1 }}
                onChange={(v) => setChat({ rate_z: v })}
              />
              <NumberField
                {...field}
                name="floor_msgs"
                label="Quiet below (msgs/min)"
                hint="Under this rate chat counts as quiet, whatever the z-score says."
                value={draft.detection.chat.floor_msgs_per_min}
                spec={{ min: 0 }}
                onChange={(v) => setChat({ floor_msgs_per_min: v })}
              />
              <NumberField
                {...field}
                name="emotes"
                label="Hype words per 100 msgs"
                hint="Emote and CAPS words per 100 messages that read as hype."
                value={draft.detection.chat.emotes_threshold}
                spec={{ min: 0 }}
                onChange={(v) => setChat({ emotes_threshold: v })}
              />
              <NumberField
                {...field}
                name="hypetrain"
                label="Hype train (msgs/min)"
                hint="Above this velocity a moment fires outright."
                value={draft.detection.chat.hypetrain_msgs_per_min}
                spec={{ min: 1 }}
                onChange={(v) => setChat({ hypetrain_msgs_per_min: v })}
              />
            </div>
          </Section>

          <Section title="Workers & disk" note="How much runs at once, and how much disk Live Watch may use.">
            <div key={revision} className="grid gap-x-8 gap-y-6 sm:grid-cols-3">
              <NumberField
                {...field}
                name="max_streams"
                label="Streams at once"
                hint="Each watched stream runs its own ffmpeg capture."
                value={draft.workers.max_concurrent_streams}
                spec={{ min: 1, integer: true }}
                onChange={(v) => setWorkers({ max_concurrent_streams: v })}
              />
              <NumberField
                {...field}
                name="clip_workers"
                label="Render workers"
                hint="Renders at once. More than one fights over the GPU."
                value={draft.workers.clip_queue_workers}
                spec={{ min: 1, integer: true }}
                onChange={(v) => setWorkers({ clip_queue_workers: v })}
              />
              <NumberField
                {...field}
                name="queue_max"
                label="Render queue size"
                hint="Accepted clips waiting to render. Overflow is dropped with a reason."
                value={draft.workers.clip_queue_max}
                spec={{ min: 1, integer: true }}
                onChange={(v) => setWorkers({ clip_queue_max: v })}
              />
            </div>
            {(typeof draft.workers.disk_budget_mb === 'number' ||
              typeof draft.workers.min_free_disk_mb === 'number') && (
              <div key={`disk-${revision}`} className="mt-6 grid gap-x-8 gap-y-6 sm:grid-cols-2">
                {typeof draft.workers.disk_budget_mb === 'number' && (
                  <NumberField
                    {...field}
                    name="disk_budget_mb"
                    label="Disk budget (MB)"
                    hint="Everything Live Watch keeps except finished clips: buffers, work in progress, leftovers of rejected moments. Over it, the oldest go first; then new moments are refused."
                    value={draft.workers.disk_budget_mb}
                    spec={{ min: 1, integer: true }}
                    onChange={(v) => setWorkers({ disk_budget_mb: v })}
                  />
                )}
                {typeof draft.workers.min_free_disk_mb === 'number' && (
                  <NumberField
                    {...field}
                    name="min_free_disk_mb"
                    label="Keep free on disk (MB)"
                    hint="New moments are refused while the drive has less free space than this."
                    value={draft.workers.min_free_disk_mb}
                    spec={{ min: 0, integer: true }}
                    onChange={(v) => setWorkers({ min_free_disk_mb: v })}
                  />
                )}
              </div>
            )}
            <p className="mt-6 text-xs leading-relaxed text-ink-500">
              The CPU and RAM budgets in config.json are advisory in this version — nothing
              enforces them yet, so they aren’t offered here.
            </p>
          </Section>
        </div>
      </section>

      {saveError && (
        <div className="mt-10">
          <ErrorNote error={saveError} onDismiss={() => setSaveError(null)} />
        </div>
      )}

      <div className="sticky bottom-0 z-10 mt-14 border-t border-ink-800 bg-ink-900/95 py-4 backdrop-blur-[2px]">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
          <p className="min-w-0 text-xs leading-snug" aria-live="polite">
            {blocked ? (
              <span className="text-signal-bad">
                Fix {plural(invalidList.length, 'field')} before saving:{' '}
                {invalidList.map((item) => item.label).join(', ')}
              </span>
            ) : dirty ? (
              <span className="text-ink-400">
                Unsaved: <span className="text-sodium-400">{dirtyTitles.join(', ')}</span>
              </span>
            ) : lastSave?.applyError ? (
              <span className="text-signal-bad">
                Saved, but the running watchers did not take it: {lastSave.applyError}
              </span>
            ) : lastSave ? (
              <span className="text-signal-good">
                {lastSave.applied ? 'Saved — applied to the running watchers.' : 'Saved.'}
              </span>
            ) : (
              <span className="text-ink-500">All changes saved.</span>
            )}
          </p>
          <div className="flex items-center gap-3">
            <button onClick={revert} disabled={(!dirty && !blocked) || saving} className="btn btn-ghost">
              Revert
            </button>
            <button onClick={() => void save()} disabled={!dirty || saving || blocked} className="btn btn-primary">
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
  sectionRef,
  children,
}: {
  title: string
  note?: string
  sectionRef?: React.Ref<HTMLElement>
  children: React.ReactNode
}) {
  const headingId = useId()
  return (
    <section ref={sectionRef} aria-labelledby={headingId} className="rise mt-14 scroll-mt-6">
      <div className="border-b border-ink-800 pb-2">
        <h2 id={headingId} className="eyebrow">
          {title}
        </h2>
        {note && <p className="mt-1 text-xs leading-relaxed text-ink-500">{note}</p>}
      </div>
      <div className="mt-6">{children}</div>
    </section>
  )
}

/**
 * Number input that validates on every keystroke and reports to the parent.
 * A valid value goes straight into the draft; an invalid one shows its reason
 * here and blocks Save there. The text is this field's own state — it never
 * re-syncs from `value` mid-edit (that would eat "0." while typing "0.05");
 * the parent remounts the field to reset it.
 */
function NumberField({
  name,
  label,
  hint,
  value,
  spec,
  onChange,
  onValidity,
}: {
  name: string
  label: string
  hint?: string
  value: number
  spec: NumberSpec
  onChange: (value: number) => void
  onValidity: Validity
}) {
  const [text, setText] = useState(() => String(value))
  const id = useId()
  const noteId = `${id}-note`
  const parsed = parseNumber(text, spec)
  const error = typeof parsed === 'string' ? parsed : null

  // Report the starting state too: a stored value outside the range shows up
  // immediately. The cleanup un-reports on remount and unmount.
  useEffect(() => {
    const initial = parseNumber(String(value), spec)
    onValidity(name, label, typeof initial === 'string' ? initial : null)
    return () => onValidity(name, label, null)
    // Mount-only by design; the parent remounts the field to reset it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div>
      <label htmlFor={id} className="eyebrow">
        {label}
      </label>
      <input
        id={id}
        type="text"
        inputMode="decimal"
        autoComplete="off"
        className={`field numeric mt-1 text-sm ${error ? 'border-b-signal-bad' : ''}`}
        value={text}
        aria-invalid={error ? true : undefined}
        aria-describedby={error || hint ? noteId : undefined}
        onChange={(event) => {
          const next = event.target.value
          setText(next)
          const result = parseNumber(next, spec)
          if (typeof result === 'string') {
            onValidity(name, label, result)
          } else {
            onValidity(name, label, null)
            if (result !== value) onChange(result)
          }
        }}
      />
      {(error || hint) && (
        <p id={noteId} className={`mt-1.5 text-xs leading-snug ${error ? 'text-signal-bad' : 'text-ink-500'}`}>
          {error ?? hint}
        </p>
      )}
    </div>
  )
}

function Toggle({
  id,
  inputRef,
  label,
  hint,
  checked,
  onChange,
}: {
  id?: string
  inputRef?: React.Ref<HTMLInputElement>
  label: string
  hint?: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  const fallbackId = useId()
  const inputId = id ?? fallbackId
  const hintId = `${inputId}-hint`
  return (
    <div className="flex items-start gap-3">
      <input
        id={inputId}
        ref={inputRef}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        aria-describedby={hint ? hintId : undefined}
        className="mt-0.5 size-4 shrink-0 accent-sodium-500"
      />
      <div>
        <label htmlFor={inputId} className="text-sm text-ink-200">
          {label}
        </label>
        {hint && (
          <p id={hintId} className="mt-1 text-xs leading-snug text-ink-500">
            {hint}
          </p>
        )}
      </div>
    </div>
  )
}
