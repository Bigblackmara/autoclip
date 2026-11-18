import { useEffect, useState } from 'react'

import {
  api,
  type CaptionLanguage,
  type Layout,
  type Motion,
  type Pace,
  type ProviderStatus,
  type Settings as SettingsData,
  type SettingsPatch,
  type SystemStatus,
  type WatermarkCorner,
  type WatermarkMode,
  type YtdlpStatus,
} from '../api'
import { ErrorNote } from '../components/ErrorNote'

const SECRET_LABELS: Record<string, string> = {
  anthropic: 'Anthropic API key',
  openai: 'OpenAI-compatible API key',
  gemini: 'Google Gemini API key',
  huggingface_token: 'HuggingFace token',
}

export function Settings() {
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [ytdlp, setYtdlp] = useState<YtdlpStatus | null>(null)
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [saved, setSaved] = useState(false)

  const reload = () => {
    void api.getSettings().then(setSettings).catch((e) => setError(e as Error))
    void api.providerStatus().then(setProviders).catch(() => undefined)
    void api.system().then(setSystem).catch(() => undefined)
    void api.ytdlpStatus().then(setYtdlp).catch(() => undefined)
  }

  const checkYtdlp = async () => {
    setChecking(true)
    setError(null)
    try {
      setYtdlp(await api.ytdlpCheck())
    } catch (err) {
      setError(err as Error)
    } finally {
      setChecking(false)
    }
  }

  useEffect(reload, [])

  /**
   * Send only what changed. Spreading the whole section from this render's
   * `settings` raced: blur the Language field and tick diarization quickly,
   * and the second PUT carried the old language and reverted the first.
   * Resolves false when the server refused, so fields can show the saved
   * value again instead of the rejected one.
   */
  const patch = async (update: SettingsPatch): Promise<boolean> => {
    setError(null)
    try {
      setSettings(await api.putSettings(update))
      setSaved(true)
      setTimeout(() => setSaved(false), 1600)
      return true
    } catch (err) {
      setError(err as Error)
      return false
    }
  }

  if (!settings) return <p className="pt-24 text-sm text-ink-500">Loading…</p>

  return (
    <div className="max-w-4xl pt-14">
      <div className="rise flex items-baseline justify-between border-b border-ink-800 pb-5">
        <h1 className="font-display text-[clamp(2rem,4vw,3rem)] leading-none text-ink-100">
          Settings
        </h1>
        <span
          className="text-xs text-signal-good transition-opacity duration-300"
          style={{ opacity: saved ? 1 : 0 }}
        >
          saved
        </span>
      </div>

      {error && (
        <div className="mt-6">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {settings.insecure_secret_storage && (
        <p className="mt-6 border-l-2 border-sodium-600 pl-4 text-sm leading-relaxed text-ink-300">
          No OS keyring is available on this machine, so API keys are stored in plain text
          in <code className="text-ink-200">config.json</code>. On headless Linux, installing
          a Secret Service provider or <code className="text-ink-200">keyrings.alt</code>{' '}
          fixes this.
        </p>
      )}

      <Section title="AI provider" note="Which model picks the clips.">
        <div className="space-y-1">
          {providers.map((provider) => (
            <button
              key={provider.name}
              onClick={() => patch({ active_provider: provider.name })}
              aria-pressed={provider.name === settings.active_provider}
              className={[
                'block w-full border-l-2 py-3 pl-3 text-left transition-colors duration-200',
                provider.name === settings.active_provider
                  ? 'border-sodium-500 bg-ink-850/60'
                  : 'border-transparent hover:border-ink-700 hover:bg-ink-850/30',
              ].join(' ')}
            >
              <div className="flex items-baseline justify-between gap-4">
                <span className="text-sm text-ink-100">{provider.name}</span>
                <span
                  className={`text-xs ${provider.available ? 'text-signal-good' : 'text-ink-500'}`}
                >
                  {provider.available ? 'reachable' : provider.detail || 'unavailable'}
                </span>
              </div>
              {provider.models.length > 0 && (
                <span className="numeric mt-1 block truncate text-xs text-ink-600">
                  {provider.models.slice(0, 6).join(' · ')}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="mt-6 grid gap-5 sm:grid-cols-2">
          <Field
            label="Model"
            value={settings.providers[settings.active_provider]?.model ?? ''}
            placeholder="model name"
            onCommit={(value) =>
              patch({ providers: { [settings.active_provider]: { model: value } } })
            }
          />
          <Field
            label="Base URL"
            hint="Point at OpenRouter, Groq, DeepSeek, or a local server."
            value={settings.providers[settings.active_provider]?.base_url ?? ''}
            placeholder="https://…"
            onCommit={(value) =>
              patch({ providers: { [settings.active_provider]: { base_url: value || null } } })
            }
          />
        </div>
      </Section>

      <Section title="Keys" note="Stored in your OS keyring. Never sent anywhere but the provider.">
        <div className="space-y-4">
          {Object.entries(SECRET_LABELS).map(([key, label]) => (
            <SecretField
              key={key}
              secretKey={key}
              label={label}
              present={settings.keys_present[key] ?? false}
              onChanged={reload}
              onError={setError}
            />
          ))}
        </div>
      </Section>

      <Section title="Transcription">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="Whisper model"
            value={settings.whisper.model}
            onChange={(value) => patch({ whisper: { model: value } })}
            options={['tiny', 'base', 'small', 'medium', 'large-v3']}
          />
          <Field
            label="Language"
            hint="Leave empty to detect automatically."
            value={settings.whisper.language}
            placeholder="auto"
            onCommit={(value) => patch({ whisper: { language: value } })}
          />
        </div>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.whisper.diarization}
            disabled={!system?.diarization_available}
            onChange={(e) =>
              patch({ whisper: { diarization: e.target.checked } })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Identify speakers
            {!system?.diarization_available && (
              <span className="mt-1 block text-xs text-ink-500">
                Needs the diarization extra:{' '}
                <code className="text-ink-300">uv pip install &apos;autoclip[diarization]&apos;</code>
              </span>
            )}
          </span>
        </label>
      </Section>

      <Section title="Clips">
        <div className="grid gap-5 sm:grid-cols-3">
          <NumberField
            label="Min length (s)"
            value={settings.clips.min_duration_s}
            onCommit={(value) => patch({ clips: { min_duration_s: value } })}
          />
          <NumberField
            label="Max length (s)"
            value={settings.clips.max_duration_s}
            onCommit={(value) => patch({ clips: { max_duration_s: value } })}
          />
          <NumberField
            label="Max clips"
            value={settings.clips.max_clips}
            integer
            min={1}
            onCommit={(value) => patch({ clips: { max_clips: value } })}
          />
        </div>
      </Section>

      <Section title="Ingest">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="YouTube cookies from"
            hint="YouTube blocks most anonymous downloads. Sign in in that browser and close it before downloading."
            value={settings.ingest.cookies_from_browser}
            onChange={(value) =>
              patch({ ingest: { cookies_from_browser: value } })
            }
            options={[
              '',
              'chrome',
              'firefox',
              'edge',
              'brave',
              'chromium',
              'opera',
              'vivaldi',
              'safari',
            ]}
            labels={{ '': 'None' }}
          />
          <NumberField
            label="Link size limit (MB)"
            value={settings.ingest.link_max_mb}
            integer
            min={1}
            onCommit={(value) => patch({ ingest: { link_max_mb: value } })}
          />
        </div>
      </Section>

      <Section
        title="Storage & updates"
        note="Old work can be cleaned up automatically, and yt-dlp kept current."
      >
        <div className="grid gap-5 sm:grid-cols-2">
          <div>
            <NumberField
              label="Delete work after (days)"
              value={settings.housekeeping.retention_days}
              integer
              min={0}
              onCommit={(value) => patch({ housekeeping: { retention_days: value } })}
            />
            <span className="mt-1.5 block text-xs leading-snug text-ink-500">
              0 keeps everything. Sources, jobs and clips older than this are deleted hourly,
              files included. Anything still processing, and GTA clips waiting to be posted,
              are kept.
            </span>
          </div>
        </div>

        <div className="mt-6">
          <Toggle
            label="Keep yt-dlp up to date (checks daily; retries a failed YouTube download once after updating)"
            checked={settings.ingest.ytdlp_auto_update}
            onChange={(checked) => patch({ ingest: { ytdlp_auto_update: checked } })}
          />
        </div>
        {ytdlp && (
          <dl className="mt-5 grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
            <Row label="yt-dlp installed" value={ytdlp.installed_version ?? 'not found'} />
            <Row label="Latest on PyPI" value={ytdlp.latest_version ?? '—'} />
            <Row
              label="Last check"
              value={ytdlp.last_check_at ? new Date(ytdlp.last_check_at).toLocaleString() : 'never'}
            />
            <Row label="Result" value={ytdlpResult(ytdlp)} />
          </dl>
        )}
        {ytdlp?.last_message && (
          <p className="mt-3 text-xs leading-relaxed text-ink-500">{ytdlp.last_message}</p>
        )}
        {ytdlp?.restart_pending && (
          <p className="mt-3 border-l-2 border-sodium-600 pl-4 text-xs leading-relaxed text-ink-300">
            yt-dlp {ytdlp.installed_version} is installed; this session still runs{' '}
            {ytdlp.loaded_version}. YouTube downloads already use the new version; restart
            AutoClip so GTA Mode and Live Watch do too.
          </p>
        )}
        <button
          type="button"
          className="btn btn-quiet mt-4 -ml-1"
          disabled={checking}
          onClick={() => void checkYtdlp()}
        >
          {checking ? 'Checking…' : 'Check for an update now'}
        </button>
      </Section>

      <Section title="Export">
        <div className="grid gap-5 sm:grid-cols-2">
          <Select
            label="Default ratio"
            value={settings.export.ratio}
            onChange={(value) => patch({ export: { ratio: value } })}
            options={['9:16', '1:1', '16:9']}
          />
          <Select
            label="Layout (9:16)"
            hint="Gameplay streams: stack the streamer's facecam above the game instead of cropping it out."
            value={settings.export.layout}
            onChange={(value) => patch({ export: { layout: value as Layout } })}
            options={['auto', 'split', 'crop']}
            labels={{
              auto: 'Auto — facecam + gameplay when a cam is found',
              split: 'Facecam + gameplay',
              crop: 'Crop — follow the speaker',
            }}
          />
          <NumberField
            label="Loudness target (LUFS)"
            value={settings.export.loudness_lufs}
            onCommit={(value) => patch({ export: { loudness_lufs: value } })}
          />
        </div>

        <label className="mt-5 flex items-start gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.prefer_hardware_encoder}
            onChange={(e) =>
              patch({ export: { prefer_hardware_encoder: e.target.checked } })
            }
            className="mt-0.5 size-4 accent-sodium-500"
          />
          <span>
            Use GPU encoding when available
            {system && !system.nvenc_works && (
              <span className="mt-1 block text-xs text-ink-500">
                Not usable on this machine — exports will use the CPU encoder. Same quality,
                slower.
              </span>
            )}
          </span>
        </label>

        <label className="mt-4 flex items-center gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={settings.export.write_srt}
            onChange={(e) => patch({ export: { write_srt: e.target.checked } })}
            className="size-4 accent-sodium-500"
          />
          Also write an .srt sidecar
        </label>
      </Section>

      <Section
        title="Editing"
        note="Applied to every normal-mode export and re-export: pause cuts, zooms, headline, key words, music and logo."
      >
        <div className="grid gap-5 sm:grid-cols-3">
          <Select
            label="Cut pauses"
            hint="Cuts only on measured silences; captions are re-timed."
            value={settings.export.pace}
            onChange={(value) => patch({ export: { pace: value as Pace } })}
            options={['off', 'natural', 'fast']}
            labels={{
              off: 'Off — keep everything',
              natural: 'Natural — pauses over 0.6 s',
              fast: 'Fast — nearly every pause',
            }}
          />
          <Select
            label="Motion"
            hint="Punch-in zooms at natural pauses. Jump cuts always get a zoom step."
            value={settings.export.motion}
            onChange={(value) => patch({ export: { motion: value as Motion } })}
            options={['off', 'subtle', 'dynamic']}
            labels={{
              off: 'Off — steady framing',
              subtle: 'Subtle — every few seconds',
              dynamic: 'Dynamic — often, zoom-out on the hook',
            }}
          />
          <NumberField
            label="Frame rate (fps)"
            value={settings.export.fps}
            integer
            min={15}
            onCommit={(value) => patch({ export: { fps: value } })}
          />
          <Select
            label="Caption language"
            hint="Burned captions, the hook and the post text. Roman and English use the AI provider for scripts the rules can't handle."
            value={settings.export.caption_language}
            onChange={(value) => patch({ export: { caption_language: value as CaptionLanguage } })}
            options={['as_spoken', 'roman', 'english']}
            labels={{
              as_spoken: 'As spoken — Hindi as Hinglish',
              roman: 'Roman — every language in Latin letters',
              english: 'English — translated, timings kept',
            }}
          />
        </div>

        <div className="mt-5 flex flex-wrap gap-x-8 gap-y-3">
          <Toggle
            label="Hook headline (first 3 s, the hook picked in review)"
            checked={settings.export.hook_title}
            onChange={(checked) => patch({ export: { hook_title: checked } })}
          />
          <Toggle
            label="Flash-forward opening by default (best line first)"
            checked={settings.export.flash_forward}
            onChange={(checked) => patch({ export: { flash_forward: checked } })}
          />
          <Toggle
            label="Highlight key words in captions"
            checked={settings.export.keyword_highlight}
            onChange={(checked) => patch({ export: { keyword_highlight: checked } })}
          />
          <Toggle
            label="Two-pass loudness (lands on the target)"
            checked={settings.export.two_pass_loudnorm}
            onChange={(checked) => patch({ export: { two_pass_loudnorm: checked } })}
          />
        </div>

        <div className="mt-8 grid gap-5 sm:grid-cols-3">
          <div className="sm:col-span-3">
            <Toggle
              label="Background music"
              checked={settings.export.music_enabled}
              onChange={(checked) => patch({ export: { music_enabled: checked } })}
            />
          </div>
          <div className="sm:col-span-2">
            <Field
              label="Music track or folder"
              hint="A file, or a folder to pick one track per clip from. Looped, faded, under the voice."
              value={settings.export.music_path}
              placeholder="C:\Music\background.mp3"
              onCommit={(value) => patch({ export: { music_path: value } })}
            />
          </div>
          <NumberField
            label="Music volume (%)"
            value={settings.export.music_volume}
            integer
            min={0}
            onCommit={(value) => patch({ export: { music_volume: value } })}
          />
          <div className="sm:col-span-3">
            <Toggle
              label="Duck the music while someone talks"
              checked={settings.export.music_duck}
              onChange={(checked) => patch({ export: { music_duck: checked } })}
            />
          </div>
        </div>

        <div className="mt-8 grid gap-5 sm:grid-cols-3">
          <Select
            label="Watermark"
            value={settings.export.watermark}
            onChange={(value) => patch({ export: { watermark: value as WatermarkMode } })}
            options={['off', 'text', 'image']}
            labels={{ off: 'Off', text: 'Text logo', image: 'PNG image' }}
          />
          <Select
            label="Corner"
            value={settings.export.watermark_corner}
            onChange={(value) =>
              patch({ export: { watermark_corner: value as WatermarkCorner } })
            }
            options={['top_left', 'top_right', 'bottom_left', 'bottom_right']}
            labels={{
              top_left: 'Top left',
              top_right: 'Top right',
              bottom_left: 'Bottom left',
              bottom_right: 'Bottom right',
            }}
          />
          <NumberField
            label="Opacity (0.05–1)"
            value={settings.export.watermark_opacity}
            min={0.05}
            onCommit={(value) => patch({ export: { watermark_opacity: value } })}
          />
          {settings.export.watermark === 'text' && (
            <div className="sm:col-span-3">
              <Field
                label="Logo text"
                value={settings.export.watermark_text}
                placeholder="MY CHANNEL"
                onCommit={(value) => patch({ export: { watermark_text: value } })}
              />
            </div>
          )}
          {settings.export.watermark === 'image' && (
            <div className="sm:col-span-3">
              <Field
                label="Logo PNG"
                value={settings.export.watermark_image}
                placeholder="C:\Brand\logo.png"
                onCommit={(value) => patch({ export: { watermark_image: value } })}
              />
            </div>
          )}
        </div>
      </Section>

      {system && (
        <Section title="This machine">
          <dl className="grid gap-x-8 gap-y-3 text-sm sm:grid-cols-2">
            <Row label="Platform" value={system.platform} />
            <Row label="Python" value={system.python_version} />
            <Row label="ffmpeg" value={system.ffmpeg_version ?? 'not found'} />
            <Row label="Acceleration" value={system.accel.toUpperCase()} />
            <Row label="Device" value={system.gpu_name ?? '—'} />
            <Row label="Whisper compute" value={system.compute_type} />
            <Row label="GPU encode" value={system.nvenc_works ? 'available' : 'unavailable'} />
            <Row label="Captions" value={system.has_libass ? 'libass present' : 'libass missing'} />
          </dl>
        </Section>
      )}
    </div>
  )
}

function ytdlpResult(status: YtdlpStatus): string {
  switch (status.last_result) {
    case 'up_to_date':
      return 'up to date'
    case 'updated':
      return 'updated'
    case 'failed':
      return 'check failed'
    default:
      return status.auto_update ? 'not checked yet' : 'auto-update off'
  }
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
        <h2 className="eyebrow">{title}</h2>
        {note && <p className="mt-1 text-xs text-ink-500">{note}</p>}
      </div>
      <div className="mt-5">{children}</div>
    </section>
  )
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="flex items-center gap-3 text-sm text-ink-200">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="size-4 accent-sodium-500"
      />
      {label}
    </label>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-ink-850 pb-2">
      <dt className="text-ink-500">{label}</dt>
      <dd className="numeric truncate text-right text-ink-200">{value}</dd>
    </div>
  )
}

/** Commits on blur rather than per keystroke, so a PUT isn't fired per letter. */
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
  onCommit: (value: string) => Promise<boolean>
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])

  const commit = async () => {
    if (draft === value) return
    if (!(await onCommit(draft))) setDraft(value)
  }

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        className="field mt-1 text-sm"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => void commit()}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        spellCheck={false}
      />
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

function NumberField({
  label,
  value,
  integer = false,
  min,
  onCommit,
}: {
  label: string
  value: number
  integer?: boolean
  min?: number
  onCommit: (value: number) => Promise<boolean>
}) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => setDraft(String(value)), [value])

  // Number('') is 0, not NaN: clearing "Max clips" used to save 0 clips. A
  // blank, malformed or refused value snaps back to the saved one.
  const commit = async () => {
    const parsed = Number(draft)
    const valid =
      draft.trim() !== '' &&
      Number.isFinite(parsed) &&
      (!integer || Number.isInteger(parsed)) &&
      (min === undefined || parsed >= min)
    if (!valid) {
      setDraft(String(value))
      return
    }
    if (parsed !== value && !(await onCommit(parsed))) setDraft(String(value))
  }

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric mt-1 text-sm"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        min={min}
        step={integer ? 1 : 'any'}
        onBlur={() => void commit()}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
    </label>
  )
}

function Select({
  label,
  hint,
  value,
  onChange,
  options,
  labels = {},
}: {
  label: string
  hint?: string
  value: string
  onChange: (value: string) => void
  options: string[]
  labels?: Record<string, string>
}) {
  // A value set outside this page (a local Whisper model path, a browser
  // not listed here) must still show as selected; otherwise the select
  // silently displays its first option as if that were saved.
  const choices = options.includes(value) ? options : [...options, value]

  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <select
        className="field mt-1 cursor-pointer text-sm"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {choices.map((option) => (
          <option key={option} value={option} className="bg-ink-850">
            {labels[option] ?? option}
          </option>
        ))}
      </select>
      {hint && <span className="mt-1.5 block text-xs leading-snug text-ink-500">{hint}</span>}
    </label>
  )
}

function SecretField({
  secretKey,
  label,
  present,
  onChanged,
  onError,
}: {
  secretKey: string
  label: string
  present: boolean
  onChanged: () => void
  onError: (error: Error) => void
}) {
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)

  const save = async () => {
    if (!value.trim()) return
    setBusy(true)
    try {
      await api.putSecret(secretKey, value.trim())
      setValue('')
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  const remove = async () => {
    setBusy(true)
    try {
      await api.deleteSecret(secretKey)
      onChanged()
    } catch (err) {
      onError(err as Error)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="min-w-56 flex-1">
        <span className="eyebrow">
          {label}
          {present && <span className="ml-2 text-signal-good">set</span>}
        </span>
        <input
          type="password"
          className="field mt-1 text-sm"
          value={value}
          placeholder={present ? '••••••••••••' : 'paste to add'}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          autoComplete="off"
        />
      </label>
      <button onClick={save} disabled={!value.trim() || busy} className="btn btn-ghost">
        Save
      </button>
      {present && (
        <button onClick={remove} disabled={busy} className="btn btn-quiet">
          Remove
        </button>
      )}
    </div>
  )
}
