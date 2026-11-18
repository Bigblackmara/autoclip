import { useEffect, useId, useState, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'

import {
  api,
  type CaptionLanguage,
  type Framing,
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
import {
  Badge,
  Button,
  ButtonLink,
  Callout,
  CaptionsIcon,
  Card,
  CheckIcon,
  CloudIcon,
  CpuIcon,
  Field,
  FilmIcon,
  GamepadIcon,
  HardDriveIcon,
  Input,
  KeyIcon,
  KeyValue,
  MusicIcon,
  PageHeader,
  PageSkeleton,
  RadioIcon,
  RefreshIcon,
  Select,
  SendIcon,
  Skeleton,
  SparklesIcon,
  StampIcon,
  Tabs,
  Toggle,
  UsersIcon,
  cx,
  tabPanelProps,
  useToast,
  type TabItem,
} from '../ui'

const SECRET_LABELS: Record<string, string> = {
  anthropic: 'Anthropic API key',
  openai: 'OpenAI-compatible API key',
  gemini: 'Google Gemini API key',
  huggingface_token: 'HuggingFace token',
}

type Section = 'general' | 'clips' | 'captions' | 'audio' | 'branding' | 'sources' | 'storage' | 'accounts' | 'system'

const SECTIONS: readonly (TabItem<Section> & { description: string })[] = [
  { key: 'general', label: 'General & AI', icon: SparklesIcon, description: 'Which AI picks your clips, your API keys, and how speech is transcribed.' },
  { key: 'clips', label: 'Clips & video', icon: FilmIcon, description: 'Clip lengths, shape, framing and how videos are rendered.' },
  { key: 'captions', label: 'Captions & editing', icon: CaptionsIcon, description: 'Caption language, the hook headline, pause cuts and zooms.' },
  { key: 'audio', label: 'Audio & music', icon: MusicIcon, description: 'Loudness and background music.' },
  { key: 'branding', label: 'Branding', icon: StampIcon, description: 'Put your logo or channel name on every clip.' },
  { key: 'sources', label: 'Sources & downloads', icon: CloudIcon, description: 'How videos are downloaded from YouTube and file links.' },
  { key: 'storage', label: 'Storage', icon: HardDriveIcon, description: 'Clean up old work automatically.' },
  { key: 'accounts', label: 'Accounts & posting', icon: UsersIcon, description: 'Where to connect YouTube, Instagram and Twitch.' },
  { key: 'system', label: 'This machine', icon: CpuIcon, description: 'What AutoClip found on this computer.' },
]

export function Settings() {
  const [params, setParams] = useSearchParams()
  const section: Section = SECTIONS.find((s) => s.key === params.get('tab'))?.key ?? 'general'
  const [settings, setSettings] = useState<SettingsData | null>(null)
  const [providers, setProviders] = useState<ProviderStatus[]>([])
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [ytdlp, setYtdlp] = useState<YtdlpStatus | null>(null)
  const [checking, setChecking] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const toast = useToast()
  const tabsId = useId()

  const show = (key: Section) =>
    setParams(
      (p) => {
        const next = new URLSearchParams(p)
        if (key === 'general') next.delete('tab')
        else next.set('tab', key)
        return next
      },
      { replace: true },
    )

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
      toast.success('Saved')
      return true
    } catch (err) {
      setError(err as Error)
      return false
    }
  }

  if (!settings) {
    return error ? (
      <div className="max-w-3xl">
        <PageHeader title="Settings" />
        <ErrorNote error={error} />
      </div>
    ) : (
      <PageSkeleton />
    )
  }

  const current = SECTIONS.find((s) => s.key === section)!
  const activeProvider = providers.find((p) => p.name === settings.active_provider)

  return (
    <div>
      <PageHeader title="Settings" description="Changes save as you go. They apply to new clips and exports." />

      <div className="grid items-start gap-6 lg:grid-cols-[14rem_minmax(0,1fr)] lg:gap-8">
        <div className="min-w-0 lg:sticky lg:top-6">
          <Tabs label="Settings sections" idBase={tabsId} variant="side" items={SECTIONS} value={section} onChange={show} />
        </div>

        <div className="min-w-0 max-w-4xl space-y-6" {...tabPanelProps(tabsId, section)}>
          <div>
            <h2 className="text-lg font-semibold text-fg">{current.label}</h2>
            <p className="mt-0.5 text-sm text-fg-muted">{current.description}</p>
          </div>

          {error && <ErrorNote error={error} onDismiss={() => setError(null)} />}

          {settings.insecure_secret_storage && (section === 'general' || section === 'system') && (
            <Callout tone="warn" title="API keys are stored in plain text">
              No OS keyring is available on this machine, so keys are saved in{' '}
              <code className="font-mono text-fg">config.json</code>. On headless Linux, installing a Secret Service
              provider or <code className="font-mono text-fg">keyrings.alt</code> fixes this.
            </Callout>
          )}

          {section === 'general' && (
            <>
              <Card title="AI provider" description="Which AI model picks the clips. Pick one that shows “Reachable”.">
                <div className="grid gap-2 sm:grid-cols-2">
                  {providers.length === 0 &&
                    [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-[4.5rem] rounded-xl" />)}
                  {providers.map((provider) => {
                    const active = provider.name === settings.active_provider
                    return (
                      <button
                        key={provider.name}
                        onClick={() => patch({ active_provider: provider.name })}
                        aria-pressed={active}
                        className={cx(
                          'rounded-xl border px-4 py-3 text-left transition-colors duration-150',
                          active ? 'border-accent bg-accent-soft ring-1 ring-accent' : 'border-line hover:border-line-strong hover:bg-subtle',
                        )}
                      >
                        <div className="flex items-center justify-between gap-3">
                          <span className="flex items-center gap-2 text-sm font-semibold text-fg">
                            {active && <CheckIcon size={15} className="text-accent-text" />}
                            {provider.name}
                          </span>
                          <Badge tone={provider.available ? 'success' : 'neutral'} dot>
                            {provider.available ? 'Reachable' : 'Unavailable'}
                          </Badge>
                        </div>
                        {!provider.available && provider.detail && (
                          <span className="mt-1 block text-xs leading-snug text-fg-muted">{provider.detail}</span>
                        )}
                        {provider.models.length > 0 && (
                          <span className="numeric mt-1 block truncate text-xs text-fg-subtle">
                            {provider.models.slice(0, 6).join(' · ')}
                          </span>
                        )}
                      </button>
                    )
                  })}
                </div>

                <div className="mt-6 grid gap-5 sm:grid-cols-2">
                  <TextField
                    label={`Model${activeProvider ? ` for ${activeProvider.name}` : ''}`}
                    value={settings.providers[settings.active_provider]?.model ?? ''}
                    placeholder="model name"
                    onCommit={(value) => patch({ providers: { [settings.active_provider]: { model: value } } })}
                  />
                  <TextField
                    label="Base URL"
                    optional
                    hint="Point at OpenRouter, Groq, DeepSeek, or a local server."
                    value={settings.providers[settings.active_provider]?.base_url ?? ''}
                    placeholder="https://…"
                    onCommit={(value) =>
                      patch({ providers: { [settings.active_provider]: { base_url: value || null } } })
                    }
                  />
                </div>
              </Card>

              <Card
                title="API keys"
                icon={KeyIcon}
                description="Stored in your computer's keyring. Only ever sent to that provider."
              >
                <div className="space-y-5">
                  {Object.entries(SECRET_LABELS).map(([key, label]) => (
                    <SecretField
                      key={key}
                      secretKey={key}
                      label={label}
                      present={settings.keys_present[key] ?? false}
                      onChanged={() => {
                        reload()
                        toast.success('Key updated')
                      }}
                      onError={setError}
                    />
                  ))}
                </div>
              </Card>

              <Card title="Transcription" description="Turns speech into words with timing, for captions and clip picking.">
                <div className="grid gap-5 sm:grid-cols-2">
                  <SelectField
                    label="Whisper model"
                    hint="Bigger is slower but more accurate."
                    value={settings.whisper.model}
                    onChange={(value) => patch({ whisper: { model: value } })}
                    options={['tiny', 'base', 'small', 'medium', 'large-v3']}
                  />
                  <TextField
                    label="Language"
                    hint="Leave empty to detect automatically."
                    value={settings.whisper.language}
                    placeholder="auto"
                    onCommit={(value) => patch({ whisper: { language: value } })}
                  />
                </div>
                <Toggle
                  className="mt-5"
                  label="Identify speakers"
                  checked={settings.whisper.diarization}
                  disabled={!system?.diarization_available}
                  onChange={(checked) => patch({ whisper: { diarization: checked } })}
                  hint={
                    !system?.diarization_available ? (
                      <>
                        Needs the diarization extra:{' '}
                        <code className="font-mono text-fg">uv pip install &apos;autoclip[diarization]&apos;</code>
                      </>
                    ) : (
                      'Label who is talking, and cut to them.'
                    )
                  }
                />
              </Card>
            </>
          )}

          {section === 'clips' && (
            <>
              <Card title="Clip length" description="The defaults for every new video. You can change them per video under Advanced options.">
                <div className="grid gap-5 sm:grid-cols-3">
                  <NumberField
                    label="Shortest clip (seconds)"
                    value={settings.clips.min_duration_s}
                    onCommit={(value) => patch({ clips: { min_duration_s: value } })}
                  />
                  <NumberField
                    label="Longest clip (seconds)"
                    value={settings.clips.max_duration_s}
                    onCommit={(value) => patch({ clips: { max_duration_s: value } })}
                  />
                  <NumberField
                    label="Most clips per video"
                    value={settings.clips.max_clips}
                    integer
                    min={1}
                    onCommit={(value) => patch({ clips: { max_clips: value } })}
                  />
                </div>
              </Card>

              <Card title="Shape & framing" description="How the picture is fitted into the clip.">
                <div className="grid gap-5 sm:grid-cols-2">
                  <SelectField
                    label="Default shape"
                    value={settings.export.ratio}
                    onChange={(value) => patch({ export: { ratio: value } })}
                    options={['9:16', '1:1', '16:9']}
                    labels={{ '9:16': '9:16 — vertical (Shorts, Reels, TikTok)', '1:1': '1:1 — square', '16:9': '16:9 — wide' }}
                  />
                  <SelectField
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
                  <SelectField
                    label="Framing"
                    hint="Podcasts: two far-apart speakers get their own halves. Screen and game footage with no face: the whole frame over a blurred copy."
                    value={settings.export.framing}
                    onChange={(value) => patch({ export: { framing: value as Framing } })}
                    options={['auto', 'fill', 'split', 'fit']}
                    labels={{
                      auto: 'Auto — split two speakers, fit screens',
                      fill: 'Fill — always crop',
                      split: 'Split speakers whenever two faces',
                      fit: 'Fit — whole frame, blurred fill',
                    }}
                  />
                  <NumberField
                    label="Frame rate (fps)"
                    value={settings.export.fps}
                    integer
                    min={15}
                    onCommit={(value) => patch({ export: { fps: value } })}
                  />
                </div>
              </Card>

              <Card title="Rendering">
                <div className="space-y-4">
                  <Toggle
                    label="Use GPU encoding when available"
                    checked={settings.export.prefer_hardware_encoder}
                    onChange={(checked) => patch({ export: { prefer_hardware_encoder: checked } })}
                    hint={
                      system && !system.nvenc_works
                        ? 'Not usable on this machine — exports use the CPU encoder. Same quality, slower.'
                        : 'Faster exports on NVIDIA graphics cards.'
                    }
                  />
                  <Toggle
                    label="Also save an .srt subtitle file"
                    hint="Next to each exported clip, for platforms that take separate captions."
                    checked={settings.export.write_srt}
                    onChange={(checked) => patch({ export: { write_srt: checked } })}
                  />
                </div>
              </Card>
            </>
          )}

          {section === 'captions' && (
            <>
              <Card title="Captions" description="Applied to every normal-mode export and re-export.">
                <div className="grid gap-5 sm:grid-cols-2">
                  <SelectField
                    label="Caption language"
                    hint="For burned-in captions, the hook and the post text. Roman and English use the AI for scripts the rules can't handle."
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
                <div className="mt-5 space-y-4">
                  <Toggle
                    label="Highlight key words in captions"
                    checked={settings.export.keyword_highlight}
                    onChange={(checked) => patch({ export: { keyword_highlight: checked } })}
                  />
                  <Toggle
                    label="Hook headline"
                    hint="Shows the hook picked in review for the first 3 seconds."
                    checked={settings.export.hook_title}
                    onChange={(checked) => patch({ export: { hook_title: checked } })}
                  />
                  <Toggle
                    label="Flash-forward opening by default"
                    hint="Starts the clip with its best line, then plays from the top."
                    checked={settings.export.flash_forward}
                    onChange={(checked) => patch({ export: { flash_forward: checked } })}
                  />
                </div>
              </Card>

              <Card title="Pacing & motion">
                <div className="grid gap-5 sm:grid-cols-2">
                  <SelectField
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
                  <SelectField
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
                </div>
              </Card>
            </>
          )}

          {section === 'audio' && (
            <>
              <Card title="Loudness" description="Makes every clip about as loud as the platforms expect.">
                <div className="grid gap-5 sm:grid-cols-2">
                  <NumberField
                    label="Loudness target (LUFS)"
                    hint="-14 suits YouTube, Instagram and TikTok."
                    value={settings.export.loudness_lufs}
                    onCommit={(value) => patch({ export: { loudness_lufs: value } })}
                  />
                </div>
                <Toggle
                  className="mt-5"
                  label="Two-pass loudness"
                  hint="Lands exactly on the target. Slightly slower."
                  checked={settings.export.two_pass_loudnorm}
                  onChange={(checked) => patch({ export: { two_pass_loudnorm: checked } })}
                />
              </Card>

              <Card title="Background music" icon={MusicIcon}>
                <Toggle
                  label="Add background music"
                  hint="Looped and faded under the voice."
                  checked={settings.export.music_enabled}
                  onChange={(checked) => patch({ export: { music_enabled: checked } })}
                />
                <div className={cx('mt-5 grid gap-5 sm:grid-cols-3', !settings.export.music_enabled && 'opacity-60')}>
                  <div className="sm:col-span-2">
                    <TextField
                      label="Music track or folder"
                      hint="A file, or a folder to pick one track per clip from."
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
                </div>
                <Toggle
                  className="mt-5"
                  label="Lower the music while someone talks"
                  checked={settings.export.music_duck}
                  onChange={(checked) => patch({ export: { music_duck: checked } })}
                />
              </Card>
            </>
          )}

          {section === 'branding' && (
            <Card title="Watermark" icon={StampIcon} description="Your logo or channel name in a corner of every clip.">
              <div className="grid gap-5 sm:grid-cols-3">
                <SelectField
                  label="Watermark"
                  value={settings.export.watermark}
                  onChange={(value) => patch({ export: { watermark: value as WatermarkMode } })}
                  options={['off', 'text', 'image']}
                  labels={{ off: 'Off', text: 'Text logo', image: 'PNG image' }}
                />
                <SelectField
                  label="Corner"
                  value={settings.export.watermark_corner}
                  onChange={(value) => patch({ export: { watermark_corner: value as WatermarkCorner } })}
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
                    <TextField
                      label="Logo text"
                      value={settings.export.watermark_text}
                      placeholder="MY CHANNEL"
                      onCommit={(value) => patch({ export: { watermark_text: value } })}
                    />
                  </div>
                )}
                {settings.export.watermark === 'image' && (
                  <div className="sm:col-span-3">
                    <TextField
                      label="Logo PNG"
                      hint="Full path to a PNG file on this computer."
                      value={settings.export.watermark_image}
                      placeholder="C:\Brand\logo.png"
                      onCommit={(value) => patch({ export: { watermark_image: value } })}
                    />
                  </div>
                )}
              </div>
              <p className="mt-5 text-xs text-fg-muted">GTA Mode has its own logo settings under GTA Mode → Settings.</p>
            </Card>
          )}

          {section === 'sources' && (
            <>
              <Card title="Downloads" description="How AutoClip fetches videos from links.">
                <div className="grid gap-5 sm:grid-cols-2">
                  <SelectField
                    label="YouTube cookies from"
                    hint="YouTube blocks most anonymous downloads. Sign in in that browser and close it before downloading."
                    value={settings.ingest.cookies_from_browser}
                    onChange={(value) => patch({ ingest: { cookies_from_browser: value } })}
                    options={['', 'chrome', 'firefox', 'edge', 'brave', 'chromium', 'opera', 'vivaldi', 'safari']}
                    labels={{ '': 'None' }}
                  />
                  <NumberField
                    label="Link size limit (MB)"
                    hint="Bigger files from Drive, Dropbox or direct links are refused."
                    value={settings.ingest.link_max_mb}
                    integer
                    min={1}
                    onCommit={(value) => patch({ ingest: { link_max_mb: value } })}
                  />
                </div>
              </Card>

              <Card
                title="yt-dlp (the YouTube downloader)"
                description="YouTube changes often; keeping yt-dlp current keeps downloads working."
                actions={
                  <Button size="sm" icon={RefreshIcon} loading={checking} onClick={() => void checkYtdlp()}>
                    {checking ? 'Checking…' : 'Check for an update'}
                  </Button>
                }
              >
                <Toggle
                  label="Keep yt-dlp up to date"
                  hint="Checks daily, and retries a failed YouTube download once after updating."
                  checked={settings.ingest.ytdlp_auto_update}
                  onChange={(checked) => patch({ ingest: { ytdlp_auto_update: checked } })}
                />
                {ytdlp && (
                  <KeyValue
                    className="mt-5"
                    items={[
                      { label: 'Installed', value: ytdlp.installed_version ?? 'not found' },
                      { label: 'Latest available', value: ytdlp.latest_version ?? '—' },
                      {
                        label: 'Last check',
                        value: ytdlp.last_check_at ? new Date(ytdlp.last_check_at).toLocaleString() : 'never',
                      },
                      { label: 'Result', value: ytdlpResult(ytdlp) },
                    ]}
                  />
                )}
                {ytdlp?.last_message && <p className="mt-3 text-xs leading-relaxed text-fg-muted">{ytdlp.last_message}</p>}
                {ytdlp?.restart_pending && (
                  <Callout tone="warn" className="mt-4">
                    yt-dlp {ytdlp.installed_version} is installed; this session still runs {ytdlp.loaded_version}.
                    YouTube downloads already use the new version; restart AutoClip so GTA Mode and Live Watch do
                    too.
                  </Callout>
                )}
              </Card>
            </>
          )}

          {section === 'storage' && (
            <Card title="Automatic clean-up" icon={HardDriveIcon}>
              <div className="grid gap-5 sm:grid-cols-2">
                <NumberField
                  label="Delete work after (days)"
                  hint="0 keeps everything. Sources, jobs and clips older than this are deleted hourly, files included. Anything still processing, and GTA clips waiting to be posted, are kept."
                  value={settings.housekeeping.retention_days}
                  integer
                  min={0}
                  onCommit={(value) => patch({ housekeeping: { retention_days: value } })}
                />
              </div>
            </Card>
          )}

          {section === 'accounts' && (
            <div className="grid gap-4 sm:grid-cols-2">
              <LinkCard
                icon={SendIcon}
                title="YouTube & Instagram posting"
                text="Connect your channel and Instagram, set posting times, see the upload history."
                to="/posting"
                cta="Open Posting & accounts"
              />
              <LinkCard
                icon={RadioIcon}
                title="Live Watch uploads"
                text="Rights notice and the accounts Live Watch posts its live clips to."
                to="/live?tab=settings"
                cta="Open Live Watch settings"
              />
              <LinkCard
                icon={GamepadIcon}
                title="Twitch viewer clips"
                text="Optional Twitch app credentials, used by GTA Mode and Explore's Twitch search."
                to="/gta?tab=settings"
                cta="Open GTA Mode settings"
              />
              <LinkCard
                icon={KeyIcon}
                title="AI provider keys"
                text="Anthropic, OpenAI-compatible, Gemini and HuggingFace keys."
                to="/settings"
                cta="Open General & AI"
              />
            </div>
          )}

          {section === 'system' &&
            (system ? (
              <Card
                title="This machine"
                actions={
                  <Badge tone={system.ready ? 'success' : 'danger'} dot>
                    {system.ready ? 'Ready' : 'Not ready'}
                  </Badge>
                }
                description={
                  system.ready
                    ? 'Everything AutoClip needs is installed.'
                    : 'Something required is missing. Run “autoclip doctor” in a terminal for details.'
                }
              >
                <KeyValue
                  items={[
                    { label: 'Platform', value: system.platform },
                    { label: 'Python', value: system.python_version },
                    { label: 'ffmpeg', value: system.ffmpeg_version ?? 'not found' },
                    { label: 'Acceleration', value: system.accel.toUpperCase() },
                    { label: 'Device', value: system.gpu_name ?? '—' },
                    { label: 'Whisper compute', value: system.compute_type },
                    { label: 'GPU encode', value: system.nvenc_works ? 'available' : 'unavailable' },
                    { label: 'Captions', value: system.has_libass ? 'libass present' : 'libass missing' },
                  ]}
                />
              </Card>
            ) : (
              <Card title="This machine">
                <p className="text-sm text-fg-muted">Couldn't read the system status.</p>
              </Card>
            ))}
        </div>
      </div>
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

function LinkCard({
  icon,
  title,
  text,
  to,
  cta,
}: {
  icon: typeof SendIcon
  title: string
  text: string
  to: string
  cta: string
}) {
  return (
    <Card title={title} icon={icon} padding="md" className="flex flex-col">
      <p className="-mt-2 flex-1 text-sm leading-relaxed text-fg-muted">{text}</p>
      <ButtonLink to={to} size="sm" className="mt-4 self-start">
        {cta}
      </ButtonLink>
    </Card>
  )
}

/** Commits on blur rather than per keystroke, so a PUT isn't fired per letter. */
function TextField({
  label,
  hint,
  value,
  placeholder,
  optional,
  onCommit,
}: {
  label: string
  hint?: ReactNode
  value: string
  placeholder?: string
  optional?: boolean
  onCommit: (value: string) => Promise<boolean>
}) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])

  const commit = async () => {
    if (draft === value) return
    if (!(await onCommit(draft))) setDraft(value)
  }

  return (
    <Field label={label} hint={hint} optional={optional}>
      <Input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => void commit()}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
        spellCheck={false}
      />
    </Field>
  )
}

function NumberField({
  label,
  hint,
  value,
  integer = false,
  min,
  onCommit,
}: {
  label: string
  hint?: ReactNode
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
    <Field label={label} hint={hint}>
      <Input
        type="number"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        min={min}
        step={integer ? 1 : 'any'}
        onBlur={() => void commit()}
        onKeyDown={(e) => e.key === 'Enter' && e.currentTarget.blur()}
      />
    </Field>
  )
}

function SelectField({
  label,
  hint,
  value,
  onChange,
  options,
  labels = {},
}: {
  label: string
  hint?: ReactNode
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
    <Field label={label} hint={hint}>
      <Select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        options={choices.map((option) => ({ value: option, label: labels[option] ?? option }))}
      />
    </Field>
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
    <div className="flex flex-wrap items-end gap-2">
      <Field
        className="min-w-56 flex-1"
        label={
          <>
            {label}
            {present && (
              <Badge tone="success" className="ml-1">
                Saved
              </Badge>
            )}
          </>
        }
      >
        <Input
          type="password"
          value={value}
          placeholder={present ? '••••••••••••' : 'Paste your key here'}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && save()}
          autoComplete="off"
        />
      </Field>
      <Button onClick={save} disabled={!value.trim() || busy}>
        Save
      </Button>
      {present && (
        <Button variant="ghost" onClick={remove} disabled={busy}>
          Remove
        </Button>
      )}
    </div>
  )
}
