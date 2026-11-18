import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import {
  autopilotApi,
  errorText,
  type AutopilotInput,
  type AutopilotSource,
  type Platform,
} from '../publishing/api'

const EMPTY: AutopilotInput = {
  name: '',
  sources: [{ kind: 'channel', value: '', country: '', language: '' }],
  filters: { min_views: 10_000, min_minutes: 5, max_minutes: 180, max_age_days: 14 },
  clips_per_video: 2,
  slots: ['12:00', '18:00'],
  timezone: '',
  platforms: ['youtube'],
  approve_before_posting: true,
  heatmap_wait_hours: 24,
  rights_confirmed: false,
}

function NumberField({
  label,
  value,
  onChange,
  min,
  step,
  hint,
}: {
  label: string
  value: number
  onChange: (n: number) => void
  min?: number
  step?: number
  hint?: string
}) {
  return (
    <label className="block">
      <span className="eyebrow">{label}</span>
      <input
        type="number"
        className="field numeric"
        min={min}
        step={step}
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <span className="mt-1 block text-xs text-ink-500">{hint}</span>}
    </label>
  )
}

/** Create or edit an autopilot: sources, filters, posting times, platforms. */
export function AutopilotEditor() {
  const { autopilotId } = useParams()
  const navigate = useNavigate()
  const [form, setForm] = useState<AutopilotInput>(EMPTY)
  const [slotsText, setSlotsText] = useState(EMPTY.slots.join(', '))
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loaded, setLoaded] = useState(!autopilotId)

  useEffect(() => {
    if (!autopilotId) return
    autopilotApi
      .get(autopilotId)
      .then((ap) => {
        const input: AutopilotInput = {
          name: ap.name,
          sources: ap.sources,
          filters: ap.filters,
          clips_per_video: ap.clips_per_video,
          slots: ap.slots,
          timezone: ap.timezone,
          platforms: ap.platforms,
          approve_before_posting: ap.approve_before_posting,
          heatmap_wait_hours: ap.heatmap_wait_hours,
          rights_confirmed: ap.rights_confirmed,
        }
        setForm(input)
        setSlotsText(ap.slots.join(', '))
        setLoaded(true)
      })
      .catch((err) => setError(errorText(err)))
  }, [autopilotId])

  const set = <K extends keyof AutopilotInput>(key: K, value: AutopilotInput[K]) =>
    setForm((f) => ({ ...f, [key]: value }))
  const setFilter = (key: keyof AutopilotInput['filters'], value: number) =>
    setForm((f) => ({ ...f, filters: { ...f.filters, [key]: value } }))
  const setSource = (i: number, change: Partial<AutopilotSource>) =>
    setForm((f) => ({
      ...f,
      sources: f.sources.map((s, j) => (j === i ? { ...s, ...change } : s)),
    }))
  const togglePlatform = (p: Platform) =>
    set(
      'platforms',
      form.platforms.includes(p) ? form.platforms.filter((x) => x !== p) : [...form.platforms, p],
    )

  async function save() {
    setBusy(true)
    setError('')
    const input: AutopilotInput = {
      ...form,
      sources: form.sources.filter((s) => s.value.trim()),
      slots: slotsText
        .split(/[,\s]+/)
        .map((s) => s.trim())
        .filter(Boolean),
    }
    try {
      const saved = autopilotId
        ? await autopilotApi.update(autopilotId, input)
        : await autopilotApi.create(input)
      navigate(`/autopilot/${saved.id}`)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  if (!loaded) return <p className="pt-10 text-sm text-ink-400">{error || 'Loading…'}</p>

  return (
    <div className="rise max-w-3xl space-y-10 pt-10">
      <header>
        <p className="eyebrow">
          <Link to="/autopilot" className="hover:text-ink-200">
            Autopilot
          </Link>
        </p>
        <h1 className="font-display text-5xl text-ink-100">
          {autopilotId ? 'Edit autopilot' : 'New autopilot'}
        </h1>
      </header>

      <label className="block">
        <span className="eyebrow">Name</span>
        <input
          className="field"
          maxLength={80}
          value={form.name}
          placeholder="Podcast highlights"
          onChange={(e) => set('name', e.target.value)}
        />
      </label>

      <section className="space-y-4">
        <h2 className="eyebrow border-b border-ink-800 pb-2">Where to look</h2>
        {form.sources.map((source, i) => (
          <div key={i} className="grid gap-3 sm:grid-cols-[8rem_1fr_5rem_5rem_auto] sm:items-end">
            <select
              className="field"
              value={source.kind}
              onChange={(e) => setSource(i, { kind: e.target.value as AutopilotSource['kind'] })}
            >
              <option value="channel">Channel</option>
              <option value="topic">Topic search</option>
            </select>
            <input
              className="field"
              value={source.value}
              placeholder={source.kind === 'channel' ? '@handle or channel link' : 'e.g. chess blunders'}
              onChange={(e) => setSource(i, { value: e.target.value })}
            />
            <input
              className="field uppercase"
              maxLength={2}
              value={source.country}
              placeholder="US"
              title="Country (optional)"
              disabled={source.kind === 'channel'}
              onChange={(e) => setSource(i, { country: e.target.value })}
            />
            <input
              className="field lowercase"
              maxLength={3}
              value={source.language}
              placeholder="en"
              title="Spoken language (optional)"
              onChange={(e) => setSource(i, { language: e.target.value })}
            />
            <button
              type="button"
              className="btn btn-quiet"
              disabled={form.sources.length === 1}
              onClick={() => set('sources', form.sources.filter((_, j) => j !== i))}
            >
              Remove
            </button>
          </div>
        ))}
        <button
          type="button"
          className="btn btn-ghost"
          onClick={() =>
            set('sources', [...form.sources, { kind: 'topic', value: '', country: '', language: '' }])
          }
        >
          Add source
        </button>
        <p className="text-xs text-ink-500">
          Country and language are optional. A language keeps only videos YouTube reports in that
          language; a country alone implies its main language.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="eyebrow border-b border-ink-800 pb-2">Which videos</h2>
        <div className="grid gap-6 sm:grid-cols-2">
          <NumberField
            label="Minimum views"
            value={form.filters.min_views}
            min={0}
            step={1000}
            onChange={(n) => setFilter('min_views', n)}
          />
          <NumberField
            label="Not older than (days)"
            value={form.filters.max_age_days}
            min={1}
            onChange={(n) => setFilter('max_age_days', n)}
          />
          <NumberField
            label="Minimum length (minutes)"
            value={form.filters.min_minutes}
            min={0}
            onChange={(n) => setFilter('min_minutes', n)}
          />
          <NumberField
            label="Maximum length (minutes)"
            value={form.filters.max_minutes}
            min={1}
            onChange={(n) => setFilter('max_minutes', n)}
          />
          <NumberField
            label="Wait for “Most replayed” (hours)"
            value={form.heatmap_wait_hours}
            min={0}
            onChange={(n) => set('heatmap_wait_hours', n)}
            hint="New videos wait for YouTube's rewatch graph; after this the AI picks from audio and comments."
          />
          <NumberField
            label="Clips per video"
            value={form.clips_per_video}
            min={1}
            onChange={(n) => set('clips_per_video', n)}
          />
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="eyebrow border-b border-ink-800 pb-2">When and where to post</h2>
        <div className="grid gap-6 sm:grid-cols-2">
          <label className="block">
            <span className="eyebrow">Daily posting times</span>
            <input
              className="field numeric"
              value={slotsText}
              placeholder="12:00, 18:00"
              onChange={(e) => setSlotsText(e.target.value)}
            />
          </label>
          <label className="block">
            <span className="eyebrow">Time zone</span>
            <input
              className="field"
              value={form.timezone}
              placeholder="This PC's time (or e.g. Europe/Berlin)"
              onChange={(e) => set('timezone', e.target.value)}
            />
          </label>
        </div>
        <div className="flex flex-wrap gap-6 text-sm text-ink-200">
          {(['youtube', 'instagram'] as Platform[]).map((p) => (
            <label key={p} className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={form.platforms.includes(p)}
                onChange={() => togglePlatform(p)}
              />
              {p === 'youtube' ? 'YouTube Shorts' : 'Instagram Reels'}
            </label>
          ))}
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={form.approve_before_posting}
              onChange={(e) => set('approve_before_posting', e.target.checked)}
            />
            Let me approve each clip before it is booked
          </label>
        </div>
      </section>

      <label className="flex items-start gap-3 text-sm text-ink-200">
        <input
          type="checkbox"
          className="mt-1"
          checked={form.rights_confirmed}
          onChange={(e) => set('rights_confirmed', e.target.checked)}
        />
        I have the rights (or the creators' permission) to clip and repost videos from these
        sources.
      </label>

      {error && <p className="whitespace-pre-line text-sm text-signal-bad">{error}</p>}
      <div className="flex gap-3">
        <button type="button" className="btn btn-primary" disabled={busy} onClick={save}>
          {busy ? 'Saving…' : autopilotId ? 'Save' : 'Create autopilot'}
        </button>
        <Link to={autopilotId ? `/autopilot/${autopilotId}` : '/autopilot'} className="btn btn-ghost">
          Cancel
        </Link>
      </div>
    </div>
  )
}
