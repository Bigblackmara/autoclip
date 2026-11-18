import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { autopilotApi, errorText, type AutopilotInput, type AutopilotSource, type Platform } from '../publishing/api'
import {
  Button,
  ButtonLink,
  Callout,
  Card,
  Checkbox,
  Field,
  IconButton,
  Input,
  PageHeader,
  PageSkeleton,
  PlusIcon,
  Select,
  Toggle,
  TrashIcon,
  cx,
} from '../ui'

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
    <Field label={label} hint={hint}>
      <Input
        type="number"
        min={min}
        step={step}
        value={Number.isFinite(value) ? value : ''}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </Field>
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
    set('platforms', form.platforms.includes(p) ? form.platforms.filter((x) => x !== p) : [...form.platforms, p])

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
      const saved = autopilotId ? await autopilotApi.update(autopilotId, input) : await autopilotApi.create(input)
      navigate(`/autopilot/${saved.id}`)
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }

  const back = autopilotId ? `/autopilot/${autopilotId}` : '/autopilot'

  if (!loaded) {
    return error ? (
      <div className="max-w-3xl">
        <PageHeader back={{ to: '/autopilot', label: 'All autopilots' }} title="Edit autopilot" />
        <Callout tone="danger">{error}</Callout>
      </div>
    ) : (
      <PageSkeleton cards={3} />
    )
  }

  return (
    <div className="max-w-3xl">
      <PageHeader
        back={{ to: back, label: autopilotId ? 'Back to autopilot' : 'All autopilots' }}
        title={autopilotId ? 'Edit autopilot' : 'New autopilot'}
        description="Tell it where to look, which videos are worth clipping, and when to post."
      />

      <div className="grid gap-6">
        <Card title="Name">
          <Field label="What to call it" hint="Only you see this.">
            <Input
              maxLength={80}
              value={form.name}
              placeholder="Podcast highlights"
              onChange={(e) => set('name', e.target.value)}
            />
          </Field>
        </Card>

        <Card
          title="Where to look"
          description="YouTube channels to follow, or topics to search. Country and language are optional."
        >
          <div className="space-y-4">
            {form.sources.map((source, i) => (
              <div key={i} className="rounded-xl border border-line p-3 sm:p-4">
                <div className="grid gap-3 sm:grid-cols-[9rem_minmax(0,1fr)_auto] sm:items-end">
                  <Field label="Type">
                    <Select
                      value={source.kind}
                      onChange={(e) => setSource(i, { kind: e.target.value as AutopilotSource['kind'] })}
                      options={[
                        { value: 'channel', label: 'Channel' },
                        { value: 'topic', label: 'Topic search' },
                      ]}
                    />
                  </Field>
                  <Field label={source.kind === 'channel' ? 'Channel' : 'Search words'}>
                    <Input
                      value={source.value}
                      placeholder={source.kind === 'channel' ? '@handle or channel link' : 'e.g. chess blunders'}
                      onChange={(e) => setSource(i, { value: e.target.value })}
                    />
                  </Field>
                  <IconButton
                    icon={TrashIcon}
                    label="Remove this source"
                    variant="secondary"
                    disabled={form.sources.length === 1}
                    onClick={() => set('sources', form.sources.filter((_, j) => j !== i))}
                    className="justify-self-end"
                  />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3 sm:w-80">
                  <Field label="Country" optional>
                    <Input
                      className="uppercase"
                      maxLength={2}
                      value={source.country}
                      placeholder="US"
                      disabled={source.kind === 'channel'}
                      title={source.kind === 'channel' ? 'Only for topic searches' : undefined}
                      onChange={(e) => setSource(i, { country: e.target.value })}
                    />
                  </Field>
                  <Field label="Language" optional>
                    <Input
                      className="lowercase"
                      maxLength={3}
                      value={source.language}
                      placeholder="en"
                      onChange={(e) => setSource(i, { language: e.target.value })}
                    />
                  </Field>
                </div>
              </div>
            ))}
            <Button
              icon={PlusIcon}
              onClick={() => set('sources', [...form.sources, { kind: 'topic', value: '', country: '', language: '' }])}
            >
              Add another source
            </Button>
            <p className="text-xs leading-relaxed text-fg-muted">
              A language keeps only videos YouTube reports in that language; a country alone implies its main
              language.
            </p>
          </div>
        </Card>

        <Card title="Which videos" description="Only videos that pass all of these get clipped.">
          <div className="grid gap-5 sm:grid-cols-2">
            <NumberField label="Minimum views" value={form.filters.min_views} min={0} step={1000} onChange={(n) => setFilter('min_views', n)} />
            <NumberField label="Not older than (days)" value={form.filters.max_age_days} min={1} onChange={(n) => setFilter('max_age_days', n)} />
            <NumberField label="Shortest video (minutes)" value={form.filters.min_minutes} min={0} onChange={(n) => setFilter('min_minutes', n)} />
            <NumberField label="Longest video (minutes)" value={form.filters.max_minutes} min={1} onChange={(n) => setFilter('max_minutes', n)} />
            <NumberField
              label="Wait for “Most replayed” (hours)"
              value={form.heatmap_wait_hours}
              min={0}
              onChange={(n) => set('heatmap_wait_hours', n)}
              hint="New videos wait for YouTube's rewatch graph; after this the AI picks from audio and comments."
            />
            <NumberField label="Clips per video" value={form.clips_per_video} min={1} onChange={(n) => set('clips_per_video', n)} />
          </div>
        </Card>

        <Card title="When and where to post">
          <div className="grid gap-5 sm:grid-cols-2">
            <Field label="Daily posting times" hint="24-hour times, separated by commas.">
              <Input className="numeric" value={slotsText} placeholder="12:00, 18:00" onChange={(e) => setSlotsText(e.target.value)} />
            </Field>
            <Field label="Time zone" optional hint="Empty uses this computer's time zone.">
              <Input value={form.timezone} placeholder="e.g. Europe/Berlin" onChange={(e) => set('timezone', e.target.value)} />
            </Field>
          </div>
          <p className="mt-5 mb-2 text-sm font-medium text-fg">Post to</p>
          <div className="grid gap-2 sm:grid-cols-2">
            {(['youtube', 'instagram'] as Platform[]).map((p) => {
              const on = form.platforms.includes(p)
              return (
                <label
                  key={p}
                  className={cx(
                    'flex cursor-pointer items-center gap-3 rounded-xl border px-3.5 py-3 text-sm font-medium text-fg transition-colors',
                    on ? 'border-accent bg-accent-soft ring-1 ring-accent' : 'border-line hover:bg-subtle',
                  )}
                >
                  <input type="checkbox" className="size-4" checked={on} onChange={() => togglePlatform(p)} />
                  {p === 'youtube' ? 'YouTube Shorts' : 'Instagram Reels'}
                </label>
              )
            })}
          </div>
          <Toggle
            className="mt-5"
            label="Let me approve each clip before it's booked"
            hint="Recommended. Clips wait on the autopilot's page until you approve them."
            checked={form.approve_before_posting}
            onChange={(on) => set('approve_before_posting', on)}
          />
        </Card>

        <Card>
          <Checkbox
            label="I have the rights (or the creators' permission) to clip and repost videos from these sources."
            checked={form.rights_confirmed}
            onChange={(on) => set('rights_confirmed', on)}
          />
        </Card>

        {error && (
          <Callout tone="danger">
            <span className="whitespace-pre-line">{error}</span>
          </Callout>
        )}
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" loading={busy} onClick={save}>
            {busy ? 'Saving…' : autopilotId ? 'Save changes' : 'Create autopilot'}
          </Button>
          <ButtonLink to={back}>Cancel</ButtonLink>
        </div>
      </div>
    </div>
  )
}
