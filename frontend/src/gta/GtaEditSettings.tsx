import { useEffect, useState } from 'react'

import { Callout, Card, Field, Input, Select, SkeletonCard, SparklesIcon, Toggle, useToast } from '../ui'
import { CAPTION_LANGUAGES, errorMessage, gtaEditsApi, type CaptionLanguage, type GtaEditSettings as Values } from './editsApi'

/**
 * GTA settings for the flash-forward opening and the caption language.
 * Saves only its own keys (the settings API merges), so it never fights the
 * rest of the settings form.
 */
export function GtaEditSettings() {
  const [values, setValues] = useState<Values | null>(null)
  const [length, setLength] = useState('2')
  const [message, setMessage] = useState<{ tone: 'good' | 'bad'; text: string } | null>(null)
  const toast = useToast()

  useEffect(() => {
    gtaEditsApi
      .editSettings()
      .then((s) => {
        setValues({
          flash_forward: Boolean(s.flash_forward),
          flash_forward_s: s.flash_forward_s ?? 2,
          caption_language: s.caption_language ?? 'as_spoken',
        })
        setLength(String(s.flash_forward_s ?? 2))
      })
      .catch((err) => setMessage({ tone: 'bad', text: errorMessage(err) }))
  }, [])

  if (!values) return message ? <Callout tone="danger">{message.text}</Callout> : <SkeletonCard lines={3} />

  const save = async (patch: Partial<Values>) => {
    setMessage(null)
    try {
      const saved = await gtaEditsApi.saveEditSettings(patch)
      setValues({
        flash_forward: saved.flash_forward,
        flash_forward_s: saved.flash_forward_s,
        caption_language: saved.caption_language,
      })
      setLength(String(saved.flash_forward_s))
      setMessage(null)
      toast.success('Saved', 'New renders use it.')
    } catch (err) {
      setMessage({ tone: 'bad', text: errorMessage(err) })
    }
  }

  const language = CAPTION_LANGUAGES.find((l) => l.value === values.caption_language)

  return (
    <Card title="Opening & caption language" icon={SparklesIcon} description="These two save as soon as you change them.">
      <div className="grid gap-5 sm:grid-cols-2">
        <div className="grid content-start gap-4">
          <Toggle
            label="Flash-forward opening"
            hint="Open on the peak reaction (punched in), then play from the start. Each clip can override it."
            checked={values.flash_forward}
            onChange={(on) => void save({ flash_forward: on })}
          />
          <Field label="Tease length (1–3 seconds)" hint="The tease counts towards the 60 s limit.">
            <Input
              type="number"
              min={1}
              max={3}
              step={0.25}
              value={length}
              onChange={(e) => setLength(e.target.value)}
              onBlur={() => {
                const n = Number(length)
                if (Number.isFinite(n) && n !== values.flash_forward_s) void save({ flash_forward_s: n })
              }}
            />
          </Field>
        </div>
        <Field label="Caption language" hint={`Captions, hook title and post text. ${language?.hint ?? ''}`}>
          <Select
            value={values.caption_language}
            onChange={(e) => void save({ caption_language: e.target.value as CaptionLanguage })}
            options={CAPTION_LANGUAGES.map((o) => ({ value: o.value, label: o.label }))}
          />
        </Field>
      </div>
      {message?.tone === 'bad' && (
        <Callout tone="danger" className="mt-4">
          <span className="whitespace-pre-line">{message.text}</span>
        </Callout>
      )}
    </Card>
  )
}
