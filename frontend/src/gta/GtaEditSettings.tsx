import { useEffect, useState } from 'react'

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

  if (!values) return message ? <p className="text-xs text-signal-bad">{message.text}</p> : null

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
      setMessage({ tone: 'good', text: 'Saved. New renders use it.' })
    } catch (err) {
      setMessage({ tone: 'bad', text: errorMessage(err) })
    }
  }

  const language = CAPTION_LANGUAGES.find((l) => l.value === values.caption_language)

  return (
    <section className="grid gap-5 sm:grid-cols-2">
      <div className="flex flex-col gap-2 text-xs text-ink-400">
        Flash-forward
        <label className="flex items-center gap-2 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={values.flash_forward}
            onChange={(e) => void save({ flash_forward: e.target.checked })}
          />
          Open on the peak reaction (punched in), then play from the start
        </label>
        <label className="text-xs text-ink-400">
          Tease length (1–3 s)
          <input
            className="field"
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
        </label>
        <span className="text-ink-500">Each clip can override it. The tease counts towards the 60 s limit.</span>
      </div>
      <label className="text-xs text-ink-400">
        Caption language (captions, hook title and post text)
        <select
          className="field"
          value={values.caption_language}
          onChange={(e) => void save({ caption_language: e.target.value as CaptionLanguage })}
        >
          {CAPTION_LANGUAGES.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <span className="mt-1 block text-ink-500">{language?.hint}</span>
      </label>
      {message && (
        <p
          className={`whitespace-pre-line text-xs sm:col-span-2 ${
            message.tone === 'good' ? 'text-signal-good' : 'text-signal-bad'
          }`}
        >
          {message.text}
        </p>
      )}
    </section>
  )
}
