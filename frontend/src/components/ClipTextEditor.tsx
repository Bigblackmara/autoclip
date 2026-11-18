import { useEffect, useState } from 'react'

import type { Clip, ClipPatch } from '../api'

/** The render cuts the headline here (see edit.HOOK_MAX_CHARS). */
const HOOK_MAX = 80
const DESCRIPTION_MAX = 2200

function splitTags(text: string): string[] {
  return text.split(/[\s,]+/).filter(Boolean)
}

/**
 * The words around a clip: the on-screen hook (one of the AI's three options
 * or free text), the post description and hashtags, and the flash-forward
 * opening. Text is saved with one button; the flash-forward toggle saves at
 * once, like the style and ratio pickers.
 */
export function ClipTextEditor({
  clip,
  onSave,
}: {
  clip: Clip
  onSave: (patch: ClipPatch) => Promise<void>
}) {
  const [hook, setHook] = useState(clip.headline)
  const [description, setDescription] = useState(clip.description)
  const [tags, setTags] = useState(clip.hashtags.join(' '))
  const [saving, setSaving] = useState(false)

  // A different clip, or new server values after a save: start from them.
  // Compared by value, so an unrelated update (a style pick returns a fresh
  // clip object) does not wipe text being typed.
  const storedTags = clip.hashtags.join(' ')
  useEffect(() => {
    setHook(clip.headline)
    setDescription(clip.description)
    setTags(storedTags)
  }, [clip.id, clip.headline, clip.description, storedTags])

  const hashtags = splitTags(tags)
  const dirty =
    hook.trim() !== clip.headline ||
    description.trim() !== clip.description ||
    hashtags.join(' ') !== clip.hashtags.join(' ')

  const save = async () => {
    setSaving(true)
    try {
      await onSave({ hook_text: hook.trim(), description: description.trim(), hashtags })
    } finally {
      setSaving(false)
    }
  }

  const flashOn = clip.flash_forward ?? clip.flash_forward_default

  return (
    <div className="space-y-8">
      <div>
        <p className="eyebrow border-b border-ink-800 pb-2">On-screen hook</p>
        {clip.hook_options.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2" role="radiogroup" aria-label="Hook options">
            {clip.hook_options.map((option) => (
              <button
                key={option}
                type="button"
                role="radio"
                aria-checked={hook.trim() === option}
                onClick={() => setHook(option)}
                className={[
                  'border px-3 py-1.5 text-left text-sm leading-snug transition-colors duration-200',
                  hook.trim() === option
                    ? 'border-sodium-500 bg-ink-850/70 text-ink-100'
                    : 'border-ink-700 text-ink-300 hover:border-ink-500 hover:text-ink-100',
                ].join(' ')}
              >
                {option}
              </button>
            ))}
          </div>
        )}
        <label className="mt-3 block">
          <span className="sr-only">Hook text</span>
          <input
            className="field text-sm"
            value={hook}
            maxLength={HOOK_MAX}
            onChange={(e) => setHook(e.target.value)}
            placeholder={clip.hook_options.length ? 'Or write your own…' : 'Nobody tells you this…'}
          />
        </label>
        <p className="numeric mt-1 text-right text-xs text-ink-600">
          {hook.length}/{HOOK_MAX}
        </p>
      </div>

      <div>
        <p className="eyebrow border-b border-ink-800 pb-2">Post text</p>
        <label className="mt-3 block">
          <span className="text-xs text-ink-500">Description</span>
          <textarea
            className="field mt-1 min-h-24 text-sm leading-relaxed"
            value={description}
            maxLength={DESCRIPTION_MAX}
            rows={4}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="What viewers read under the video"
          />
        </label>
        <label className="mt-3 block">
          <span className="text-xs text-ink-500">Hashtags (separate with spaces)</span>
          <input
            className="field mt-1 text-sm"
            value={tags}
            onChange={(e) => setTags(e.target.value)}
            placeholder="#podcast #startup"
          />
        </label>
        <div className="mt-3 flex items-center justify-between gap-3">
          <span className="numeric text-xs text-ink-600">
            {hashtags.length} tags · {description.length}/{DESCRIPTION_MAX}
          </span>
          <button onClick={save} disabled={!dirty || saving} className="btn btn-ghost">
            {saving ? 'Saving…' : 'Save text'}
          </button>
        </div>
      </div>

      <div>
        <p className="eyebrow border-b border-ink-800 pb-2">Flash-forward opening</p>
        <label className="mt-3 flex items-center gap-3 text-sm text-ink-200">
          <input
            type="checkbox"
            checked={flashOn}
            onChange={(e) => void onSave({ flash_forward: e.target.checked })}
            className="size-4 accent-sodium-500"
          />
          Play the best line first, then the clip from the start
        </label>
        <p className="mt-2 text-xs leading-snug text-ink-500">
          {clip.has_teaser
            ? 'Uses the line the AI flagged as the tease.'
            : 'The AI flagged no tease line; the strongest short line is picked by rule.'}{' '}
          {clip.flash_forward === null ? (
            `Following the default (${clip.flash_forward_default ? 'on' : 'off'}).`
          ) : (
            <button
              type="button"
              onClick={() => void onSave({ flash_forward: null })}
              className="text-sodium-500 underline underline-offset-4"
            >
              Use the default
            </button>
          )}
        </p>
      </div>
    </div>
  )
}
