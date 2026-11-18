import { useState } from 'react'
import { Link } from 'react-router-dom'

import { gtaApi } from './api'
import { errorMessage } from './editsApi'

/**
 * Queue a link in GTA Mode's stream queue (POST /api/gta/streams), from an
 * Explore result card or the Peek panel. YouTube VODs work like Twitch ones.
 */
export function SendToGtaButton({
  url,
  disabled = false,
  compact = false,
}: {
  url: string
  disabled?: boolean
  compact?: boolean
}) {
  const [state, setState] = useState<'idle' | 'busy' | 'done'>('idle')
  const [error, setError] = useState('')

  const send = async () => {
    setState('busy')
    setError('')
    try {
      const result = await gtaApi.addStreams([url], {})
      if (result.added.length === 0 && result.skipped.length > 0) throw new Error(result.skipped[0].reason)
      setState('done')
    } catch (err) {
      setError(errorMessage(err))
      setState('idle')
    }
  }

  const label = state === 'busy' ? 'Queueing…' : state === 'done' ? 'Queued in GTA Mode' : 'Send to GTA Mode'

  return (
    <span className={compact ? 'inline-flex items-baseline gap-2' : 'inline-flex flex-wrap items-center gap-2'}>
      <button
        type="button"
        className={compact ? 'text-xs text-ink-400 underline-offset-4 hover:text-sodium-400 hover:underline disabled:opacity-60' : 'btn btn-ghost'}
        onClick={() => void send()}
        disabled={disabled || state !== 'idle'}
        title="Queue this video in GTA Mode (Hype Curve, two judging rounds, uncropped frame)"
      >
        {label}
      </button>
      {state === 'done' && !compact && (
        <Link to="/gta" className="text-xs text-sodium-500 underline underline-offset-4">
          Open GTA Mode
        </Link>
      )}
      {error && <span className="text-xs text-signal-bad">{error}</span>}
    </span>
  )
}
