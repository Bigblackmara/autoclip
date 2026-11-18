import { useState } from 'react'
import { Link } from 'react-router-dom'

import { Button, CheckIcon, GamepadIcon } from '../ui'
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
    <span className={compact ? 'inline-flex items-center gap-2' : 'inline-flex flex-wrap items-center gap-2'}>
      {compact ? (
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-md text-xs font-medium text-accent-text hover:underline disabled:text-fg-subtle disabled:no-underline"
          onClick={() => void send()}
          disabled={disabled || state !== 'idle'}
          title="Queue this video in GTA Mode (Hype Curve, two judging rounds, uncropped frame)"
        >
          <GamepadIcon size={13} />
          {label}
        </button>
      ) : (
        <Button
          icon={state === 'done' ? CheckIcon : GamepadIcon}
          loading={state === 'busy'}
          onClick={() => void send()}
          disabled={disabled || state !== 'idle'}
          title="Queue this video in GTA Mode (Hype Curve, two judging rounds, uncropped frame)"
        >
          {label}
        </Button>
      )}
      {state === 'done' && !compact && (
        <Link to="/gta" className="link text-sm">
          Open GTA Mode
        </Link>
      )}
      {error && <span className="text-xs text-danger">{error}</span>}
    </span>
  )
}
