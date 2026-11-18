/** Small shared pieces for the Schedule, Autopilot and Post screens. */

import { Badge, type Tone } from '../ui'
import { STATUS_LABEL, type PostStatus } from './api'

const POST_TONE: Record<string, Tone> = {
  posted: 'success',
  scheduled: 'accent',
  planned: 'accent',
  queued: 'neutral',
  uploading: 'info',
  checking: 'info',
  failed: 'danger',
  interrupted: 'danger',
  canceled: 'neutral',
  skipped: 'neutral',
}

/** A post's status as a coloured pill. */
export function PostStatusBadge({ status }: { status: PostStatus | string }) {
  return (
    <Badge tone={POST_TONE[status] ?? 'neutral'} dot pulse={status === 'uploading' || status === 'checking'}>
      {STATUS_LABEL[status] ?? status}
    </Badge>
  )
}

/** YouTube / Instagram as a small label. */
export function PlatformTag({ platform }: { platform: 'youtube' | 'instagram' }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium text-fg-muted">
      <span className={platform === 'youtube' ? 'size-2 rounded-sm bg-[#e62117]' : 'size-2 rounded-sm bg-[#d62976]'} aria-hidden />
      {platform === 'youtube' ? 'YouTube' : 'Instagram'}
    </span>
  )
}
