import { formatDuration } from '../api'
import type { HeatPoint, Peak } from './api'

/**
 * YouTube's "Most replayed" graph as an area chart, with the top peaks
 * numbered on it. Clicking anywhere on the graph, or on a peak, seeks.
 *
 * Drawn in a fixed 1000-unit-wide viewBox and stretched to the container, so
 * the geometry never needs a resize observer.
 */
export function HeatChart({
  heatmap,
  peaks,
  duration,
  position,
  onSeek,
  height = 96,
}: {
  heatmap: HeatPoint[]
  peaks: Peak[]
  duration: number
  /** Where the player was last sent, in seconds. */
  position: number | null
  onSeek: (seconds: number) => void
  height?: number
}) {
  const W = 1000
  const H = height
  const top = 18 // room for the peak numbers
  const span = Math.max(1, duration)
  const x = (t: number) => (Math.max(0, Math.min(span, t)) / span) * W
  const y = (v: number) => top + (1 - Math.max(0, Math.min(1, v))) * (H - top)

  // A smooth-enough line through each bin's midpoint, closed to the baseline.
  const points = heatmap.map((p) => [x((p.start_s + p.end_s) / 2), y(p.value)] as const)
  const line = points.map(([px, py], i) => `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`).join(' ')
  const area = points.length
    ? `M0,${H} L0,${points[0][1].toFixed(1)} ${line.replace(/^M/, 'L')} L${W},${points[points.length - 1][1].toFixed(1)} L${W},${H} Z`
    : ''

  const valueAt = (t: number) => {
    const bin = heatmap.find((p) => t >= p.start_s && t < p.end_s)
    return bin ? bin.value : 0
  }

  const seekFromEvent = (event: React.MouseEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const fraction = (event.clientX - rect.left) / rect.width
    onSeek(Math.max(0, Math.min(span, fraction * span)))
  }

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="block w-full cursor-pointer"
        style={{ height: H }}
        onClick={seekFromEvent}
        role="img"
        aria-label="Most replayed graph. Click to jump the player there."
      >
        <defs>
          <linearGradient id="heat-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="var(--color-sodium-500)" stopOpacity="0.55" />
            <stop offset="100%" stopColor="var(--color-sodium-500)" stopOpacity="0.04" />
          </linearGradient>
        </defs>

        {/* Peak bands under the curve */}
        {peaks.map((p, i) => (
          <rect
            key={`band-${i}`}
            x={x(p.start_s)}
            y={top}
            width={Math.max(2, x(p.end_s) - x(p.start_s))}
            height={H - top}
            fill="var(--color-sodium-500)"
            opacity={0.1}
          />
        ))}

        <path d={area} fill="url(#heat-fill)" />
        <path
          d={line}
          fill="none"
          stroke="var(--color-sodium-500)"
          strokeWidth={1.5}
          vectorEffect="non-scaling-stroke"
        />
        <line x1={0} x2={W} y1={H - 0.5} y2={H - 0.5} stroke="var(--color-ink-700)" vectorEffect="non-scaling-stroke" />

        {position !== null && (
          <line
            x1={x(position)}
            x2={x(position)}
            y1={top - 4}
            y2={H}
            stroke="var(--color-ink-100)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}

        {/* Peak markers: a dot on the curve with its rank above. */}
        {peaks.map((p, i) => (
          <g
            key={`peak-${i}`}
            onClick={(e) => {
              e.stopPropagation()
              onSeek(p.start_s)
            }}
            className="cursor-pointer"
          >
            <title>{`Peak ${i + 1} at ${formatDuration(p.peak_s)}`}</title>
            <line
              x1={x(p.peak_s)}
              x2={x(p.peak_s)}
              y1={top - 2}
              y2={y(valueAt(p.peak_s))}
              stroke="var(--color-sodium-400)"
              strokeDasharray="2 2"
              vectorEffect="non-scaling-stroke"
            />
            {/* Wide invisible hit target: the viewBox is stretched, so a
                narrow rect would be a sliver on a wide screen. */}
            <rect x={x(p.peak_s) - 14} y={0} width={28} height={H} fill="transparent" />
          </g>
        ))}
      </svg>

      {/* Rank labels as HTML so the text isn't stretched with the viewBox. */}
      <div className="relative h-0">
        {peaks.map((p, i) => (
          <button
            key={`label-${i}`}
            type="button"
            onClick={() => onSeek(p.start_s)}
            className="numeric absolute flex h-4 w-4 -translate-x-1/2 items-center justify-center rounded-full bg-sodium-500 text-[10px] font-bold text-ink-900 hover:bg-sodium-400"
            style={{ left: `${(x(p.peak_s) / W) * 100}%`, top: -H }}
            aria-label={`Jump to peak ${i + 1} at ${formatDuration(p.start_s)}`}
          >
            {i + 1}
          </button>
        ))}
      </div>

      <div className="numeric mt-1 flex justify-between text-[11px] text-ink-500">
        <span>0:00</span>
        <span>{formatDuration(span / 2)}</span>
        <span>{formatDuration(span)}</span>
      </div>
    </div>
  )
}
