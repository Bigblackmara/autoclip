import { useMemo, useRef, useState } from 'react'

import { formatDuration } from '../api'
import { MAX_PICK_S, MIN_PICK_S, type AudienceClipBand, type Hotspot, type TimeRange } from '../picks/api'

type Component = 'heatmap' | 'comments' | 'energy'

/** Line colour per audience signal; the blend is drawn in sodium. */
export const COMPONENT_STYLE: Record<Component, { label: string; color: string; dash?: string }> = {
  heatmap: { label: 'Most replayed', color: 'var(--color-ink-200)' },
  comments: { label: 'Comment timestamps', color: 'var(--color-signal-good)', dash: '5 4' },
  energy: { label: 'Audio energy', color: 'var(--color-ink-400)', dash: '2 4' },
}

const W = 1000
const PLOT_TOP = 14
const BAND_H = 14
const EDGE_PX = 9
const DRAG_PX = 5

/**
 * "Where viewers rewatch" graph with drag-to-pick ranges.
 *
 * Drag across empty graph to add a part; drag a part's edge to adjust it;
 * click without dragging to seek. Picks are numbered in time order — the
 * order a joined clip plays them in.
 */
export function RewatchGraph({
  curve,
  components,
  visible,
  stepS,
  duration,
  bands,
  hotspots,
  picks,
  onPicksChange,
  onSeek,
  playhead,
  height = 200,
}: {
  curve: number[]
  components: Partial<Record<Component, number[]>>
  visible: Record<Component, boolean>
  stepS: number
  duration: number
  bands: AudienceClipBand[]
  hotspots: Hotspot[]
  picks: TimeRange[]
  onPicksChange: (picks: TimeRange[]) => void
  onSeek?: (t: number) => void
  playhead?: number | null
  height?: number
}) {
  const H = height
  const plotBottom = H - BAND_H - 10
  const plotH = plotBottom - PLOT_TOP
  const ref = useRef<SVGSVGElement>(null)
  const drag = useRef<{ mode: 'new' | 'start' | 'end'; anchor: number; idx: number; moved: boolean } | null>(
    null,
  )
  const [draft, setDraft] = useState<TimeRange | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [edgeHover, setEdgeHover] = useState(false)

  const safeDuration = Math.max(duration, 1)
  const x = (t: number) => (t / safeDuration) * W

  const path = (series: number[]) => {
    if (!series.length) return ''
    const pts = series.map((v, i) => [x((i + 0.5) * stepS), plotBottom - Math.max(0, Math.min(1, v)) * plotH])
    let d = `M0,${pts[0][1]} L${pts[0][0]},${pts[0][1]}`
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1]
      const [x1, y1] = pts[i]
      const mx = (x0 + x1) / 2
      d += ` C${mx},${y0} ${mx},${y1} ${x1},${y1}`
    }
    return `${d} L${W},${pts[pts.length - 1][1]}`
  }

  // At most ~2000 points per series (the server pools longer curves), so
  // rebuilding the paths on each hover move is cheap.
  const line = path(curve)
  const area = line ? `${line} L${W},${plotBottom} L0,${plotBottom} Z` : ''
  const componentLines = (Object.keys(COMPONENT_STYLE) as Component[])
    .filter((name) => visible[name] && components[name]?.length)
    .map((name) => ({ name, d: path(components[name]!) }))

  const timeAt = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return 0
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * safeDuration
  }

  const edgeAt = (clientX: number): { idx: number; edge: 'start' | 'end' } | null => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect || !picks.length) return null
    const tolerance = (EDGE_PX / rect.width) * safeDuration
    const t = timeAt(clientX)
    let best: { idx: number; edge: 'start' | 'end'; d: number } | null = null
    picks.forEach((p, idx) => {
      for (const edge of ['start', 'end'] as const) {
        const d = Math.abs(t - p[edge])
        if (d <= tolerance && (!best || d < best.d)) best = { idx, edge, d }
      }
    })
    return best
  }

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    const t = timeAt(e.clientX)
    const edge = edgeAt(e.clientX)
    drag.current = edge
      ? { mode: edge.edge, anchor: t, idx: edge.idx, moved: false }
      : { mode: 'new', anchor: t, idx: -1, moved: false }
    ;(e.currentTarget as Element).setPointerCapture?.(e.pointerId)
  }

  const onMove = (e: React.PointerEvent) => {
    const t = timeAt(e.clientX)
    setHover(t)
    const d = drag.current
    if (!d) {
      setEdgeHover(Boolean(edgeAt(e.clientX)))
      return
    }
    const rect = ref.current?.getBoundingClientRect()
    const px = rect ? (Math.abs(t - d.anchor) / safeDuration) * rect.width : 0
    if (!d.moved && px < DRAG_PX) return
    d.moved = true
    if (d.mode === 'new') {
      let start = Math.min(d.anchor, t)
      let end = Math.max(d.anchor, t)
      if (end - start > MAX_PICK_S) {
        if (t < d.anchor) start = end - MAX_PICK_S
        else end = start + MAX_PICK_S
      }
      setDraft({ start, end })
      return
    }
    const next = picks.map((p) => ({ ...p }))
    const p = next[d.idx]
    if (d.mode === 'start') p.start = Math.max(0, Math.min(t, p.end - MIN_PICK_S), p.end - MAX_PICK_S)
    else p.end = Math.min(safeDuration, Math.max(t, p.start + MIN_PICK_S), p.start + MAX_PICK_S)
    onPicksChange(next)
  }

  const onUp = (e: React.PointerEvent) => {
    const d = drag.current
    drag.current = null
    if (!d) return
    if (d.mode === 'new' && d.moved && draft) {
      setDraft(null)
      if (draft.end - draft.start >= MIN_PICK_S) {
        onPicksChange(
          [...picks, { start: round(draft.start), end: round(draft.end) }].sort((a, b) => a.start - b.start),
        )
      }
      return
    }
    setDraft(null)
    if (d.moved) {
      // Finished resizing: keep the list in time order.
      onPicksChange([...picks].map((p) => ({ start: round(p.start), end: round(p.end) })).sort((a, b) => a.start - b.start))
      return
    }
    onSeek?.(timeAt(e.clientX))
  }

  const hoverIndex = hover == null || !curve.length ? null : Math.min(curve.length - 1, Math.floor(hover / stepS))
  const ticks = useMemo(() => tickTimes(safeDuration), [safeDuration])
  const cursor = edgeHover ? 'ew-resize' : 'crosshair'
  const pct = (t: number) => `${(t / safeDuration) * 100}%`

  return (
    <div className="relative select-none">
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        width="100%"
        height={H}
        role="img"
        aria-label={`Where viewers rewatch, across ${formatDuration(duration)} of video, with ${picks.length} part${picks.length === 1 ? '' : 's'} picked. Drag across the graph to pick a part.`}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={() => !drag.current && setHover(null)}
        style={{ display: 'block', overflow: 'visible', cursor, touchAction: 'none' }}
      >
        <defs>
          <linearGradient id="rewatch-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--color-sodium-500)" stopOpacity="0.42" />
            <stop offset="1" stopColor="var(--color-sodium-500)" stopOpacity="0.04" />
          </linearGradient>
        </defs>

        {[0.5, 1].map((g) => (
          <line
            key={g}
            x1={0}
            x2={W}
            y1={plotBottom - g * plotH}
            y2={plotBottom - g * plotH}
            stroke="var(--color-ink-800)"
            strokeDasharray="3 6"
            vectorEffect="non-scaling-stroke"
          />
        ))}
        <line x1={0} x2={W} y1={plotBottom} y2={plotBottom} stroke="var(--color-ink-700)" vectorEffect="non-scaling-stroke" />

        {/* Picked parts: a full-height wash behind the curve. */}
        {picks.map((p, i) => (
          <rect
            key={`pick-${i}`}
            x={x(p.start)}
            y={PLOT_TOP - 4}
            width={Math.max(1, x(p.end) - x(p.start))}
            height={plotBottom - PLOT_TOP + 4}
            fill="var(--color-sodium-500)"
            fillOpacity={0.14}
          />
        ))}
        {draft && (
          <rect
            x={x(draft.start)}
            y={PLOT_TOP - 4}
            width={Math.max(1, x(draft.end) - x(draft.start))}
            height={plotBottom - PLOT_TOP + 4}
            fill="var(--color-sodium-400)"
            fillOpacity={0.2}
          />
        )}

        {area && <path d={area} fill="url(#rewatch-fill)" />}
        {componentLines.map(({ name, d }) => (
          <path
            key={name}
            d={d}
            fill="none"
            stroke={COMPONENT_STYLE[name].color}
            strokeOpacity={0.75}
            strokeWidth={1.25}
            strokeDasharray={COMPONENT_STYLE[name].dash}
            vectorEffect="non-scaling-stroke"
          />
        ))}
        {line && (
          <path d={line} fill="none" stroke="var(--color-sodium-500)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
        )}

        {/* Pick edges: the handles. */}
        {picks.map((p, i) =>
          (['start', 'end'] as const).map((edge) => (
            <line
              key={`edge-${i}-${edge}`}
              x1={x(p[edge])}
              x2={x(p[edge])}
              y1={PLOT_TOP - 6}
              y2={plotBottom}
              stroke="var(--color-sodium-400)"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          )),
        )}

        {/* The job's existing clips, as a strip under the plot. */}
        {bands.flatMap((band) =>
          (band.spans.length ? band.spans : [[band.start_s, band.end_s] as [number, number]]).map(([a, b], k) => (
            <rect
              key={`${band.id}-${k}`}
              x={x(a)}
              y={plotBottom + 6}
              width={Math.max(1.5, x(b) - x(a))}
              height={BAND_H}
              fill="var(--color-ink-600)"
            >
              <title>{`#${band.rank} ${band.title} (${formatDuration(a)}–${formatDuration(b)})`}</title>
            </rect>
          )),
        )}

        {hotspots.map((h, i) => (
          <circle
            key={`hot-${i}`}
            cx={x(h.peak_s)}
            cy={plotBottom - h.value * plotH}
            r={3}
            fill="var(--color-sodium-400)"
            vectorEffect="non-scaling-stroke"
          >
            <title>{`${formatDuration(h.peak_s)} · ${h.reasons.join(', ') || 'peak'}`}</title>
          </circle>
        ))}

        {playhead != null && (
          <line
            x1={x(playhead)}
            x2={x(playhead)}
            y1={0}
            y2={plotBottom}
            stroke="var(--color-ink-100)"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
        )}
        {hover != null && (
          <line
            x1={x(hover)}
            x2={x(hover)}
            y1={PLOT_TOP}
            y2={plotBottom}
            stroke="var(--color-ink-500)"
            strokeDasharray="2 3"
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>

      {/* HTML labels: SVG text would stretch with preserveAspectRatio="none". */}
      {picks.map((p, i) => (
        <span
          key={`label-${i}`}
          className="numeric pointer-events-none absolute -top-5 -translate-x-1/2 rounded-sm bg-sodium-500 px-1.5 text-[0.6875rem] font-semibold leading-4 text-ink-900"
          style={{ left: pct((p.start + p.end) / 2) }}
        >
          {i + 1}
        </span>
      ))}
      {hover != null && (
        <span
          className="numeric pointer-events-none absolute top-0 -translate-x-1/2 -translate-y-full whitespace-nowrap text-[0.6875rem] text-ink-300"
          style={{ left: pct(Math.min(Math.max(hover, safeDuration * 0.04), safeDuration * 0.96)) }}
        >
          {formatDuration(hover)}
          {hoverIndex != null && ` · ${Math.round((curve[hoverIndex] ?? 0) * 100)}`}
        </span>
      )}
      <div className="relative mt-1 h-4">
        {ticks.map((t) => (
          <span
            key={t}
            className="numeric absolute -translate-x-1/2 text-[0.6875rem] text-ink-600"
            style={{ left: pct(t) }}
          >
            {formatDuration(t)}
          </span>
        ))}
      </div>
    </div>
  )
}

function round(t: number) {
  return Math.round(t * 10) / 10
}

/** Five to eight round-numbered time ticks across the video. */
function tickTimes(duration: number): number[] {
  const steps = [10, 30, 60, 120, 300, 600, 900, 1800, 3600]
  const step = steps.find((s) => duration / s <= 8) ?? 3600
  const out: number[] = []
  for (let t = step; t < duration - step * 0.4; t += step) out.push(t)
  return out
}
