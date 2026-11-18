import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'

import {
  api,
  formatBytes,
  formatDuration,
  LAYOUTS,
  type CaptionStyle,
  type Clip,
  type ClipPatch,
  type CropPath,
  type Job,
  type Layout,
  type Word,
} from '../api'
import { CaptionEditor } from '../components/CaptionEditor'
import { ClipTextEditor } from '../components/ClipTextEditor'
import { ClipPlayer } from '../components/ClipPlayer'
import { ErrorNote } from '../components/ErrorNote'
import { TrimBar } from '../components/TrimBar'
import {
  Badge,
  Button,
  ButtonLink,
  Card,
  CheckIcon,
  DownloadIcon,
  EmptyState,
  FilmIcon,
  PageHeader,
  PageSkeleton,
  RefreshIcon,
  ScissorsIcon,
  Segmented,
  SparklesIcon,
  Tabs,
  TrashIcon,
  XIcon,
  cx,
  tabPanelProps,
} from '../ui'

const RATIOS = ['9:16', '1:1', '16:9'] as const

type EditorTab = 'text' | 'captions' | 'look'
const EDITOR_TABS = [
  { key: 'text', label: 'Hook & post text' },
  { key: 'captions', label: 'Captions' },
  { key: 'look', label: 'Style & layout' },
] as const

export function Review() {
  const { jobId } = useParams()
  const [job, setJob] = useState<Job | null>(null)
  const [clips, setClips] = useState<Clip[]>([])
  const [styles, setStyles] = useState<CaptionStyle[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [words, setWords] = useState<Word[]>([])
  const [cropPath, setCropPath] = useState<CropPath | null>(null)
  const [wordsDirty, setWordsDirty] = useState(false)
  const [savingWords, setSavingWords] = useState(false)
  const [exporting, setExporting] = useState<Set<string>>(new Set())
  // Per-clip layout picked here for the next export; unset means the job's.
  const [layouts, setLayouts] = useState<Record<string, Layout>>({})
  const [exportingKept, setExportingKept] = useState(false)
  const [makingVariant, setMakingVariant] = useState<string | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [editorTab, setEditorTab] = useState<EditorTab>('text')
  const editorTabs = useId()


  useEffect(() => {
    if (!jobId) return
    void Promise.all([api.getJob(jobId), api.listClips(jobId), api.captionStyles()])
      .then(([loadedJob, loadedClips, loadedStyles]) => {
        setJob(loadedJob)
        setClips(loadedClips)
        setStyles(loadedStyles)
        setSelectedId((current) => current ?? loadedClips[0]?.id ?? null)
      })
      .catch((err) => setError(err as Error))
  }, [jobId])

  const selected = useMemo(
    () => clips.find((clip) => clip.id === selectedId) ?? null,
    [clips, selectedId],
  )

  // The clip whose words are currently loaded. Responses for any other clip
  // are stale and must be dropped: flicking through the list let clip A's
  // late words land on clip B, and "Save edits" then wrote A's captions
  // onto B.
  const currentClipId = useRef<string | null>(null)
  currentClipId.current = selected?.id ?? null

  useEffect(() => {
    if (!selected) return
    let stale = false
    setWordsDirty(false)
    setWords([])
    setCropPath(null)
    api
      .getClipWords(selected.id)
      .then((loaded) => {
        if (!stale) setWords(loaded)
      })
      .catch(() => {
        if (!stale) setWords([])
      })
    // 404 is expected for audio-only sources and jobs that never reframed; the
    // player falls back to a centre crop, matching what the renderer does.
    api
      .getCropPath(selected.id)
      .then((loaded) => {
        if (!stale) setCropPath(loaded)
      })
      .catch(() => {
        if (!stale) setCropPath(null)
      })
    return () => {
      stale = true
    }
  }, [selected?.id])

  const patchClip = useCallback((updated: Clip) => {
    setClips((current) => current.map((clip) => (clip.id === updated.id ? updated : clip)))
  }, [])

  const setStatus = async (clip: Clip, status: Clip['status']) => {
    try {
      patchClip(await api.patchClip(clip.id, { status }))
    } catch (err) {
      setError(err as Error)
    }
  }

  const commitTrim = async (start: number, end: number) => {
    if (!selected) return
    const clipId = selected.id
    try {
      patchClip(await api.patchClip(clipId, { start_s: start, end_s: end }))
      const trimmedWords = await api.getClipWords(clipId)
      if (currentClipId.current === clipId) {
        setWords(trimmedWords)
        setWordsDirty(false)
      }
    } catch (err) {
      setError(err as Error)
    }
  }

  const saveWords = async () => {
    if (!selected) return
    const clipId = selected.id
    setSavingWords(true)
    try {
      patchClip(await api.patchCaptions(clipId, { words }))
      if (currentClipId.current === clipId) setWordsDirty(false)
    } catch (err) {
      setError(err as Error)
    } finally {
      setSavingWords(false)
    }
  }

  const saveText = async (clip: Clip, patch: ClipPatch) => {
    try {
      patchClip(await api.patchClip(clip.id, patch))
    } catch (err) {
      setError(err as Error)
    }
  }

  const makeVariant = async (clip: Clip) => {
    // One at a time: each variant is a full render.
    if (makingVariant || !jobId) return
    setMakingVariant(clip.id)
    setError(null)
    try {
      const variant = await api.makeVariant(clip.id, { layout: layouts[clip.id] })
      setClips(await api.listClips(jobId))
      setSelectedId(variant.id)
    } catch (err) {
      setError(err as Error)
      // The variant row exists even when its render failed; show it so the
      // export can be retried from it.
      api.listClips(jobId).then(setClips, () => undefined)
    } finally {
      setMakingVariant(null)
    }
  }

  const setStyle = async (clip: Clip, styleKey: string) => {
    try {
      patchClip(await api.patchCaptions(clip.id, { caption_style: styleKey }))
    } catch (err) {
      setError(err as Error)
    }
  }

  const setRatio = async (clip: Clip, ratio: string) => {
    try {
      patchClip(await api.patchCaptions(clip.id, { ratio }))
    } catch (err) {
      setError(err as Error)
    }
  }

  const exportClip = async (clip: Clip) => {
    setExporting((current) => new Set(current).add(clip.id))
    setError(null)
    try {
      await api.exportClip(
        clip.id,
        clip.ratio,
        clip.caption_style,
        false,
        layouts[clip.id] ?? clip.layout,
      )
      patchClip(await api.getClip(clip.id))
    } catch (err) {
      setError(err as Error)
    } finally {
      setExporting((current) => {
        const next = new Set(current)
        next.delete(clip.id)
        return next
      })
    }
  }

  const exportKept = async () => {
    // Guarded: a second click used to queue a second full pass, rendering
    // every kept clip twice.
    if (exportingKept) return
    setExportingKept(true)
    try {
      const targets = clips.filter((clip) => clip.status === 'kept')
      for (const clip of targets) await exportClip(clip)
    } finally {
      setExportingKept(false)
    }
  }

  const keptCount = clips.filter((clip) => clip.status === 'kept').length
  const activeStyle = styles.find((style) => style.key === selected?.caption_style)

  if (error && clips.length === 0) {
    return (
      <div className="max-w-3xl">
        <PageHeader back={{ to: '/projects', label: 'All projects' }} title="Couldn't open these clips" />
        <ErrorNote error={error} />
      </div>
    )
  }

  if (!job) return <PageSkeleton cards={2} />

  return (
    <div>
      <PageHeader
        back={{ to: '/projects', label: 'All projects' }}
        title={job.source?.title || 'Untitled'}
        description={
          <span className="numeric">
            {clips.length} clip{clips.length === 1 ? '' : 's'} · {keptCount} kept. Keep the ones you like, tweak
            them, then export.
          </span>
        }
        actions={
          <>
            <ButtonLink to={`/jobs/${job.id}/pick`} icon={ScissorsIcon}>
              Pick your own parts
            </ButtonLink>
            <Button
              variant="primary"
              icon={DownloadIcon}
              onClick={exportKept}
              disabled={keptCount === 0}
              loading={exportingKept}
              title={keptCount === 0 ? 'Keep at least one clip first' : undefined}
            >
              {exportingKept ? 'Exporting…' : `Export kept${keptCount ? ` (${keptCount})` : ''}`}
            </Button>
          </>
        }
      />

      {error && (
        <div className="mb-6 max-w-3xl">
          <ErrorNote error={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {clips.length === 0 ? (
        <NoClips jobId={job.id} />
      ) : (
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,19rem)_minmax(0,1fr)]">
          {/* Ranked list */}
          <Card padding="none" className="overflow-hidden lg:sticky lg:top-6">
            <div className="flex items-baseline justify-between border-b border-line px-4 py-3">
              <h2 className="text-sm font-semibold text-fg">Clips, best first</h2>
              <span className="numeric text-xs text-fg-muted">{clips.length}</span>
            </div>
            <ul className="max-h-[22rem] divide-y divide-line overflow-y-auto lg:max-h-[calc(100vh-12rem)]">
              {clips.map((clip, index) => (
                <li key={clip.id}>
                  <ClipRow
                    clip={clip}
                    index={index}
                    selected={clip.id === selectedId}
                    onSelect={() => setSelectedId(clip.id)}
                    onKeep={() => setStatus(clip, clip.status === 'kept' ? 'candidate' : 'kept')}
                    onDiscard={() => setStatus(clip, clip.status === 'discarded' ? 'candidate' : 'discarded')}
                  />
                </li>
              ))}
            </ul>
          </Card>

          {selected && jobId && (
            <Card padding="none" className="min-w-0">
              {/* Clip header: title, score, keep / drop, export */}
              <div className="flex flex-wrap items-start justify-between gap-4 border-b border-line px-5 py-4 sm:px-6">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <ScoreBadge score={selected.score} />
                    <span className="numeric text-xs text-fg-muted">{formatDuration(selected.duration_s)}</span>
                    {selected.status === 'kept' && <Badge tone="success">Kept</Badge>}
                    {selected.status === 'discarded' && <Badge>Dropped</Badge>}
                    {selected.variant_of && <Badge tone="info">Variant</Badge>}
                  </div>
                  <h2 className="mt-1.5 text-lg font-semibold leading-snug text-fg">{selected.title || 'Untitled clip'}</h2>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    icon={CheckIcon}
                    variant={selected.status === 'kept' ? 'primary' : 'secondary'}
                    aria-pressed={selected.status === 'kept'}
                    onClick={() => setStatus(selected, selected.status === 'kept' ? 'candidate' : 'kept')}
                  >
                    {selected.status === 'kept' ? 'Kept' : 'Keep'}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    icon={selected.status === 'discarded' ? RefreshIcon : TrashIcon}
                    onClick={() =>
                      setStatus(selected, selected.status === 'discarded' ? 'candidate' : 'discarded')
                    }
                  >
                    {selected.status === 'discarded' ? 'Restore' : 'Drop'}
                  </Button>
                </div>
              </div>

              <div className="grid gap-8 p-5 sm:p-6 xl:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
                {/* Player + trim + export */}
                <div className="min-w-0 space-y-6">
                  <ClipPlayer
                    src={api.mediaUrl(jobId)}
                    startS={selected.start_s}
                    endS={selected.end_s}
                    words={words}
                    style={activeStyle}
                    ratio={selected.ratio}
                    cropPath={cropPath}
                  />
                  <TrimBar
                    words={words}
                    startS={selected.start_s}
                    endS={selected.end_s}
                    originalStart={selected.start_s}
                    originalEnd={selected.end_s}
                    onCommit={commitTrim}
                  />
                  <div className="rounded-xl border border-line bg-subtle/60 p-4">
                    <Button
                      variant="primary"
                      block
                      icon={DownloadIcon}
                      onClick={() => exportClip(selected)}
                      loading={exporting.has(selected.id)}
                    >
                      {exporting.has(selected.id) ? 'Rendering…' : 'Export this clip'}
                    </Button>
                    <Button
                      block
                      className="mt-2"
                      icon={SparklesIcon}
                      onClick={() => makeVariant(selected)}
                      disabled={makingVariant !== null}
                      loading={makingVariant === selected.id}
                      title="Same moment with the next caption style, zoom pattern and hook, so it can be posted again"
                    >
                      {makingVariant === selected.id ? 'Rendering variant…' : 'Make a variant'}
                    </Button>
                    <p className="mt-2 text-xs leading-snug text-fg-muted">
                      A variant is the same moment with a different look, so you can post it again.
                    </p>

                    {selected.exports.length > 0 && (
                      <div className="mt-4 border-t border-line pt-3">
                        <p className="text-xs font-semibold text-fg">Downloads</p>
                        <ul className="mt-2 space-y-1.5">
                          {selected.exports.map((record) => (
                            <li key={record.id} className="flex items-center justify-between gap-3 text-sm">
                              <a
                                href={record.download_url}
                                download
                                className="inline-flex min-w-0 items-center gap-1.5 font-medium text-accent-text hover:underline"
                              >
                                <DownloadIcon size={14} />
                                <span className="truncate">
                                  {record.ratio} · {record.style}
                                </span>
                              </a>
                              <span className="numeric shrink-0 text-xs text-fg-muted">{formatBytes(record.size_bytes)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>

                {/* Editing */}
                <div className="min-w-0">
                  <div className="rounded-xl bg-accent-soft/60 p-4">
                    <p className="flex items-center gap-1.5 text-sm font-semibold text-fg">
                      <SparklesIcon size={15} className="text-accent-text" />
                      Why the AI picked this
                    </p>
                    <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">
                      {selected.reason || 'No reason was returned for this clip.'}
                    </p>
                    {selected.hook && (
                      <p className="mt-2 border-l-2 border-accent pl-3 text-sm font-medium italic text-fg">“{selected.hook}”</p>
                    )}
                  </div>

                  <Tabs
                    className="mt-5"
                    label="Edit this clip"
                    idBase={editorTabs}
                    items={EDITOR_TABS}
                    value={editorTab}
                    onChange={setEditorTab}
                  />

                  <div className="pt-5" {...tabPanelProps(editorTabs, editorTab)}>
                    {editorTab === 'text' && (
                      <ClipTextEditor clip={selected} onSave={(patch) => saveText(selected, patch)} />
                    )}

                    {editorTab === 'captions' && (
                      <CaptionEditor
                        words={words}
                        onChange={(next) => {
                          setWords(next)
                          setWordsDirty(true)
                        }}
                        onSave={saveWords}
                        saving={savingWords}
                        dirty={wordsDirty}
                      />
                    )}

                    {editorTab === 'look' && (
                      <div className="space-y-7">
                        <div>
                          <p className="text-sm font-semibold text-fg">Caption style</p>
                          <div className="mt-2.5 grid gap-2 sm:grid-cols-2">
                            {styles.map((style) => {
                              const active = style.key === selected.caption_style
                              return (
                                <button
                                  key={style.key}
                                  onClick={() => setStyle(selected, style.key)}
                                  aria-pressed={active}
                                  className={cx(
                                    'rounded-xl border px-3.5 py-2.5 text-left transition-colors duration-150',
                                    active
                                      ? 'border-accent bg-accent-soft ring-1 ring-accent'
                                      : 'border-line hover:border-line-strong hover:bg-subtle',
                                  )}
                                >
                                  <span className="text-sm font-semibold text-fg">{style.label}</span>
                                  <span className="mt-0.5 block text-xs leading-snug text-fg-muted">{style.description}</span>
                                </button>
                              )
                            })}
                          </div>
                        </div>

                        <div>
                          <p className="mb-2.5 text-sm font-semibold text-fg">Shape</p>
                          <Segmented
                            label="Aspect ratio"
                            options={RATIOS.map((r) => ({
                              value: r,
                              label: r === '9:16' ? '9:16 · Vertical' : r === '1:1' ? '1:1 · Square' : '16:9 · Wide',
                            }))}
                            value={selected.ratio as (typeof RATIOS)[number]}
                            onChange={(ratio) => setRatio(selected, ratio)}
                          />
                        </div>

                        <div>
                          <p className="mb-2.5 text-sm font-semibold text-fg">Layout</p>
                          <Segmented
                            label="Layout"
                            options={LAYOUTS.map(({ value, label }) => ({ value, label }))}
                            value={layouts[selected.id] ?? selected.layout}
                            onChange={(value) => setLayouts((all) => ({ ...all, [selected.id]: value }))}
                            disabled={selected.ratio !== '9:16'}
                          />
                          <p className="mt-2 text-xs leading-snug text-fg-muted">
                            {selected.ratio === '9:16'
                              ? 'Facecam + gameplay stacks a streamer’s webcam above the game. Auto uses it only when a facecam overlay is found. Applies to the next export.'
                              : 'Stacked layouts need 9:16; this shape always crops.'}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  )
}

function ScoreBadge({ score }: { score: number }) {
  const tone = score >= 85 ? 'bg-accent text-accent-fg' : score >= 70 ? 'bg-accent-soft text-accent-text' : 'bg-subtle text-fg-muted'
  return (
    <span
      className={cx('numeric inline-flex h-6 min-w-9 items-center justify-center rounded-full px-2 text-xs font-bold', tone)}
      title="How strong the AI thinks this clip is, out of 100"
    >
      {score}
    </span>
  )
}

function ClipRow({
  clip,
  index,
  selected,
  onSelect,
  onKeep,
  onDiscard,
}: {
  clip: Clip
  index: number
  selected: boolean
  onSelect: () => void
  onKeep: () => void
  onDiscard: () => void
}) {
  return (
    <div
      className={cx(
        'rise group relative grid cursor-pointer grid-cols-[auto_1fr] items-start gap-x-3 px-4 py-3 transition-colors duration-150',
        selected ? 'bg-accent-soft' : 'hover:bg-subtle',
        clip.status === 'discarded' && 'opacity-50',
      )}
      style={{ animationDelay: `${Math.min(index, 10) * 30}ms` }}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      aria-current={selected ? 'true' : undefined}
      onKeyDown={(e) => {
        // Only keys aimed at the row itself: Enter on the keep/drop buttons
        // inside it must not also re-select.
        if (e.target !== e.currentTarget) return
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onSelect()
        }
      }}
    >
      {selected && <span className="absolute inset-y-2 left-0 w-1 rounded-r-full bg-accent" aria-hidden />}
      <ScoreBadge score={clip.score} />

      <div className="min-w-0">
        <p className={cx('line-clamp-2 text-sm leading-snug', selected ? 'font-semibold text-fg' : 'font-medium text-fg')}>
          {clip.title || 'Untitled clip'}
        </p>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <span className="numeric text-xs text-fg-muted">{formatDuration(clip.duration_s)}</span>
          {clip.status === 'kept' && <span className="text-xs font-medium text-success">Kept</span>}
          {clip.user_trimmed && <span className="text-xs text-fg-subtle">trimmed</span>}
          {clip.variant_of && <span className="text-xs text-fg-subtle">variant</span>}
          {clip.exports.length > 0 && <span className="text-xs font-medium text-success">exported</span>}

          {/* Quick keep / drop: on hover, focus, or the current row. */}
          <span
            className={cx(
              'ml-auto flex gap-1 transition-opacity duration-150',
              // focus-within: keyboard users tabbing onto an invisible
              // button could not see what they were about to press.
              selected ? 'opacity-100' : 'opacity-0 group-focus-within:opacity-100 group-hover:opacity-100',
            )}
          >
            <button
              onClick={(e) => {
                e.stopPropagation()
                onKeep()
              }}
              aria-pressed={clip.status === 'kept'}
              aria-label={`Keep ${clip.title || 'clip'}`}
              title={clip.status === 'kept' ? 'Un-keep' : 'Keep'}
              className={cx(
                'grid size-7 place-items-center rounded-md transition-colors',
                clip.status === 'kept' ? 'bg-success-soft text-success' : 'text-fg-muted hover:bg-surface hover:text-success',
              )}
            >
              <CheckIcon size={15} />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation()
                onDiscard()
              }}
              aria-label={`${clip.status === 'discarded' ? 'Restore' : 'Drop'} ${clip.title || 'clip'}`}
              title={clip.status === 'discarded' ? 'Restore' : 'Drop'}
              className="grid size-7 place-items-center rounded-md text-fg-muted transition-colors hover:bg-surface hover:text-danger"
            >
              {clip.status === 'discarded' ? <RefreshIcon size={14} /> : <XIcon size={15} />}
            </button>
          </span>
        </div>
      </div>
    </div>
  )
}

function NoClips({ jobId }: { jobId: string }) {
  return (
    <EmptyState
      icon={FilmIcon}
      title="No clips came back"
      description="The AI found nothing that stands on its own — a real answer for some videos. If you expected clips, try a bigger AI model or a wider clip length in Settings, or pick the parts yourself."
      action={
        <>
          <ButtonLink to={`/jobs/${jobId}/pick`} variant="primary" icon={ScissorsIcon}>
            Pick parts myself
          </ButtonLink>
          <ButtonLink to="/settings">Open Settings</ButtonLink>
        </>
      }
    />
  )
}
