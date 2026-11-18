/**
 * Live Watch — watch livestreams, catch their moments, clip them while live.
 *
 * Two views under one header: "Streams & clips" (the operations desk) and
 * "Settings". Both stay mounted while the other is hidden, so an unsaved
 * settings draft survives a look at the clip wall, and the wall doesn't reload
 * when you come back. The view lives in the URL (?tab=settings).
 *
 * The page owns only what crosses panels: stream labels for the clip cards,
 * the judge's bar for the score badges, whether Live Watch is on, and the
 * return from an OAuth consent page (?connected=1).
 */

import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { platformLabel, type LiveSettingsData, type LiveStatus, type LiveStream } from '../live/api'
import { ClipsPanel } from '../live/ClipsPanel'
import { LiveSettingsSection } from '../live/LiveSettingsSection'
import { StreamsPanel } from '../live/StreamsPanel'
import { LiveAnnouncer, Toast, type ToastMessage } from '../live/ui'
import { takePendingConnect } from '../live/UploadsSettings'
import { Badge, Button, Card, PageHeader, RadioIcon, SettingsIcon, Tabs, tabPanelProps } from '../ui'

type View = 'watch' | 'settings'

const VIEWS: readonly { key: View; label: string }[] = [
  { key: 'watch', label: 'Streams & clips' },
  { key: 'settings', label: 'Settings' },
]

const viewFrom = (params: URLSearchParams): View => (params.get('tab') === 'settings' ? 'settings' : 'watch')

export function LiveWatch() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [view, setView] = useState<View>(() => viewFrom(searchParams))
  const [status, setStatus] = useState<LiveStatus | null>(null)
  const [streams, setStreams] = useState<LiveStream[]>([])
  const [removedStream, setRemovedStream] = useState<{ id: string; seq: number } | null>(null)
  const [minScore, setMinScore] = useState(70)
  const [settingsDirty, setSettingsDirty] = useState(false)
  const [streamsRefresh, setStreamsRefresh] = useState(0)
  const [focusEnable, setFocusEnable] = useState(0)
  const [focusUploads, setFocusUploads] = useState(0)
  const [authRefresh, setAuthRefresh] = useState(0)
  const [toast, setToast] = useState<ToastMessage | null>(null)
  const settingsSeen = useRef(false)
  const returnHandled = useRef(false)
  const baseId = useId()

  // Follow the URL when it changes underneath us (back/forward, the top nav).
  useEffect(() => {
    setView(viewFrom(searchParams))
  }, [searchParams])

  const showView = useCallback(
    (next: View) => {
      setView(next)
      setSearchParams(
        (params) => {
          const updated = new URLSearchParams(params)
          if (next === 'settings') updated.set('tab', 'settings')
          else updated.delete('tab')
          return updated
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const openUploadSettings = useCallback(() => {
    showView('settings')
    setFocusUploads((n) => n + 1)
  }, [showView])

  const requestEnable = useCallback(() => {
    showView('settings')
    setFocusEnable((n) => n + 1)
  }, [showView])

  // Back from an OAuth consent page: the server sends us to /live?connected=1
  // on success and to plain /live otherwise. The marker Connect left in
  // sessionStorage says which platform, and that we went at all.
  const checkOAuthReturn = useCallback(
    (connected: string | null) => {
      const platform = takePendingConnect()
      if (connected === null && !platform) return
      const name = platform ? platformLabel(platform) : null
      setToast(
        connected === '1'
          ? { id: Date.now(), tone: 'good', text: `${name ?? 'Your account'} is connected.` }
          : {
              id: Date.now(),
              tone: 'bad',
              text: `${name ?? 'The account'} wasn’t connected — the consent was cancelled or failed. Check that the redirect URI matches exactly, then try again.`,
            },
      )
      setAuthRefresh((n) => n + 1)
      setView('settings')
      setFocusUploads((n) => n + 1)
      setSearchParams(
        (params) => {
          const updated = new URLSearchParams(params)
          updated.delete('connected')
          updated.set('tab', 'settings')
          return updated
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  useEffect(() => {
    if (returnHandled.current) return
    returnHandled.current = true
    checkOAuthReturn(searchParams.get('connected'))
    // Mount-only: this reads the URL we arrived with.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Browser Back from a consent page can restore this page from the
  // back/forward cache without remounting; check the marker then too.
  useEffect(() => {
    const onPageShow = (event: PageTransitionEvent) => {
      if (event.persisted) checkOAuthReturn(new URLSearchParams(window.location.search).get('connected'))
    }
    window.addEventListener('pageshow', onPageShow)
    return () => window.removeEventListener('pageshow', onPageShow)
  }, [checkOAuthReturn])

  const handleStreams = useCallback((list: LiveStream[], complete: boolean) => {
    // While Live Watch is off a server may list fewer streams than it holds;
    // keep the labels we already know rather than forgetting them.
    setStreams((current) => {
      if (complete) return list
      const byId = new Map(current.map((stream) => [stream.id, stream]))
      for (const stream of list) byId.set(stream.id, stream)
      return [...byId.values()]
    })
  }, [])

  const handleRemoved = useCallback((id: string) => {
    setStreams((current) => current.filter((stream) => stream.id !== id))
    setRemovedStream((previous) => ({ id, seq: (previous?.seq ?? 0) + 1 }))
  }, [])

  const handleSettings = useCallback((saved: LiveSettingsData) => {
    setMinScore(saved.rate_limit.judge_min_score)
    // After a save (not the first load), the streams panel re-reads status at
    // once — switching Live Watch on or off shows up without waiting a poll.
    if (settingsSeen.current) setStreamsRefresh((n) => n + 1)
    settingsSeen.current = true
  }, [])

  const dismissToast = useCallback(() => setToast(null), [])

  const liveBadge = !status ? null : !status.enabled ? (
    <Badge>Off</Badge>
  ) : status.running === false ? (
    <Badge tone="warn" dot>
      On · not running
    </Badge>
  ) : (
    <Badge tone="success" dot pulse>
      On
    </Badge>
  )

  return (
    <div>
      <LiveAnnouncer />

      <PageHeader
        eyebrow={liveBadge}
        title="Live Watch"
        description="Watch livestreams around the clock. When chat and audio spike and the judge agrees, the moment becomes a captioned vertical clip while the stream is still going."
        actions={
          <Button icon={SettingsIcon} onClick={openUploadSettings}>
            Rights &amp; uploads
          </Button>
        }
      >
        <Tabs
          label="Live Watch"
          idBase={baseId}
          items={VIEWS.map((item) => ({
            key: item.key,
            label: item.label,
            icon: item.key === 'watch' ? RadioIcon : SettingsIcon,
            badge:
              item.key === 'settings' && settingsDirty ? (
                <>
                  <span aria-hidden className="size-1.5 rounded-full bg-accent" />
                  <span className="sr-only">(unsaved changes)</span>
                </>
              ) : undefined,
          }))}
          value={view}
          onChange={showView}
        />
      </PageHeader>

      <p className="-mt-2 mb-6 text-xs leading-relaxed text-fg-muted">
        Clip only streams you have the rights or permission to repost.
      </p>

      <div {...tabPanelProps(baseId, 'watch')} hidden={view !== 'watch'}>
        <Card padding="md">
          <StreamsPanel
            onStreamsChange={handleStreams}
            onStreamRemoved={handleRemoved}
            onStatusChange={setStatus}
            onRequestEnable={requestEnable}
            refreshSignal={streamsRefresh}
          />
        </Card>
        <Card padding="md" className="mt-6">
          <ClipsPanel
            streams={streams}
            removedStream={removedStream}
            minScore={minScore}
            liveEnabled={status ? status.enabled : null}
            onOpenUploadSettings={openUploadSettings}
          />
        </Card>
      </div>

      <div {...tabPanelProps(baseId, 'settings')} hidden={view !== 'settings'}>
        <LiveSettingsSection
          onSettingsChange={handleSettings}
          onDirtyChange={setSettingsDirty}
          focusEnableSignal={focusEnable}
          focusUploadsSignal={focusUploads}
          authRefreshSignal={authRefresh}
        />
      </div>

      <Toast toast={toast} onDismiss={dismissToast} />
    </div>
  )
}
