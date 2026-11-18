import { useCallback, useEffect, useState } from 'react'

import { errorText, publishingApi, type CheckStatus, type DoctorReport } from '../publishing/api'
import {
  AlertCircleIcon,
  AlertIcon,
  Badge,
  Button,
  Callout,
  Card,
  CheckCircleIcon,
  RefreshIcon,
  SkeletonCard,
  SlidersIcon,
  cx,
  type Tone,
} from '../ui'

const LOOK: Record<CheckStatus, { text: string; tone: Tone; icon: typeof CheckCircleIcon; color: string }> = {
  ok: { text: 'OK', tone: 'success', icon: CheckCircleIcon, color: 'text-success' },
  warn: { text: 'Worth a look', tone: 'warn', icon: AlertIcon, color: 'text-warn' },
  fail: { text: 'Needs fixing', tone: 'danger', icon: AlertCircleIcon, color: 'text-danger' },
}

/**
 * The same checks as `autoclip doctor`: video tools, downloads, AI, secrets,
 * posting connections, disk, database, GPU. "Test connections" also asks the
 * AI provider and Zernio (free calls; nothing is posted).
 */
export function HealthPanel() {
  const [report, setReport] = useState<DoctorReport | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const load = useCallback(async (live = false, refresh = false) => {
    setBusy(true)
    setError('')
    try {
      setReport(await publishingApi.doctor(live, refresh))
    } catch (err) {
      setError(errorText(err))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const groups = new Map<string, DoctorReport['checks']>()
  for (const check of report?.checks ?? []) {
    groups.set(check.group, [...(groups.get(check.group) ?? []), check])
  }

  const summary = report?.summary
  const look = summary ? LOOK[summary.status] : null
  const SummaryIcon = look?.icon

  return (
    <div className="space-y-6">
      <Card padding="md">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            {SummaryIcon && look ? (
              <span
                className={cx(
                  'grid size-11 shrink-0 place-items-center rounded-2xl',
                  summary?.status === 'ok' ? 'bg-success-soft' : summary?.status === 'warn' ? 'bg-warn-soft' : 'bg-danger-soft',
                  look.color,
                )}
              >
                <SummaryIcon size={22} />
              </span>
            ) : (
              <span className="size-11 shrink-0 rounded-2xl bg-subtle" />
            )}
            <div>
              <p className="text-base font-semibold text-fg">
                {!summary
                  ? 'Checking…'
                  : summary.status === 'ok'
                    ? 'Everything looks good'
                    : summary.status === 'warn'
                      ? 'Working, with a few things worth a look'
                      : 'Something needs fixing'}
              </p>
              {summary && (
                <p className="numeric mt-0.5 text-sm text-fg-muted">
                  {summary.ok} OK · {summary.warn} to check · {summary.fail} to fix
                  {report?.live && ' · connections tested'}
                </p>
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button icon={RefreshIcon} loading={busy} onClick={() => load(false, true)}>
              Check again
            </Button>
            <Button
              icon={SlidersIcon}
              disabled={busy}
              onClick={() => load(true)}
              title="Also asks the AI provider and Zernio if your keys work. Nothing is posted."
            >
              Test connections
            </Button>
          </div>
        </div>
      </Card>

      {error && <Callout tone="danger">{error}</Callout>}
      {!report && !error && (
        <div className="grid gap-6">
          <SkeletonCard lines={4} />
          <SkeletonCard lines={3} />
        </div>
      )}

      {[...groups.entries()].map(([group, checks]) => (
        <Card key={group} padding="none" title={group}>
          <ul className="divide-y divide-line border-t border-line">
            {checks.map((check) => {
              const l = LOOK[check.status]
              const Icon = l.icon
              return (
                <li key={check.id} className="flex gap-3 px-5 py-3 sm:px-6">
                  <Icon size={18} className={cx('mt-0.5', l.color)} />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
                      <p className="text-sm font-semibold text-fg">{check.label}</p>
                      {check.status !== 'ok' && <Badge tone={l.tone}>{l.text}</Badge>}
                    </div>
                    {check.detail && (
                      <p className="mt-0.5 text-sm text-fg-muted [overflow-wrap:anywhere]">{check.detail}</p>
                    )}
                    {check.fix && check.status !== 'ok' && (
                      <p className="mt-1.5 rounded-lg bg-subtle px-3 py-2 text-sm text-fg [overflow-wrap:anywhere]">
                        <span className="font-semibold">How to fix: </span>
                        {check.fix}
                      </p>
                    )}
                  </div>
                </li>
              )
            })}
          </ul>
        </Card>
      ))}
    </div>
  )
}
