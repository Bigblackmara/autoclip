import { useCallback, useEffect, useState } from 'react'

import { errorText, publishingApi, type CheckStatus, type DoctorReport } from '../publishing/api'

const BADGE: Record<CheckStatus, { text: string; tone: string }> = {
  ok: { text: 'OK', tone: 'text-signal-good' },
  warn: { text: 'Check', tone: 'text-sodium-400' },
  fail: { text: 'Fix', tone: 'text-signal-bad' },
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

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-baseline gap-4">
        {report && (
          <p className="text-sm text-ink-300">
            <span className={BADGE[report.summary.status].tone}>
              {report.summary.status === 'ok'
                ? 'Everything looks good.'
                : report.summary.status === 'warn'
                  ? 'Working, with things worth a look.'
                  : 'Something needs fixing.'}
            </span>{' '}
            <span className="numeric text-ink-500">
              {report.summary.ok} ok · {report.summary.warn} to check · {report.summary.fail} to fix
            </span>
          </p>
        )}
        <div className="ml-auto flex gap-2">
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => load(false, true)}>
            Check again
          </button>
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => load(true)}>
            Test connections
          </button>
        </div>
      </div>
      {busy && !report && <p className="text-sm text-ink-400">Checking…</p>}
      {error && <p className="text-sm text-signal-bad">{error}</p>}
      {[...groups.entries()].map(([group, checks]) => (
        <div key={group}>
          <h3 className="eyebrow border-b border-ink-800 pb-2">{group}</h3>
          <ul className="divide-y divide-ink-800">
            {checks.map((check) => (
              <li key={check.id} className="grid grid-cols-[4rem_1fr] gap-x-4 py-2.5 sm:grid-cols-[4rem_14rem_1fr]">
                <span className={`text-xs font-semibold uppercase ${BADGE[check.status].tone}`}>
                  {BADGE[check.status].text}
                </span>
                <span className="text-sm text-ink-100">{check.label}</span>
                <span className="col-start-2 text-sm text-ink-400 sm:col-start-3">
                  {check.detail}
                  {check.fix && check.status !== 'ok' && (
                    <span className="mt-0.5 block text-ink-200">{check.fix}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  )
}
