import { ApiError } from '../api'
import { Callout } from '../ui'

/**
 * A failure and, where the server offered one, the fix.
 *
 * The hint carries most of the value: "YouTube blocked this download" is a
 * dead end, while "set a browser to pull cookies from" is something the user
 * can act on without leaving the app.
 */
export function ErrorNote({
  error,
  onDismiss,
}: {
  error: Error
  onDismiss?: () => void
}) {
  const hint = error instanceof ApiError ? error.hint : ''

  return (
    <Callout tone="danger" role="alert" onDismiss={onDismiss}>
      {/* Job and provider errors are often several lines long and carry
          unbroken URLs or JSON; keep the line breaks and wrap anywhere. */}
      <p className="whitespace-pre-line font-medium text-fg">{error.message}</p>
      {hint && <p className="mt-1.5 max-w-prose whitespace-pre-line text-fg-muted">{hint}</p>}
    </Callout>
  )
}
