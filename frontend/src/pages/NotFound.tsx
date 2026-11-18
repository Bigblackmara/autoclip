import { ButtonLink, CompassIcon, EmptyState, PlusIcon } from '../ui'

export function NotFound() {
  return (
    <EmptyState
      className="mt-10"
      icon={CompassIcon}
      title="This page doesn't exist"
      description="The link may be old, or the page moved. Pick a page from the menu, or start something new."
      action={
        <ButtonLink to="/" variant="primary" icon={PlusIcon}>
          Make new clips
        </ButtonLink>
      }
    />
  )
}
