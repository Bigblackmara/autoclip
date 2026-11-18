import { YouTubeTab } from '../gta/YouTubeTab'
import { ButtonLink, PageHeader, RadioIcon } from '../ui'

/**
 * Posting & accounts: the YouTube / Instagram connections, posting schedule
 * and upload history (the same panel as GTA Mode → Posting), on its own page
 * so it's one click from the sidebar.
 */
export function Posting() {
  return (
    <div>
      <PageHeader
        title="Posting & accounts"
        description="Connect YouTube and Instagram, set when clips go out, and see what's been posted."
        actions={
          <ButtonLink to="/live?tab=settings" variant="secondary" icon={RadioIcon}>
            Live Watch uploads
          </ButtonLink>
        }
      />
      <YouTubeTab />
    </div>
  )
}
