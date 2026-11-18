# AutoClip UI — design notes

Light, friendly, easy to find your way around. One accent colour (indigo),
white cards on a very light grey page, soft borders, generous spacing,
plain-language labels. Dark mode is a toggle in the sidebar (all colours are
CSS variables).

## Adding a page (two lines)

1. **Route** — `src/routes.tsx`, add to `pageRoutes` (above the `*` catch-all):

   ```tsx
   { path: 'schedule', element: <Schedule /> },
   ```

2. **Nav entry** — `src/nav.tsx`, add to `NAV_ITEMS` (order = sidebar order):

   ```tsx
   { label: 'Schedule', path: '/schedule', icon: CalendarIcon, section: 'Publish' },
   ```

   Fields: `label`, `path` (query strings allowed), `icon` (from `src/ui/icons.tsx`),
   `section` (`'Create' | 'Library' | 'Modes' | 'Publish' | 'System'`), optional
   `end` (exact match, only for `/`) and `activeOn` (extra patterns that light
   the item up, e.g. `['/jobs/*']`).

   **Entries whose route isn't registered are hidden automatically**
   (`isRouteRegistered` in routes.tsx), so `Schedule` (`/schedule`) and
   `Autopilot` (`/autopilot`) are already in `NAV_ITEMS` and appear as soon as
   their routes are added.

The sidebar (desktop, collapsible), the phone drawer and the active state all
read from `NAV_ITEMS`; nothing else needs touching.

## Page skeleton

```tsx
import { Button, Card, PageHeader, PlusIcon } from '../ui'

export function Schedule() {
  return (
    <div>
      <PageHeader
        title="Schedule"
        description="When your clips go out, platform by platform."   // one plain line
        actions={<Button variant="primary" icon={PlusIcon}>Add a time</Button>}
        // back={{ to: '/projects', label: 'All projects' }}          // detail pages
      >
        {/* optional: <Tabs .../> for sub-navigation */}
      </PageHeader>

      <div className="grid gap-6">
        <Card title="Coming up" description="The next 7 days.">…</Card>
      </div>
    </div>
  )
}
```

The shell already provides page padding and max width — don't add top
padding (`pt-*`) to pages.

## Components (`src/ui`, import from `'../ui'`)

| Component | Use for |
| --- | --- |
| `PageHeader` | Every page: title, one-line description, `back`, `actions`, children (tabs). |
| `Card` | Every block of content. Props: `title`, `description`, `icon`, `actions`, `padding` (`none`/`sm`/`md`/`lg`). Use `padding="none"` for edge-to-edge lists, with rows `px-5 sm:px-6` and a `divide-y divide-line`. |
| `SectionTitle` | A heading for content that isn't inside a card (e.g. a grid of clip cards). |
| `Button` / `ButtonLink` / `IconButton` | `variant`: `primary` (the one main action), `secondary` (default), `ghost` (low emphasis), `danger`. `size`: `sm`/`md`/`lg`. `icon`, `iconRight`, `loading`, `block`. `IconButton` needs a `label`. |
| `Field` + `Input` / `Textarea` / `Select` | Labelled form controls. `Field` props: `label`, `hint` (short helper text — always write one if the label isn't obvious), `error`, `optional`. `Select` takes `options=[{value,label}]`. |
| `Toggle` | On/off settings (a switch). `Checkbox` for picking items from a list. |
| `Segmented` | 2–4 mutually exclusive choices shown at once. |
| `Tabs` + `tabPanelProps` | Sub-navigation. `variant`: `underline` (page level), `pill` (inside a card), `side` (vertical list on desktop, scrolling row on phones — Settings). Keep the selected tab in the URL (`?tab=`) so links and Back work. |
| `Badge` | Status pills. `tone`: `neutral`/`accent`/`success`/`warn`/`danger`/`info`; `dot`, `pulse` for live/running. |
| `Callout` | Tinted notice: `info` tips, `warn` heads-ups, `danger` errors, `success` confirmations. `ErrorNote` (components/) wraps it for API errors with hints. |
| `EmptyState` | "Nothing here yet" + the one action that fixes it. |
| `useToast()` | `toast.success('Saved')`, `toast.error('Couldn't save', detail)` — short confirmations of actions. The same message repeated replaces itself. |
| `Dialog` / `Sheet` | Modal dialog (focus trap, Esc) / side drawer. |
| `Skeleton`, `SkeletonCard`, `PageSkeleton` | Loading placeholders — prefer these over "Loading…". |
| `ProgressBar`, `Spinner` | Progress. |
| `KeyValue` | Label / value lists. |
| icons | `src/ui/icons.tsx` — inline SVG, Lucide-style. Add new ones there (copy Lucide path data). No icon package. |

## Colour tokens (Tailwind classes)

Use the semantic names in new code:

- Surfaces: `bg-page` (app background), `bg-surface` (cards), `bg-subtle` (hover / insets), `bg-hover`
- Lines: `border-line`, `border-line-strong`, `divide-line`
- Text: `text-fg` (main), `text-fg-muted` (secondary), `text-fg-subtle` (hints)
- Accent: `bg-accent`, `text-accent-text`, `bg-accent-soft`, `ring-accent`
- Status: `success`, `warn`, `danger`, `info` — each with a `-soft` background (`bg-success-soft text-success`)
- Media backgrounds: `bg-media`; modal backdrops: `bg-scrim`

The older `ink-*`, `sodium-*` and `signal-*` classes still work (they are
aliases that follow the theme) but shouldn't be used in new code. Utility CSS
classes `.card`, `.field`, `.btn btn-primary|btn-ghost|btn-quiet`, `.link`
exist for markup that can't use the components.

## Writing

The user is not a developer. Labels say what a thing does ("Shortest clip
(seconds)", not "min_duration_s"); every non-obvious control gets a one-line
`hint`; empty states say what to do next; errors say how to fix it when we
know.
