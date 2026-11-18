/**
 * Inline SVG icons (no icon dependency). 24×24 grid, 1.8 stroke, round caps —
 * drawn in the same spirit as Lucide so new ones are easy to add: copy a
 * Lucide icon's <path>s into `paths` below.
 */

import { useId, type SVGProps } from 'react'

export type IconProps = SVGProps<SVGSVGElement> & { size?: number }
export type IconComponent = (props: IconProps) => JSX.Element

function make(name: string, body: JSX.Element): IconComponent {
  const Icon = ({ size = 18, className = '', ...rest }: IconProps) => (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      focusable="false"
      className={`shrink-0 ${className}`}
      {...rest}
    >
      {body}
    </svg>
  )
  Icon.displayName = `${name}Icon`
  return Icon
}

export const PlusIcon = make('Plus', <path d="M12 5v14M5 12h14" />)
export const CompassIcon = make(
  'Compass',
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="m15.5 8.5-2 5-5 2 2-5z" />
  </>,
)
export const FolderIcon = make(
  'Folder',
  <path d="M3 7.5A1.5 1.5 0 0 1 4.5 6h4l2 2h9A1.5 1.5 0 0 1 21 9.5v8a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 17.5z" />,
)
export const FilmIcon = make(
  'Film',
  <>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4" />
  </>,
)
export const GamepadIcon = make(
  'Gamepad',
  <>
    <path d="M6 11h4M8 9v4M15 12h.01M18 10h.01" />
    <path d="M17.3 5H6.7a4 4 0 0 0-3.96 3.43l-.8 5.6A2.6 2.6 0 0 0 4.5 17c.8 0 1.5-.4 2-1l1.4-1.6a2 2 0 0 1 1.5-.7h5.2a2 2 0 0 1 1.5.7L17.5 16c.5.6 1.2 1 2 1a2.6 2.6 0 0 0 2.56-2.97l-.8-5.6A4 4 0 0 0 17.3 5Z" />
  </>,
)
export const RadioIcon = make(
  'Radio',
  <>
    <circle cx="12" cy="12" r="2" />
    <path d="M16.2 7.8a6 6 0 0 1 0 8.4M7.8 16.2a6 6 0 0 1 0-8.4M19.1 4.9a10 10 0 0 1 0 14.2M4.9 19.1a10 10 0 0 1 0-14.2" />
  </>,
)
export const CalendarIcon = make(
  'Calendar',
  <>
    <rect x="3" y="5" width="18" height="16" rx="2" />
    <path d="M16 3v4M8 3v4M3 10h18" />
  </>,
)
export const RocketIcon = make(
  'Rocket',
  <>
    <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" />
    <path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
    <path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5" />
  </>,
)
export const SendIcon = make(
  'Send',
  <>
    <path d="M22 2 11 13" />
    <path d="m22 2-7 20-4-9-9-4z" />
  </>,
)
export const SettingsIcon = make(
  'Settings',
  <>
    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
    <circle cx="12" cy="12" r="3" />
  </>,
)
export const ChevronLeftIcon = make('ChevronLeft', <path d="m15 18-6-6 6-6" />)
export const ChevronRightIcon = make('ChevronRight', <path d="m9 18 6-6-6-6" />)
export const ChevronDownIcon = make('ChevronDown', <path d="m6 9 6 6 6-6" />)
export const ArrowLeftIcon = make('ArrowLeft', <path d="M19 12H5M12 19l-7-7 7-7" />)
export const ArrowRightIcon = make('ArrowRight', <path d="M5 12h14M12 5l7 7-7 7" />)
export const MenuIcon = make('Menu', <path d="M4 6h16M4 12h16M4 18h16" />)
export const XIcon = make('X', <path d="M18 6 6 18M6 6l12 12" />)
export const SunIcon = make(
  'Sun',
  <>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
  </>,
)
export const MoonIcon = make('Moon', <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />)
export const LinkIcon = make(
  'Link',
  <>
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
  </>,
)
export const UploadIcon = make(
  'Upload',
  <>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <path d="m17 8-5-5-5 5M12 3v12" />
  </>,
)
export const CloudIcon = make('Cloud', <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" />)
export const SearchIcon = make(
  'Search',
  <>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </>,
)
export const CheckIcon = make('Check', <path d="M20 6 9 17l-5-5" />)
export const CheckCircleIcon = make(
  'CheckCircle',
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="m8.5 12 2.5 2.5 4.5-5" />
  </>,
)
export const AlertIcon = make(
  'Alert',
  <>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4M12 17h.01" />
  </>,
)
export const AlertCircleIcon = make(
  'AlertCircle',
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v4M12 16h.01" />
  </>,
)
export const InfoIcon = make(
  'Info',
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 16v-4M12 8h.01" />
  </>,
)
export const SparklesIcon = make(
  'Sparkles',
  <>
    <path d="M9.94 15.5A2 2 0 0 0 8.5 14.06l-6.13-1.58a.5.5 0 0 1 0-.96L8.5 9.94A2 2 0 0 0 9.94 8.5l1.58-6.13a.5.5 0 0 1 .96 0L14.06 8.5A2 2 0 0 0 15.5 9.94l6.13 1.58a.5.5 0 0 1 0 .96L15.5 14.06a2 2 0 0 0-1.44 1.44l-1.58 6.13a.5.5 0 0 1-.96 0z" />
    <path d="M20 3v4M22 5h-4" />
  </>,
)
export const PlayIcon = make('Play', <path d="M7 4.5v15a.5.5 0 0 0 .77.42l11.5-7.5a.5.5 0 0 0 0-.84L7.77 4.08A.5.5 0 0 0 7 4.5Z" />)
export const DownloadIcon = make(
  'Download',
  <>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <path d="m7 10 5 5 5-5M12 15V3" />
  </>,
)
export const TrashIcon = make(
  'Trash',
  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />,
)
export const CopyIcon = make(
  'Copy',
  <>
    <rect x="9" y="9" width="12" height="12" rx="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </>,
)
export const RefreshIcon = make(
  'Refresh',
  <>
    <path d="M3 12a9 9 0 0 1 15.5-6.2L21 8" />
    <path d="M21 3v5h-5M21 12a9 9 0 0 1-15.5 6.2L3 16" />
    <path d="M3 21v-5h5" />
  </>,
)
export const PanelLeftIcon = make(
  'PanelLeft',
  <>
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M9 3v18" />
  </>,
)
export const CpuIcon = make(
  'Cpu',
  <>
    <rect x="5" y="5" width="14" height="14" rx="2" />
    <rect x="9" y="9" width="6" height="6" rx="1" />
    <path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3" />
  </>,
)
export const ClockIcon = make(
  'Clock',
  <>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3 2" />
  </>,
)
export const ScissorsIcon = make(
  'Scissors',
  <>
    <circle cx="6" cy="6" r="3" />
    <circle cx="6" cy="18" r="3" />
    <path d="M20 4 8.12 15.88M14.47 14.48 20 20M8.12 8.12 12 12" />
  </>,
)
export const KeyIcon = make(
  'Key',
  <>
    <circle cx="7.5" cy="15.5" r="4.5" />
    <path d="m21 2-9.6 9.6M15.5 7.5l3 3L22 7l-3-3" />
  </>,
)
export const CaptionsIcon = make(
  'Captions',
  <>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <path d="M7 15h4M15 15h2M7 11h2M13 11h4" />
  </>,
)
export const MusicIcon = make(
  'Music',
  <>
    <path d="M9 18V5l12-2v13" />
    <circle cx="6" cy="18" r="3" />
    <circle cx="18" cy="16" r="3" />
  </>,
)
export const StampIcon = make(
  'Stamp',
  <>
    <path d="M5 22h14M19.27 13.73A2.5 2.5 0 0 0 17.5 13h-11A2.5 2.5 0 0 0 4 15.5V17a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-1.5c0-.66-.26-1.3-.73-1.77Z" />
    <path d="M14 13V8.5C14 7 15 7 15 5a3 3 0 0 0-6 0c0 2 1 2 1 3.5V13" />
  </>,
)
export const HardDriveIcon = make(
  'HardDrive',
  <>
    <path d="M22 12H2M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11Z" />
    <path d="M6 16h.01M10 16h.01" />
  </>,
)
export const UsersIcon = make(
  'Users',
  <>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
  </>,
)
export const ExternalIcon = make(
  'External',
  <path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />,
)
export const ActivityIcon = make('Activity', <path d="M22 12h-4l-3 9L9 3l-3 9H2" />)
export const SlidersIcon = make(
  'Sliders',
  <path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6" />,
)
export const LayersIcon = make(
  'Layers',
  <>
    <path d="m12 2 10 5-10 5L2 7z" />
    <path d="m2 17 10 5 10-5M2 12l10 5 10-5" />
  </>,
)
export const TrendingIcon = make('Trending', <path d="m22 7-8.5 8.5-5-5L2 17M16 7h6v6" />)
export const BoltIcon = make('Bolt', <path d="M13 2 3 14h9l-1 8 10-12h-9z" />)
export function LogoMark({ size = 28 }: { size?: number }) {
  // A per-instance gradient id: the same id twice in a page resolves to the
  // first one, which may sit inside a hidden sidebar and then paints nothing.
  const id = `ac-logo-${useId().replace(/:/g, '')}`
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden className="shrink-0">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#6366f1" />
          <stop offset="1" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8" fill={`url(#${id})`} />
      <path d="M12.5 9.5v13l10-6.5z" fill="#fff" />
    </svg>
  )
}
