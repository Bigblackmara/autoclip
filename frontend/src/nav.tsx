/**
 * The app's navigation — ONE list. The sidebar, the phone drawer and the
 * page titles all read from here.
 *
 * To add a page: register its route in routes.tsx, then add one line here.
 * Entries whose route isn't registered yet are hidden automatically, so an
 * entry can land before its page does (Schedule and Autopilot below).
 */

import {
  CalendarIcon,
  CompassIcon,
  FolderIcon,
  ActivityIcon,
  GamepadIcon,
  PlusIcon,
  RadioIcon,
  RocketIcon,
  SendIcon,
  SettingsIcon,
  type SidebarItem,
} from './ui'

/** Sections, in sidebar order. */
export const NAV_SECTIONS = ['Create', 'Library', 'Modes', 'Publish', 'System'] as const
export type NavSection = (typeof NAV_SECTIONS)[number]

export interface NavItem extends SidebarItem {
  section: NavSection
}

export const NAV_ITEMS: NavItem[] = [
  { label: 'New clips', path: '/', icon: PlusIcon, section: 'Create', end: true },
  { label: 'Explore', path: '/explore', icon: CompassIcon, section: 'Create' },
  { label: 'Projects', path: '/projects', icon: FolderIcon, section: 'Library', activeOn: ['/jobs/*'] },
  { label: 'GTA Mode', path: '/gta', icon: GamepadIcon, section: 'Modes' },
  { label: 'Live Watch', path: '/live', icon: RadioIcon, section: 'Modes' },
  { label: 'Schedule', path: '/schedule', icon: CalendarIcon, section: 'Publish' },
  { label: 'Autopilot', path: '/autopilot', icon: RocketIcon, section: 'Publish' },
  { label: 'Posting & accounts', path: '/posting', icon: SendIcon, section: 'Publish' },
  { label: 'Settings', path: '/settings', icon: SettingsIcon, section: 'System' },
  { label: 'Health check', path: '/health', icon: ActivityIcon, section: 'System' },
]
