import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'
const KEY = 'autoclip.theme'

function read(): Theme {
  try {
    return localStorage.getItem(KEY) === 'dark' ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme
}

/** Light by default; the choice is remembered per browser. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(read)
  useEffect(() => apply(theme), [theme])
  const toggle = useCallback(() => {
    setTheme((current) => {
      const next = current === 'dark' ? 'light' : 'dark'
      try {
        localStorage.setItem(KEY, next)
      } catch {
        /* private mode: the toggle still works for this visit */
      }
      return next
    })
  }, [])
  return [theme, toggle]
}
