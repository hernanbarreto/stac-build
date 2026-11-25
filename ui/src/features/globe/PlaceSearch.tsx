/**
 * THE PLACE SEARCH of the globe and of a work's Location window (USER 2026-11-24: "agrega varias herramientas para
 * buscar ciudad, por ejemplo, como google maps tiene … debe ser mas amigable"): one box for names, addresses and
 * coordinates in any form people paste (places.ts), suggestions as one types, Enter for the first, and the device's
 * own position ("my location": the browser asks the user once).
 */
import { useEffect, useRef, useState } from 'react'
import { Loader2, LocateFixed, MapPin, Navigation } from 'lucide-react'
import { IconButton } from '../../components/ui/IconButton'
import { SearchInput } from '../../components/ui/Field'
import { useT } from '../../i18n'
import { findPlace, formatDms, parseCoordinates, suggestPlaces, type Place } from './places'
import './places.css'

interface Item { text: string; magicKey?: string; place?: Place }

export function PlaceSearch({ onPick, autoFocus = false }: { onPick: (p: Place) => void; autoFocus?: boolean }) {
  const t = useT()
  const [text, setText] = useState('')
  const [items, setItems] = useState<Item[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [open, setOpen] = useState(false)
  const abort = useRef<AbortController | null>(null)

  // as the user types: coordinates are read at once; a name asks the geocoder (a quarter of a second after the last key)
  useEffect(() => {
    abort.current?.abort()
    setError(null)
    const q = text.trim()
    const coords = parseCoordinates(q)
    if (coords) {
      setItems([{ text: t('place.goToCoords', { place: formatDms(coords.lat, coords.lon) }), place: { ...coords, label: formatDms(coords.lat, coords.lon) } }])
      return
    }
    if (q.length < 3) { setItems([]); return }
    const ac = new AbortController()
    abort.current = ac
    const timer = setTimeout(() => {
      suggestPlaces(q, ac.signal)
        .then(s => { if (!ac.signal.aborted) setItems(s.map(x => ({ text: x.text, magicKey: x.magicKey }))) })
        .catch(e => { if (!ac.signal.aborted) setError(t('place.failed', { detail: e instanceof Error ? e.message : String(e) })) })
    }, 250)
    return () => { clearTimeout(timer); ac.abort() }
  }, [text, t])

  const pick = async (item?: Item) => {
    const q = text.trim()
    if (!item && !q) return
    setOpen(false)
    if (item?.place) { onPick(item.place); return }
    setBusy(true); setError(null)
    try {
      const place = await findPlace(item?.text ?? q, item?.magicKey)
      if (place) { onPick(place); setText(place.label) } else setError(t('place.notFound', { q: item?.text ?? q }))
    } catch (e) {
      setError(t('place.failed', { detail: e instanceof Error ? e.message : String(e) }))
    } finally {
      setBusy(false)
    }
  }

  const myLocation = () => {
    if (!navigator.geolocation) { setError(t('place.noGeolocation')); return }
    setBusy(true); setError(null)
    navigator.geolocation.getCurrentPosition(
      pos => {
        setBusy(false)
        const { latitude: lat, longitude: lon, accuracy } = pos.coords
        onPick({ lat, lon, label: t('place.myLocationAt', { place: formatDms(lat, lon), acc: Math.round(accuracy) }) })
      },
      err => { setBusy(false); setError(t('place.geolocationFailed', { detail: err.message })) },
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 0 })
  }

  return (
    <div className="stac-place">
      <div className="stac-place__row">
        <SearchInput size="sm" value={text} autoFocus={autoFocus} placeholder={t('place.placeholder')} aria-label={t('place.placeholder')}
          onChange={e => { setText(e.target.value); setOpen(true) }} onClear={() => { setText(''); setItems([]) }}
          onFocus={() => setOpen(true)}
          onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); void pick(items[0]) } if (e.key === 'Escape') setOpen(false) }} />
        {busy ? <Loader2 className="stac-spin stac-place__busy" aria-hidden />
          : <IconButton size="sm" label={t('place.myLocation')} icon={<LocateFixed aria-hidden />} onClick={myLocation} />}
      </div>
      {open && items.length > 0 && (
        <ul className="stac-place__list" role="listbox" aria-label={t('place.results')}>
          {items.map((it, k) => (
            <li key={`${it.text}-${k}`} role="option" aria-selected={false}>
              <button type="button" onMouseDown={e => e.preventDefault()} onClick={() => void pick(it)}>
                {it.place ? <Navigation aria-hidden /> : <MapPin aria-hidden />} <span>{it.text}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {error && <div className="stac-place__error" role="alert">{error}</div>}
    </div>
  )
}
