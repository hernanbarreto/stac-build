// The coordinate reader of the globe (src/features/globe/places.ts) on what people paste — USER 2026-11-24: "quise
// agregar este punto 19°23'53.8"N 99°11'58.4"W y me fue imposible". Run from ui/:
//   node_modules/.bin/esbuild scripts/check-places.ts --bundle --platform=node --outfile=/tmp/check-places.cjs && node /tmp/check-places.cjs
import { parseCoordinates, formatDms } from '../src/features/globe/places'
const cases: [string, number | null, number | null][] = [
  [`19°23'53.8"N 99°11'58.4"W`, 19.398278, -99.199556],
  [`19°23'53.8"N, 99°11'58.4"W`, 19.398278, -99.199556],
  [`19° 23' 53.8" N 99° 11' 58.4" O`, 19.398278, -99.199556],
  [`N 19 23 53.8 W 99 11 58.4`, 19.398278, -99.199556],
  [`19°23.8967'N 99°11.9733'W`, 19.398278, -99.199555],
  [`19.398278, -99.199556`, 19.398278, -99.199556],
  [`19.398278 -99.199556`, 19.398278, -99.199556],
  [`19.398278N 99.199556W`, 19.398278, -99.199556],
  [`19,398278; -99,199556`, 19.398278, -99.199556],
  [`-34.6037, -58.3816`, -34.6037, -58.3816],
  [`https://www.google.com/maps/@19.3982778,-99.1995556,17z`, 19.3982778, -99.1995556],
  [`https://www.google.com/maps/place/x/@19.3,-99.1,17z/data=!3m1!4b1!4m6!3m5!3d19.398278!4d-99.199556`, 19.398278, -99.199556],
  [`41°39'24.6"N 0°52'45.9"W`, 41.656833, -0.879417],
  [`Zaragoza`, null, null], [`19°61'00"N 99°00'00"W`, null, null], [`95, 10`, null, null], [``, null, null],
]
let bad = 0
for (const [text, lat, lon] of cases) {
  const r = parseCoordinates(text)
  const ok = lat === null ? r === null : !!r && Math.abs(r.lat - lat) < 2e-6 && Math.abs(r.lon - (lon as number)) < 2e-6
  if (!ok) bad++
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${JSON.stringify(text).slice(0, 70).padEnd(72)} -> ${r ? `${r.lat.toFixed(6)}, ${r.lon.toFixed(6)}` : 'null'}`)
}
console.log('dms:', formatDms(19.398278, -99.199556), '|', formatDms(-34.6037, -58.3816))
console.log(bad ? `${bad} FAILED` : 'all passed')
