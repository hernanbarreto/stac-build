#!/usr/bin/env node
/**
 * responsive-shots.mjs — the page at every screen size, with what is CUT measured (USER 2026-11-20: "hace la
 * pagina responsiva para PC tablet y telefonos … muchas cosas no tienen scroll, y quedan ocultas segun la
 * resolucion … debe funcionar perfectamente adaptada a cada pantalla").
 *
 * It needs, already running: the mock backend (node scripts/mock-server.mjs 8790) and a Vite dev server of the
 * tree under test pointed at it (STAC_API_TARGET=http://localhost:8790 vite --port 5199) — NEVER the live backend:
 * opening a session there changes the server's session for everyone. The browser renders in SOFTWARE
 * (SwiftShader) and is checked against nvidia-smi right after it starts: it must never be on the card.
 *
 * For every size × state it writes <out>/<state>__<size>.png and, in <out>/report.json, what a user cannot reach:
 *   cut        every control (button, input, tab, link, slider) partly or wholly outside what its clipping
 *              ancestors or the page show, where no scroll can bring it back
 *   overlaps   pairs of the viewer's overlays (toolbar, view cube, legend, scale block, cards) that intersect
 *   small      on a touch size, how many controls are under 40 px
 *
 * Usage: node scripts/responsive-shots.mjs <outDir> [--page URL] [--mock URL] [--sizes a,b] [--states a,b]
 */
import { chromium } from 'playwright'
import { execSync } from 'node:child_process'
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const UI = path.resolve(__dirname, '..')
const argv = process.argv.slice(2)
const opt = (name, def) => (argv.includes(name) ? argv[argv.indexOf(name) + 1] : def)
const OUT = path.resolve(argv[0] && !argv[0].startsWith('--') ? argv[0] : 'responsive-shots')
const PAGE = opt('--page', 'http://localhost:5199')
const MOCK = opt('--mock', 'http://localhost:8790')
const LANG = opt('--lang', 'es')

// CSS pixels of the PAGE (a browser's own bars already taken off a laptop's screen)
const SIZES = {
  pc1920: { w: 1920, h: 950 },
  pc1366: { w: 1366, h: 625 },              // the user's own PC: 1366 × 768 less the taskbar and the browser's bars
  pc1280: { w: 1280, h: 620 },
  short: { w: 1366, h: 480 },               // a squat window
  tabL: { w: 1024, h: 768, touch: true },
  tabP: { w: 768, h: 1024, touch: true },
  phoneP: { w: 390, h: 844, touch: true },
  phoneL: { w: 844, h: 390, touch: true },
}
const sizeNames = opt('--sizes', Object.keys(SIZES).join(',')).split(',')

const dict = JSON.parse(readFileSync(path.join(UI, `src/i18n/${LANG}.json`), 'utf8'))
const glossaryFile = path.join(UI, 'src/i18n/glossary.json')
const glossary = existsSync(glossaryFile) ? (JSON.parse(readFileSync(glossaryFile, 'utf8'))[LANG] ?? {}) : {}
const FALLBACK = { session: 'sesión', sessions: 'sesiones', Session: 'Sesión', Sessions: 'Sesiones', scan: 'escaneo', scans: 'escaneos', Scan: 'Escaneo', Scans: 'Escaneos', instance: 'instancia', instances: 'instancias', Instance: 'Instancia', Instances: 'Instancias' }
const t = key => key.split('.').reduce((o, k) => (o ? o[k] : undefined), dict) ?? key
const T = (key, vars = {}) => t(key).replace(/\{(@?\w+)\}/g, (m, k) => (k.startsWith('@') ? glossary[k.slice(1)] ?? FALLBACK[k.slice(1)] ?? m : vars[k] ?? m))

const clickIf = async (loc, timeout = 2500) => { try { await loc.click({ timeout }); return true } catch { return false } }
const byLabel = (page, key) => page.getByLabel(T(key), { exact: true }).first()
const setScenario = name => fetch(`${MOCK}/__mock/scenario`, { method: 'POST', body: JSON.stringify({ name }) })

async function openApp(page, { layout = {}, density } = {}) {
  await page.addInitScript(({ layout, density, lang }) => {
    try {
      localStorage.clear()
      localStorage.setItem('stac_token', 'mock-token')
      localStorage.setItem('stac.lang', lang)
      localStorage.setItem('stac.layout.v1', JSON.stringify({ leftTab: 'sessions', leftWidth: 280, inspectorOpen: true, inspectorTab: 'assistant', inspectorWidth: 320, dockOpen: false, dockTab: 'console', dockHeight: 200, ...(density ? { density } : {}), ...layout }))
    } catch { /* storage unavailable */ }
  }, { layout, density, lang: LANG })
  await page.goto(`${PAGE}/`, { waitUntil: 'networkidle', timeout: 60000 })
  // connected = the welcome's Connect button is gone (the page connects by itself or on this click)
  const connect = page.getByRole('button', { name: T('welcome.connect') }).first()
  for (let i = 0; i < 20; i++) {
    if (await page.locator('.stac-live-dot--ok').count() && !(await connect.isVisible().catch(() => false))) break
    await clickIf(connect, 700)
    await page.waitForTimeout(400)
  }
}
// a session is open = the viewer's toolbar is drawn
const loadSession = async page => {
  const load = byLabel(page, 'sessions.load')
  try { await load.waitFor({ state: 'visible', timeout: 10000 }) } catch { return false }
  await clickIf(load, 3000)
  try { await page.locator('.stac-vptoolbar').waitFor({ state: 'visible', timeout: 20000 }) } catch { return false }
  await page.waitForTimeout(2500)                      // the octree's first nodes
  return true
}

// ── the states: what is on screen when the picture is taken ──────────────────────────────────────────────
const STATES = {
  // the frame as a first visit finds it: both side panels open, a session with its cloud
  frame: { scenario: 'cloud', run: async page => { await loadSession(page) } },
  // the viewer alone: both side panels closed
  bare: { scenario: 'cloud', run: async page => {
    await loadSession(page)
    await clickIf(byLabel(page, 'sessions.title'), 1500)
    await clickIf(byLabel(page, 'layout.hideInspector'), 1500)
    await page.waitForTimeout(500)
  } },
  instances: { scenario: 'cloud', run: async page => { await loadSession(page); await clickIf(byLabel(page, 'instances.title')); await page.waitForTimeout(500) } },
  properties: { scenario: 'cloud', layout: { inspectorTab: 'properties' }, run: async page => { await loadSession(page) } },
  // the dock open and the brush's options under the toolbar
  dockbrush: { scenario: 'cloud', layout: { dockOpen: true, dockTab: 'jobs' }, run: async page => {
    await loadSession(page)
    await clickIf(byLabel(page, 'toolbar.brush'))
    await page.waitForTimeout(500)
  } },
  display: { scenario: 'cloud', run: async page => { await loadSession(page); await clickIf(byLabel(page, 'toolbar.display')); await page.waitForTimeout(400) } },
  // the comparison's deviation sheet (its chip in the toolbar, its legend)
  bim: { scenario: 'bim', run: async page => { await loadSession(page); await clickIf(byLabel(page, 'bim.title')); await page.waitForTimeout(500) } },
  // a row of the instances list under the pointer: its actions show
  instanceshover: { scenario: 'cloud', run: async page => {
    await loadSession(page); await clickIf(byLabel(page, 'instances.title')); await page.waitForTimeout(500)
    await page.locator('.stac-tree__row').nth(1).hover({ timeout: 3000 }).catch(() => {})
    await page.waitForTimeout(300)
  }, keepPointer: true },
  // the menus of the header: the longest one, and the one at the right edge
  toolsmenu: { scenario: 'cloud', run: async page => { await loadSession(page); await clickIf(page.getByRole('button', { name: T('menu.tools'), exact: true }).first()); await page.waitForTimeout(300) }, keepPointer: true },
  usermenu: { scenario: 'cloud', run: async page => { await loadSession(page); await clickIf(page.getByRole('button', { name: /hbarreto/ }).first()); await page.waitForTimeout(300) }, keepPointer: true },
  nosession: { scenario: 'sessions', run: async () => {} },
  settings: { scenario: 'sessions', run: async page => { await clickIf(byLabel(page, 'menu.settings')); await page.waitForTimeout(400) } },
  login: { scenario: 'sessions', loggedOut: true, run: async () => {} },
}
const stateNames = opt('--states', 'frame,instances,properties,dockbrush,bim').split(',')

// ── what a user cannot reach, measured in the page ────────────────────────────────────────────────────────
function audit(touch) {
  const vw = window.innerWidth, vh = window.innerHeight
  const name = el => (el.getAttribute('aria-label') || el.getAttribute('title') || (el.innerText || '').trim().split('\n')[0] || el.getAttribute('placeholder') || el.className || el.tagName).toString().slice(0, 48)
  const visible = el => {
    const cs = getComputedStyle(el)
    if (cs.display === 'none' || cs.visibility === 'hidden') return false
    if (el.checkVisibility && !el.checkVisibility({ checkVisibilityCSS: true })) return false
    const r = el.getBoundingClientRect()
    return r.width >= 1 && r.height >= 1
  }
  const cut = []
  let small = 0, controls = 0
  for (const el of document.querySelectorAll('button, a[href], input, select, textarea, [role="tab"], [role="menuitem"], [role="button"], [role="slider"], [role="checkbox"], [role="switch"]')) {
    if (!visible(el)) continue
    controls++
    const r = el.getBoundingClientRect()
    if (touch && Math.min(r.width, r.height) < 40) small++
    let l = r.left, t = r.top, rt = r.right, b = r.bottom
    let lost = null
    let fixed = getComputedStyle(el).position === 'fixed'
    for (let n = el.parentElement; n && !fixed; n = n.parentElement) {
      const s = getComputedStyle(n)
      const cx = s.overflowX !== 'visible', cy = s.overflowY !== 'visible'
      if (cx || cy) {
        const a = n.getBoundingClientRect()
        const outX = cx && (l < a.left - 0.5 || rt > a.right + 0.5)
        const outY = cy && (t < a.top - 0.5 || b > a.bottom + 0.5)
        if (outX || outY) {
          const canX = /auto|scroll/.test(s.overflowX) && n.scrollWidth > n.clientWidth + 1
          const canY = /auto|scroll/.test(s.overflowY) && n.scrollHeight > n.clientHeight + 1
          if ((outX && !canX) || (outY && !canY)) lost = lost || `.${(n.className || n.tagName).toString().trim().split(/\s+/)[0]}`
          if (cx) { l = Math.max(l, a.left); rt = Math.min(rt, a.right) }
          if (cy) { t = Math.max(t, a.top); b = Math.min(b, a.bottom) }
        }
      }
      if (s.position === 'fixed') fixed = true
    }
    if (l < -0.5 || t < -0.5 || rt > vw + 0.5 || b > vh + 0.5) lost = lost || 'the page edge'    // the page cannot scroll
    l = Math.max(l, 0); t = Math.max(t, 0); rt = Math.min(rt, vw); b = Math.min(b, vh)
    const seen = Math.max(0, rt - l) * Math.max(0, b - t)
    const hidden = Math.round(100 * (1 - seen / (r.width * r.height)))
    if (lost && hidden >= 3) cut.push({ what: name(el), hidden, by: lost, at: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)] })
  }
  // the viewer's overlays against each other
  const sel = { toolbar: '.stac-vptoolbar', popover: '.stac-vppopover', cube: '.stac-hud__cube', legend: '.stac-hud__legend', bottomleft: '.stac-hud__bottomleft', banner: '.stac-hud__banner', card: '.stac-vpcard', physics: '.stac-physics, .stac-physics-panel' }
  const boxes = []
  for (const [k, q] of Object.entries(sel)) for (const el of document.querySelectorAll(q)) if (visible(el)) { const r = el.getBoundingClientRect(); boxes.push({ k, l: r.left, t: r.top, r: r.right, b: r.bottom }) }
  const overlaps = []
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const A = boxes[i], B = boxes[j]
    if (A.k === B.k || (A.k === 'toolbar' && B.k === 'popover') || (A.k === 'popover' && B.k === 'toolbar')) continue
    const w = Math.min(A.r, B.r) - Math.max(A.l, B.l), h = Math.min(A.b, B.b) - Math.max(A.t, B.t)
    if (w > 1 && h > 1) overlaps.push(`${A.k} × ${B.k}: ${Math.round(w)} × ${Math.round(h)} px`)
  }
  const box = q => { const el = document.querySelector(q); if (!el) return null; const r = el.getBoundingClientRect(); return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)] }
  return { page: [vw, vh], controls, cut, overlaps, small: touch ? small : undefined, viewer: box('.stac-main__viewport'), toolbar: box('.stac-vptoolbar'), rail: box('.stac-activity'), scrollX: document.documentElement.scrollWidth > vw + 1 }
}

// ── run ───────────────────────────────────────────────────────────────────────────────────────────────────
mkdirSync(OUT, { recursive: true })
// SOFTWARE rendering only (measured 2026-11-20 with scripts/webgl-probe.mjs: the renderer reads "SwiftShader
// Device (Subzero)" and nvidia-smi's process list does not change); the system's EGL vendors are hidden so that
// nothing can reach the card's driver. (Pointing VK_ICD_FILENAMES at the bundled SwiftShader file BREAKS it.)
const env = { ...process.env, CUDA_VISIBLE_DEVICES: '', __EGL_VENDOR_LIBRARY_DIRS: '/nonexistent' }
const cardBefore = (() => { try { return execSync('nvidia-smi --query-compute-apps=pid --format=csv,noheader', { encoding: 'utf8' }).trim().split('\n').filter(Boolean).sort().join(',') } catch { return 'n/a' } })()
const browser = await chromium.launch({ env, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'] })
const cardAfter = (() => { try { return execSync('nvidia-smi --query-compute-apps=pid --format=csv,noheader', { encoding: 'utf8' }).trim().split('\n').filter(Boolean).sort().join(',') } catch { return 'n/a' } })()
if (cardAfter !== cardBefore) { await browser.close(); console.error(`THE CARD'S PROCESS LIST CHANGED WHEN THE BROWSER STARTED (${cardBefore} → ${cardAfter}) — closed, nothing captured`); process.exit(3) }

const report = {}
for (const sn of sizeNames) {
  const size = SIZES[sn]
  if (!size) { console.error(`unknown size ${sn}`); continue }
  for (const st of stateNames) {
    const state = STATES[st]
    if (!state) { console.error(`unknown state ${st}`); continue }
    const ctx = await browser.newContext({ viewport: { width: size.w, height: size.h }, deviceScaleFactor: 1, hasTouch: !!size.touch, isMobile: !!size.touch, locale: 'es-MX' })
    const page = await ctx.newPage()
    const errors = []
    page.on('pageerror', e => errors.push(String(e).slice(0, 160)))
    try {
      await setScenario(state.scenario)
      if (state.loggedOut) {
        await page.addInitScript(lang => { try { localStorage.clear(); localStorage.setItem('stac.lang', lang) } catch { /* */ } }, LANG)
        await page.goto(`${PAGE}/`, { waitUntil: 'networkidle', timeout: 60000 })
      } else {
        await openApp(page, { layout: state.layout ?? {}, density: size.touch ? 'comfortable' : 'compact' })
        await state.run(page)
      }
      if (!state.keepPointer) await page.mouse.move(8, 8)
      await page.waitForTimeout(350)
      const a = await page.evaluate(audit, !!size.touch)
      a.session = await page.locator('.stac-vptoolbar').count() > 0
      await page.screenshot({ path: path.join(OUT, `${st}__${sn}.png`) })
      report[`${st}__${sn}`] = { ...a, errors }
      console.log(`${st.padEnd(11)} ${sn.padEnd(7)} ${String(a.page[0]).padStart(4)}×${String(a.page[1]).padEnd(4)} ${a.session ? 'session' : 'NO SESS'} viewer ${a.viewer ? `${a.viewer[2]}×${a.viewer[3]}` : '—'}  cut ${String(a.cut.length).padStart(2)} of ${a.controls}  overlaps ${a.overlaps.length}${a.small != null ? `  under 40 px: ${a.small}` : ''}${errors.length ? `  page errors: ${errors.length}` : ''}`)
    } catch (e) {
      report[`${st}__${sn}`] = { failed: String(e).slice(0, 300) }
      console.log(`${st.padEnd(11)} ${sn.padEnd(7)} FAILED: ${String(e).slice(0, 140)}`)
    }
    await ctx.close()
  }
}
await browser.close()
writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 1))
const cardEnd = (() => { try { return execSync('nvidia-smi --query-compute-apps=pid --format=csv,noheader', { encoding: 'utf8' }).trim().split('\n').filter(Boolean).sort().join(',') } catch { return 'n/a' } })()
console.log(`card's processes before / with the browser / after: ${cardBefore} / ${cardAfter} / ${cardEnd}`)
