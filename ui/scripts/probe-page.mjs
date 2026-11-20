// open one page of the dev server under test, measure which controls are cut inside their boxes, take a picture
import { chromium } from 'playwright'
import path from 'node:path'
const [url, out, w = '1000', h = '700'] = process.argv.slice(2)
const env = { ...process.env, CUDA_VISIBLE_DEVICES: '', __EGL_VENDOR_LIBRARY_DIRS: '/nonexistent' }
const browser = await chromium.launch({ env, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--no-sandbox'] })
const page = await (await browser.newContext({ viewport: { width: +w, height: +h } })).newPage()
const errors = []
page.on('pageerror', e => errors.push(String(e).slice(0, 200)))
page.on('console', m => { if (m.type() === 'error') errors.push(m.text().slice(0, 200)) })
await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 })
await page.waitForTimeout(800)
const res = await page.evaluate(() => [...document.querySelectorAll('[data-box]')].map(box => {
  const b = box.getBoundingClientRect()
  const cols = [...box.querySelectorAll('thead th')].map(th => Math.round(th.getBoundingClientRect().width))
  const btns = [...box.querySelectorAll('tbody tr:first-child button')].map(el => {
    const r = el.getBoundingClientRect()
    const td = el.closest('td').getBoundingClientRect()
    const seen = Math.max(0, Math.min(r.right, td.right, b.right) - Math.max(r.left, td.left, b.left))
    return `${el.getAttribute('aria-label')}: ${Math.round(100 * seen / r.width)} % visible`
  })
  return { box: box.getAttribute('data-box'), columns: cols, buttons: btns }
}))
for (const r of res) console.log(`box ${r.box} px — column widths ${JSON.stringify(r.columns)}\n     ${r.buttons.join(' | ')}`)
if (errors.length) console.log('page errors:', errors.slice(0, 3))
await page.screenshot({ path: path.resolve(out) })
await browser.close()
