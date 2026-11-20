// which launch settings give a SOFTWARE WebGL context in the headless shell (and never touch the card)
import { chromium } from 'playwright'
import { execSync } from 'node:child_process'
import path from 'node:path'
const shell = path.dirname(chromium.executablePath())
const icd = path.join(shell, 'vk_swiftshader_icd.json')
const card = () => { try { return execSync('nvidia-smi --query-compute-apps=pid --format=csv,noheader', { encoding: 'utf8' }).trim().split('\n').filter(Boolean).sort().join(',') } catch { return 'n/a' } }
const hide = { CUDA_VISIBLE_DEVICES: '', __EGL_VENDOR_LIBRARY_DIRS: '/nonexistent' }
const variants = {
  A: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'], env: hide },
  B: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'], env: { ...hide, VK_ICD_FILENAMES: icd, VK_DRIVER_FILES: icd } },
  C: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'], env: hide },
  D: { args: ['--use-angle=swiftshader-webgl', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'], env: hide },
  E: { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--disable-gpu-compositing'], env: hide },
}
for (const [name, v] of Object.entries(variants)) {
  const before = card()
  let line = ''
  try {
    const browser = await chromium.launch({ env: { ...process.env, ...v.env }, args: [...v.args, '--no-sandbox'] })
    const page = await browser.newPage()
    await page.goto('about:blank')
    const r = await page.evaluate(() => {
      const out = {}
      for (const kind of ['webgl2', 'webgl']) {
        const c = document.createElement('canvas')
        const gl = c.getContext(kind)
        if (!gl) { out[kind] = null; continue }
        const ext = gl.getExtension('WEBGL_debug_renderer_info')
        out[kind] = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'context (renderer hidden)'
      }
      return out
    })
    const during = card()
    await browser.close()
    line = `webgl2: ${r.webgl2} | webgl: ${r.webgl} | card ${before === during ? 'unchanged' : `CHANGED ${before} → ${during}`}`
  } catch (e) { line = `failed: ${String(e).slice(0, 120)}` }
  console.log(`${name}  ${v.args.slice(0, 3).join(' ')}${v.env.VK_ICD_FILENAMES ? ' +ICD env' : ''}\n     ${line}`)
}
