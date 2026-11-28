#!/usr/bin/env node
/**
 * STAC Build — an IFC as 3D Tiles for the viewer (USER 2026-11-28: "se pone densa la pagina al mostrar el ifc y ademas
 * aun luego de apagar el ifc queda lento, como lo podemos manejar?, tipo los tiles del glb?" / "si, mejora todo" /
 * "las soluciones siempre deben ser completas y de fondo, no parches").
 *
 * WHY. The viewer built EVERY triangle of an IFC at once (grupo_electrogeno's: 336 parts, 7.54 M triangles) and kept
 * them all; a tileset is loaded by the camera — near, each element at full detail; far, simplified — like the GenRecon
 * mesh's (tile_mesh.mjs, whose tiling this uses).
 *
 * WHAT. The IFC read by the page's own reader (ui/scripts/ifc-parts.ts → ifcParse.readIfc: the same parts, names,
 * colours and tree), each part a primitive in the model's frame (web-ifc's Y up, metres) carrying its identity in its
 * extras — { name, expressID, ifc_type, ifc_name, color } — so every tile that holds it says which element it is (the
 * viewer selects, hides and fades by it). A part over SPLIT_TRIANGLES is cut into spatial pieces (median splits of its
 * triangles' centroids along the longest axis) so that a big element — a floor — is not one huge leaf: each piece is a
 * primitive with the part's extras, and the tiler LOCKS every primitive's border, so pieces never open between them.
 * Beside the tiles: elements.json — the parts' list and the spatial tree (the BIM navigator's).
 *
 * Usage: node ifc_tiles.mjs <file.ifc> <out dir> [--jobs N]
 *   built in <out dir>.part and renamed at the end; its read of the IFC in <out dir>.work (removed at the end).
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, realpathSync } from 'node:fs';
import { mkdir, readFile, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { tileItems } from './tile_mesh.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const UI = path.join(REPO, 'ui');
const SPLIT_TRIANGLES = 50000;

/** a file's sha256, streamed (an IFC can be over the 2 GiB a single read allows) */
function sha256Of(file) {
  return new Promise((resolve, reject) => {
    const h = createHash('sha256');
    createReadStream(file).on('data', d => h.update(d)).on('error', reject).on('end', () => resolve(h.digest('hex')));
  });
}

const log = (msg) => console.log(`${new Date().toISOString().slice(11, 19)} [ifc_tiles] ${msg}`);

function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: ['ignore', 'inherit', 'inherit'], ...opts });
    p.on('error', reject);
    p.on('close', code => (code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} ${args.join(' ')} → exit ${code}`))));
  });
}

/**
 * A part's vertices joined where they are the same point (the float32 bits of x, y, z), its triangles renumbered, those
 * left with two corners on one point dropped. web-ifc hands every triangle its OWN three vertices (its flat normals):
 * a surface of loose triangles has a border at every edge, and the simplifier, which locks borders, could take
 * nothing out — every level of the first tiles of grupo_electrogeno's IFC held all its 7.54 M triangles (1.17 GB).
 * The normals are not carried: the viewer shades the elements flat, from the surface itself, as they were shown.
 */
export function weld(pos, idx) {
  const nv = pos.length / 3;
  const bits = new Uint32Array(pos.buffer, pos.byteOffset, pos.length);
  let cap = 1;
  while (cap < nv * 2) cap *= 2;
  const table = new Int32Array(cap).fill(-1);
  const out = new Float32Array(pos.length), outBits = new Uint32Array(out.buffer);
  const remap = new Uint32Array(nv);
  let n = 0;
  for (let v = 0; v < nv; v++) {
    const a = bits[3 * v], b = bits[3 * v + 1], c = bits[3 * v + 2];
    let slot = ((Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca77) ^ Math.imul(c, 0xc2b2ae3d)) >>> 0) & (cap - 1);
    for (;;) {
      const w = table[slot];
      if (w < 0) {
        table[slot] = n;
        outBits[3 * n] = a; outBits[3 * n + 1] = b; outBits[3 * n + 2] = c;
        remap[v] = n++;
        break;
      }
      if (outBits[3 * w] === a && outBits[3 * w + 1] === b && outBits[3 * w + 2] === c) { remap[v] = w; break; }
      slot = (slot + 1) & (cap - 1);
    }
  }
  const tri = new Uint32Array(idx.length);
  let m = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const i0 = remap[idx[t]], i1 = remap[idx[t + 1]], i2 = remap[idx[t + 2]];
    if (i0 === i1 || i1 === i2 || i0 === i2) continue;
    tri[m++] = i0; tri[m++] = i1; tri[m++] = i2;
  }
  return { pos: out.slice(0, 3 * n), idx: tri.slice(0, m) };
}

/** a part's triangles as items for the tiler: whole, or cut in spatial pieces of at most SPLIT_TRIANGLES */
export function itemsOf(meta, rawPos, rawIdx, startIndex) {
  const { pos, idx } = weld(rawPos, rawIdx);
  const nt = idx.length / 3;
  if (nt === 0) return [];
  const cen = new Float32Array(nt * 3);
  for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
    cen[3 * t + k] = (pos[3 * idx[3 * t] + k] + pos[3 * idx[3 * t + 1] + k] + pos[3 * idx[3 * t + 2] + k]) / 3;
  }
  const groups = [];
  const cut = (tris) => {
    if (tris.length <= SPLIT_TRIANGLES) { groups.push(tris); return; }
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const t of tris) for (let k = 0; k < 3; k++) { const c = cen[3 * t + k]; if (c < lo[k]) lo[k] = c; if (c > hi[k]) hi[k] = c; }
    const ax = [0, 1, 2].reduce((b, k) => (hi[k] - lo[k] > hi[b] - lo[b] ? k : b), 0);
    // the median centroid by the engine's own numeric sort (a comparator in JavaScript takes minutes on millions);
    // below it one side, above it the other, the ties shared in the triangles' order — halves of equal size
    const keys = new Float64Array(tris.length);
    for (let i = 0; i < tris.length; i++) keys[i] = cen[3 * tris[i] + ax];
    const mid = tris.length >> 1;
    const median = Float64Array.from(keys).sort()[mid];
    let below = 0;
    for (let i = 0; i < keys.length; i++) if (keys[i] < median) below += 1;
    let ties = mid - below;
    const a = new Uint32Array(mid), b = new Uint32Array(tris.length - mid);
    let na = 0, nb = 0;
    for (let i = 0; i < tris.length; i++) {
      if (keys[i] < median || (keys[i] === median && ties-- > 0)) a[na++] = tris[i]; else b[nb++] = tris[i];
    }
    cut(a); cut(b);
  };
  cut(Uint32Array.from({ length: nt }, (_, t) => t));
  const [r, g, b] = meta.color;
  const material = { name: `ifc_${meta.expressID}`, baseColorFactor: [r, g, b, 1], metallicFactor: 0, roughnessFactor: 0.6,
                     doubleSided: true, alphaMode: 'OPAQUE', alphaCutoff: 0.5, images: {} };
  const extras = { name: meta.name, expressID: meta.expressID, ifc_type: meta.ifcType, ifc_name: meta.ifcName, color: meta.color };
  return groups.map((tris, gi) => {
    const remap = new Map();
    const oi = new Uint32Array(tris.length * 3);
    for (let i = 0; i < tris.length; i++) for (let c = 0; c < 3; c++) {
      const v = idx[3 * tris[i] + c];
      let w = remap.get(v);
      if (w === undefined) { w = remap.size; remap.set(v, w); }
      oi[3 * i + c] = w;
    }
    const p = new Float32Array(remap.size * 3);
    for (const [v, w] of remap) p.set(pos.subarray(3 * v, 3 * v + 3), 3 * w);
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < remap.size; v++) for (let k = 0; k < 3; k++) {
      const x = p[3 * v + k]; if (x < min[k]) min[k] = x; if (x > max[k]) max[k] = x;
    }
    let area = 0;
    for (let t = 0; t < oi.length; t += 3) {
      const i0 = oi[t], i1 = oi[t + 1], i2 = oi[t + 2];
      const ux = p[3 * i1] - p[3 * i0], uy = p[3 * i1 + 1] - p[3 * i0 + 1], uz = p[3 * i1 + 2] - p[3 * i0 + 2];
      const vx = p[3 * i2] - p[3 * i0], vy = p[3 * i2 + 1] - p[3 * i0 + 1], vz = p[3 * i2 + 2] - p[3 * i0 + 2];
      area += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    }
    return { index: startIndex + gi, pos: p, nrm: null, uv: null, idx: oi, min, max, area, uvArea: 0, material,
             extras: { ...extras, piece: gi, pieces: groups.length }, centroid: [0, 1, 2].map(k => 0.5 * (min[k] + max[k])) };
  });
}

async function main() {
  const argv = process.argv.slice(2);
  const pos = [];
  let jobs = 4;
  for (let i = 0; i < argv.length; i++) { if (argv[i] === '--jobs') jobs = parseInt(argv[++i], 10); else pos.push(argv[i]); }
  if (pos.length !== 2 || !(jobs > 0)) { console.error('usage: node ifc_tiles.mjs <file.ifc> <out dir> [--jobs N]'); process.exit(2); }
  const [ifc, outDir] = pos.map(p => path.resolve(p));
  const t0 = Date.now();
  // the IFC's identity as it was read: the server compares it with the file's own to know the tiles are its
  const src = await stat(ifc, { bigint: true });
  const work = `${outDir}.work`;
  await rm(work, { recursive: true, force: true });
  await mkdir(work, { recursive: true });
  // 1. the page's reader, bundled for node (web-ifc from the interface's own node_modules)
  const bundle = path.join(work, 'ifc-parts.cjs');
  await run(path.join(UI, 'node_modules', '.bin', 'esbuild'), [path.join(UI, 'scripts', 'ifc-parts.ts'), '--bundle',
    '--platform=node', '--format=cjs', '--external:web-ifc', '--log-level=warning', `--outfile=${bundle}`]);
  const prefix = path.join(work, 'parts');
  await run(process.execPath, [bundle, ifc, prefix], { env: { ...process.env, NODE_PATH: path.join(UI, 'node_modules') } });
  // 2. its parts as the tiler's items
  const head = JSON.parse(await readFile(`${prefix}.json`, 'utf8'));
  const bin = await readFile(`${prefix}.bin`);
  const items = [];
  let split = 0;
  for (const meta of head.parts) {
    const nv = meta.vertices, ni = meta.indices;
    if (ni < 3) continue;
    const pos = new Float32Array(bin.buffer.slice(bin.byteOffset + meta.offset, bin.byteOffset + meta.offset + nv * 12));
    const idx = new Uint32Array(bin.buffer.slice(bin.byteOffset + meta.offset + nv * 24, bin.byteOffset + meta.offset + nv * 24 + ni * 4));
    const its = itemsOf(meta, pos, idx, items.length);
    if (its.length > 1) split += 1;
    items.push(...its);
  }
  if (!items.length) throw new Error(`${ifc}: no element with a triangle`);
  const tris = items.reduce((s, it) => s + it.idx.length / 3, 0);
  log(`${path.basename(ifc)}: ${head.parts.length} parts, ${tris.toLocaleString('en')} triangles → ${items.length} primitives `
      + `(${split} part(s) cut in pieces of ≤ ${SPLIT_TRIANGLES.toLocaleString('en')} triangles)`);
  const elements = { source: path.basename(ifc), source_bytes: Number(src.size),
                     parts: head.parts.map(({ name, expressID, ifcType, ifcName, color }) => ({ name, expressID, ifcType, ifcName, color })),
                     hierarchy: head.hierarchy };
  await tileItems(items, outDir, { jobs, t0, files: { 'elements.json': JSON.stringify(elements) },
    report: { source: ifc, source_bytes: Number(src.size), source_mtime_ns: src.mtimeNs.toString(),
              ifc_parts: head.parts.length, split_triangles: SPLIT_TRIANGLES,
              parts_cut: split, sha256: await sha256Of(ifc) } });
  await rm(work, { recursive: true, force: true });
  log(`done in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}

// a command when run, a module when imported
if (process.argv[1] && realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url))) {
  main().catch((err) => {
    console.error('[ifc_tiles] FAILED:', err?.stack || err);
    process.exit(1);
  });
}
