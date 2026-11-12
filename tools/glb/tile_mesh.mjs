#!/usr/bin/env node
/**
 * STAC Build — a big textured mesh as 3D Tiles (USER 2026-11-02: "hay que implementar el tema de los tiles … para que
 * pueda ser manejable una malla grande").
 *
 * WHY. station's GenRecon mesh (91 baked chunks, 10.9 M triangles, 182 PNG atlases) loaded as ONE GLB: the browser
 * decodes every texture to RGBA on the card (~4 GB at 2048 px) and the viewer restarted. A tileset is to a mesh what
 * the Potree octree is to the cloud: the viewer (3d-tiles-renderer) loads only the tiles on screen, each at the
 * detail the screen needs, and drops the others.
 *
 * WHAT. The mesh's own primitives are the leaves, at full detail (GenRecon: one per baked chunk, with its own atlas).
 * A k-d tree over them (split at the median centroid along the longest axis of the centroids) builds the coarser
 * tiles: a tile of height h holds every primitive below it, each simplified by meshoptimizer to 1/2^h of its
 * triangles with its topological border LOCKED (the cut between two chunks never opens), its textures at the
 * source side >> ceil(h/2) (per tile about the texel budget of a leaf). A tile's geometricError is the larger of
 * the simplifier's own error (absolute, metres) and the world size of one texel at that side (the primitive's
 * surface over its UV area), never under its children's. Textures: KTX2 (Basis UASTC + zstd, mipmaps generated) —
 * they stay compressed on the card. Geometry: quantized + EXT_meshopt_compression, as compress_glb.mjs.
 *
 * NEVER AN EMPTY PRIMITIVE (pccr 2026-08-24, 2026-11-12: a speck of 8 triangles in 1.4 cm, its budget one triangle
 * at height 3, came back from the simplifier with NO triangle; written as an empty primitive it stopped the meshopt
 * encoder — "Assertion failed" — after 223 of 251 tiles). A primitive the simplifier leaves with no triangle keeps
 * the geometry, and the error, of the tile below: a tile never loses a primitive. A primitive of the SOURCE with no
 * triangle has nothing to tile and is left out, declared in the log and the report.
 *
 * Usage: node tile_mesh.mjs <in.glb> <out dir> [--jobs N] [--uastc-quality Q]
 *   <out dir>/tileset.json + tile_<id>.glb + tiles_report.json, built in <out dir>.part and renamed at the end (a
 *   viewer never reads half a tileset). asset.gltfUpAxis "Z": 3d-tiles-renderer applies no axis turn then — the
 *   content and the boxes stay in the mesh's own frame, so the tileset sits exactly where the GLB sat.
 *   The ktx tool: $KTX_BIN (default <repo>/vendor/KTX-Software-4.4.2-Linux-x86_64/bin/ktx, its lib/ beside it).
 */
import { spawn } from 'node:child_process';
import { closeSync, existsSync, fstatSync, openSync, readSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Document, Logger, NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression, KHRTextureBasisu } from '@gltf-transform/extensions';
import { quantize, reorder } from '@gltf-transform/functions';
import { MeshoptDecoder, MeshoptEncoder, MeshoptSimplifier } from 'meshoptimizer';
import sharp from 'sharp';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const KTX_BIN = process.env.KTX_BIN || path.join(REPO, 'vendor', 'KTX-Software-4.4.2-Linux-x86_64', 'bin', 'ktx');
const TRIANGLES = 4;
// the simplifier's error bound with ErrorAbsolute, in metres: none — the triangle budget of the level decides and
// the error it reaches is MEASURED (it becomes the tile's geometricError)
const NO_ERROR_BOUND = 1e9;
const CARRIED_SLOTS = ['baseColorTexture', 'metallicRoughnessTexture'];

function parseArgs(argv) {
  const a = { jobs: 8, uastcQuality: 2, pos: [] };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--jobs') a.jobs = parseInt(argv[++i], 10);
    else if (argv[i] === '--uastc-quality') a.uastcQuality = parseInt(argv[++i], 10);
    else a.pos.push(argv[i]);
  }
  if (a.pos.length !== 2 || !(a.jobs > 0) || !(a.uastcQuality >= 0 && a.uastcQuality <= 4)) {
    console.error('usage: node tile_mesh.mjs <in.glb> <out dir> [--jobs N] [--uastc-quality 0..4]');
    process.exit(2);
  }
  return a;
}

const log = (msg) => console.log(`${new Date().toISOString().slice(11, 19)} [tile_mesh] ${msg}`);

/** An accessor's elements as a Float32Array (dequantized when stored as normalized integers). */
function floatArray(acc) {
  const n = acc.getCount(), k = acc.getElementSize();
  const arr = acc.getArray();
  if (arr instanceof Float32Array && !acc.getNormalized()) return Float32Array.from(arr);
  const out = new Float32Array(n * k), el = new Array(k);
  for (let i = 0; i < n; i++) {
    acc.getElement(i, el);
    for (let j = 0; j < k; j++) out[i * k + j] = el[j];
  }
  return out;
}

function mat3NormalOf(m) {
  // the inverse transpose of the upper 3x3 of a column-major mat4
  const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], i = m[10];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A / det, B / det, C / det,
          -(b * i - c * h) / det, (a * i - c * g) / det, -(a * h - b * g) / det,
          (b * f - c * e) / det, -(a * f - c * d) / det, (a * e - b * d) / det];
}

/** One primitive of the source, its vertices in the scene's frame. */
function extract(prim, M, index) {
  if (prim.getMode() !== TRIANGLES) throw new Error(`primitive ${index}: mode ${prim.getMode()}, only triangles are tiled`);
  const mat = prim.getMaterial();
  for (const slot of ['normalTexture', 'occlusionTexture', 'emissiveTexture']) {
    const getter = `get${slot[0].toUpperCase()}${slot.slice(1)}`;
    if (mat && mat[getter]()) throw new Error(`primitive ${index}: its material has a ${slot} — not carried by the tiler`);
  }
  const pos = floatArray(prim.getAttribute('POSITION'));
  const nrmAcc = prim.getAttribute('NORMAL'), uvAcc = prim.getAttribute('TEXCOORD_0');
  const nrm = nrmAcc ? floatArray(nrmAcc) : null;
  const uv = uvAcc ? floatArray(uvAcc) : null;
  const nv = pos.length / 3;
  const ia = prim.getIndices();
  const idx = ia ? Uint32Array.from(ia.getArray()) : Uint32Array.from({ length: nv }, (_, i) => i);
  const identity = M.every((v, i) => v === (i % 5 === 0 ? 1 : 0));
  if (!identity) {
    const N = mat3NormalOf(M);
    for (let v = 0; v < nv; v++) {
      const x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
      pos[3 * v] = M[0] * x + M[4] * y + M[8] * z + M[12];
      pos[3 * v + 1] = M[1] * x + M[5] * y + M[9] * z + M[13];
      pos[3 * v + 2] = M[2] * x + M[6] * y + M[10] * z + M[14];
      if (nrm) {
        const p = nrm[3 * v], q = nrm[3 * v + 1], r = nrm[3 * v + 2];
        const nx = N[0] * p + N[3] * q + N[6] * r, ny = N[1] * p + N[4] * q + N[7] * r, nz = N[2] * p + N[5] * q + N[8] * r;
        const l = Math.hypot(nx, ny, nz) || 1;
        nrm[3 * v] = nx / l; nrm[3 * v + 1] = ny / l; nrm[3 * v + 2] = nz / l;
      }
    }
  }
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < nv; v++) for (let k = 0; k < 3; k++) {
    const x = pos[3 * v + k];
    if (x < min[k]) min[k] = x;
    if (x > max[k]) max[k] = x;
  }
  // surface (m²) and UV area (fraction of the atlas): one texel's world size at a given side
  let area = 0, uvArea = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const i0 = idx[t], i1 = idx[t + 1], i2 = idx[t + 2];
    const ux = pos[3 * i1] - pos[3 * i0], uy = pos[3 * i1 + 1] - pos[3 * i0 + 1], uz = pos[3 * i1 + 2] - pos[3 * i0 + 2];
    const vx = pos[3 * i2] - pos[3 * i0], vy = pos[3 * i2 + 1] - pos[3 * i0 + 1], vz = pos[3 * i2 + 2] - pos[3 * i0 + 2];
    area += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    if (uv) {
      const a0 = uv[2 * i1] - uv[2 * i0], a1 = uv[2 * i1 + 1] - uv[2 * i0 + 1];
      const b0 = uv[2 * i2] - uv[2 * i0], b1 = uv[2 * i2 + 1] - uv[2 * i0 + 1];
      uvArea += 0.5 * Math.abs(a0 * b1 - a1 * b0);
    }
  }
  const material = mat ? {
    name: mat.getName(),
    baseColorFactor: mat.getBaseColorFactor(), metallicFactor: mat.getMetallicFactor(),
    roughnessFactor: mat.getRoughnessFactor(), doubleSided: mat.getDoubleSided(), alphaMode: mat.getAlphaMode(),
    alphaCutoff: mat.getAlphaCutoff(),
    images: Object.fromEntries(CARRIED_SLOTS.map(slot => {
      const tex = mat[`get${slot[0].toUpperCase()}${slot.slice(1)}`]();
      return [slot, tex ? { bytes: tex.getImage(), mime: tex.getMimeType() } : null];
    })),
  } : null;
  return { index, pos, nrm, uv, idx, min, max, area, uvArea, material,
           centroid: [0, 1, 2].map(k => 0.5 * (min[k] + max[k])) };
}

/** The k-d tree: leaves are single primitives; a node splits its set at the median centroid of its longest axis. */
function build(items) {
  if (items.length === 1) return { items, children: [], h: 0 };
  const lo = [0, 1, 2].map(k => Math.min(...items.map(it => it.centroid[k])));
  const hi = [0, 1, 2].map(k => Math.max(...items.map(it => it.centroid[k])));
  const ax = [0, 1, 2].reduce((best, k) => (hi[k] - lo[k] > hi[best] - lo[best] ? k : best), 0);
  const sorted = [...items].sort((p, q) => (p.centroid[ax] - q.centroid[ax]) || (p.index - q.index));
  const mid = Math.floor(sorted.length / 2);
  const a = build(sorted.slice(0, mid)), b = build(sorted.slice(mid));
  return { items, children: [a, b], h: 1 + Math.max(a.h, b.h) };
}

/** The used vertices of a simplified index list, renumbered in first-use order. */
function compact(it, idx) {
  const remap = new Int32Array(it.pos.length / 3).fill(-1);
  let n = 0;
  const out = new Uint32Array(idx.length);
  for (let t = 0; t < idx.length; t++) {
    const v = idx[t];
    if (remap[v] < 0) remap[v] = n++;
    out[t] = remap[v];
  }
  const pos = new Float32Array(3 * n), nrm = it.nrm ? new Float32Array(3 * n) : null, uv = it.uv ? new Float32Array(2 * n) : null;
  for (let v = 0; v < remap.length; v++) {
    const w = remap[v];
    if (w < 0) continue;
    pos.set(it.pos.subarray(3 * v, 3 * v + 3), 3 * w);
    if (nrm) nrm.set(it.nrm.subarray(3 * v, 3 * v + 3), 3 * w);
    if (uv) uv.set(it.uv.subarray(2 * v, 2 * v + 2), 2 * w);
  }
  return { pos, nrm, uv, idx: n < 65536 ? Uint16Array.from(out) : out, vertices: n };
}

function run(cmd, args, env) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { env, stdio: ['ignore', 'ignore', 'pipe'] });
    let err = '';
    p.stderr.on('data', d => { err += d; });
    p.on('error', reject);
    p.on('close', code => (code === 0 ? resolve() : reject(new Error(`${path.basename(cmd)} ${args.join(' ')} → exit ${code}: ${err.trim()}`))));
  });
}


function readLarge(p) {
  const fd = openSync(p, 'r');
  try {
    const size = fstatSync(fd).size;
    const buf = Buffer.allocUnsafe(size);
    let off = 0;
    while (off < size) {
      const n = readSync(fd, buf, off, Math.min(1 << 30, size - off), off);
      if (n <= 0) throw new Error(`short read at ${off} of ${size} bytes`);
      off += n;
    }
    return new Uint8Array(buf.buffer, buf.byteOffset, size);
  } finally {
    closeSync(fd);
  }
}

async function pool(tasks, n) {
  let next = 0;
  const worker = async () => { while (next < tasks.length) { const t = tasks[next++]; await t(); } };
  await Promise.all(Array.from({ length: Math.min(n, tasks.length) }, worker));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const [inPath, outDir] = args.pos;
  if (!existsSync(KTX_BIN)) throw new Error(`the ktx tool is missing: ${KTX_BIN} (set KTX_BIN)`);
  const ktxEnv = { ...process.env, LD_LIBRARY_PATH: [path.join(path.dirname(KTX_BIN), '..', 'lib'), process.env.LD_LIBRARY_PATH].filter(Boolean).join(':') };
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;
  await MeshoptSimplifier.ready;
  const io = new NodeIO().registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder });
  const t0 = Date.now();
  // STAC 2026-11-05 (zaragoza: a 2.3 GB GLB — fs.readFile stops at 2 GiB, ERR_FS_FILE_TOO_LARGE): the GLB is read in
  // 1 GiB pieces into one buffer (Node allows up to 4 GiB; a GLB counts its length in 32 bits anyway) and parsed from it
  const src = inPath.toLowerCase().endsWith('.glb') ? await io.readBinary(readLarge(inPath)) : await io.read(inPath);
  const items = [];
  let empty = 0;                                    // primitives of the source with no triangle: nothing to tile
  for (const scene of src.getRoot().listScenes()) {
    scene.traverse(node => {
      const mesh = node.getMesh();
      if (!mesh) return;
      const M = node.getWorldMatrix();
      for (const prim of mesh.listPrimitives()) {
        const it = extract(prim, M, items.length);
        if (it.idx.length >= 3) items.push(it); else empty += 1;
      }
    });
  }
  if (items.length === 0) throw new Error(`${inPath}: no mesh primitive${empty ? ` with a triangle (${empty} without)` : ''}`);
  const nTri = items.reduce((s, it) => s + it.idx.length / 3, 0);
  log(`${inPath}: ${items.length} primitives, ${nTri.toLocaleString('en')} triangles (${((Date.now() - t0) / 1000).toFixed(1)} s)`);
  if (empty) log(`${empty} primitive(s) of the source have no triangle — left out of the tiles`);

  const tree = build(items);
  let nextId = 0;
  const nodes = [];
  (function number(node) { node.id = nextId++; nodes.push(node); node.children.forEach(number); })(tree);

  // the textures every tile needs: (primitive, slot, level) → the side and the KTX2 bytes
  const sides = new Map();                          // primitive index → { slot: [w, h] } of the source images
  for (const it of items) {
    const s = {};
    for (const slot of CARRIED_SLOTS) {
      const img = it.material?.images?.[slot];
      if (img) { const m = await sharp(Buffer.from(img.bytes)).metadata(); s[slot] = [m.width, m.height]; }
    }
    sides.set(it.index, s);
  }
  const sideAt = (w, h) => Math.max(4, w >> Math.ceil(h / 2));
  const encodes = new Map();                        // key → { it, slot, w, h, bytes }
  for (const node of nodes) for (const it of node.items) for (const slot of CARRIED_SLOTS) {
    const s = sides.get(it.index)[slot];
    if (!s) continue;
    const w = sideAt(s[0], node.h), h = sideAt(s[1], node.h);
    const key = `${it.index}:${slot}:${w}x${h}`;
    if (!encodes.has(key)) encodes.set(key, { it, slot, w, h, bytes: null });
  }
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'tile_mesh_'));
  const threads = Math.max(1, Math.floor(Math.min(os.cpus().length, 30) / args.jobs));
  let done = 0;
  const t1 = Date.now();
  await pool([...encodes.entries()].map(([key, e]) => async () => {
    const base = path.join(tmp, key.replace(/[:x]/g, '_'));
    const img = e.it.material.images[e.slot];
    let im = sharp(Buffer.from(img.bytes)).removeAlpha();
    const s = sides.get(e.it.index)[e.slot];
    if (e.w !== s[0] || e.h !== s[1]) im = im.resize(e.w, e.h, { fit: 'fill', kernel: 'lanczos3' });
    await im.png({ compressionLevel: 1 }).toFile(`${base}.png`);
    const color = e.slot === 'baseColorTexture';
    await run(KTX_BIN, ['create', '--format', color ? 'R8G8B8_SRGB' : 'R8G8B8_UNORM',
      ...(color ? [] : ['--assign-tf', 'linear']), '--encode', 'uastc', '--uastc-quality', String(args.uastcQuality),
      '--zstd', '18', '--generate-mipmap', '--threads', String(threads), `${base}.png`, `${base}.ktx2`], ktxEnv);
    e.bytes = new Uint8Array(await readFile(`${base}.ktx2`));
    await rm(`${base}.png`); await rm(`${base}.ktx2`);
    done += 1;
    if (done % 100 === 0 || done === encodes.size) log(`textures ${done}/${encodes.size} (${((Date.now() - t1) / 1000).toFixed(0)} s)`);
  }), args.jobs);
  await rm(tmp, { recursive: true, force: true });

  const part = `${outDir}.part`;
  await rm(part, { recursive: true, force: true });
  await mkdir(part, { recursive: true });
  const wio = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.encoder': MeshoptEncoder });
  const simplified = new Map();                     // `${index}:${h}` → { idx, err }
  // what the tile below holds of a primitive: its entry at the nearest lower height (a primitive is in one tile per
  // height of its path, its own leaf at 0, and the tiles are written children first)
  const below = (it, h) => {
    for (let k = h - 1; k >= 0; k--) {
      const e = simplified.get(`${it.index}:${k}`);
      if (e) return e;
    }
    throw new Error(`primitive ${it.index}: no tile under height ${h} holds it`);
  };
  const keptBelow = new Set();                      // the primitives the simplifier left with no triangle somewhere
  const report = [];
  const t2 = Date.now();
  // children before parents: a parent's geometricError is never under its children's
  for (const node of [...nodes].sort((p, q) => p.h - q.h || p.id - q.id)) {
    const doc = new Document().setLogger(new Logger(Logger.Verbosity.WARN));
    const buffer = doc.createBuffer();
    doc.createExtension(KHRTextureBasisu).setRequired(true);
    const mesh = doc.createMesh();
    doc.createScene().addChild(doc.createNode().setMesh(mesh));
    let ge = 0, tris = 0, simplErr = 0, texel = 0, nKept = 0;
    for (const it of node.items) {
      const key = `${it.index}:${node.h}`;
      if (!simplified.has(key)) {
        if (node.h === 0) simplified.set(key, { idx: it.idx, err: 0 });
        else {
          const target = Math.max(3, Math.floor(it.idx.length / 3 / 2 ** node.h) * 3);
          const [idx, err] = MeshoptSimplifier.simplify(it.idx, it.pos, 3, target, NO_ERROR_BOUND, ['LockBorder', 'ErrorAbsolute']);
          // the simplifier can leave a small closed piece with NO triangle (the header): the tile then holds the
          // piece as the tile below does, with that geometry's error — never an empty primitive
          simplified.set(key, idx.length >= 3 ? { idx, err } : { ...below(it, node.h), kept: true });
        }
      }
      const { idx, err, kept } = simplified.get(key);
      if (kept) { nKept += 1; keptBelow.add(it.index); }
      const c = compact(it, idx);
      tris += idx.length / 3;
      simplErr = Math.max(simplErr, err);
      const prim = doc.createPrimitive()
        .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(c.pos).setBuffer(buffer))
        .setIndices(doc.createAccessor().setType('SCALAR').setArray(c.idx).setBuffer(buffer));
      if (c.nrm) prim.setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(c.nrm).setBuffer(buffer));
      if (c.uv) prim.setAttribute('TEXCOORD_0', doc.createAccessor().setType('VEC2').setArray(c.uv).setBuffer(buffer));
      const m = it.material;
      if (m) {
        const mat = doc.createMaterial(m.name).setBaseColorFactor(m.baseColorFactor).setMetallicFactor(m.metallicFactor)
          .setRoughnessFactor(m.roughnessFactor).setDoubleSided(m.doubleSided).setAlphaMode(m.alphaMode)
          .setAlphaCutoff(m.alphaCutoff);
        for (const slot of CARRIED_SLOTS) {
          const s = sides.get(it.index)[slot];
          if (!s) continue;
          const w = sideAt(s[0], node.h), h = sideAt(s[1], node.h);
          const e = encodes.get(`${it.index}:${slot}:${w}x${h}`);
          const tex = doc.createTexture(`${it.index}_${slot}`).setImage(e.bytes).setMimeType('image/ktx2');
          if (slot === 'baseColorTexture') mat.setBaseColorTexture(tex); else mat.setMetallicRoughnessTexture(tex);
          if (it.uvArea > 0) texel = Math.max(texel, Math.sqrt(it.area / (it.uvArea * w * h)));
        }
        prim.setMaterial(mat);
      }
      mesh.addPrimitive(prim);
    }
    ge = node.h === 0 ? 0 : Math.max(simplErr, texel, ...node.children.map(ch => ch.ge));
    node.ge = ge;
    await doc.transform(reorder({ encoder: MeshoptEncoder }),
                        quantize({ quantizePosition: 16, quantizeNormal: 10, quantizeTexcoord: 14 }));
    doc.createExtension(EXTMeshoptCompression).setRequired(true)
      .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.QUANTIZE });
    const file = path.join(part, `tile_${node.id}.glb`);
    await wio.write(file, doc);
    const bytes = (await stat(file)).size;
    report.push({ id: node.id, h: node.h, primitives: node.items.length, triangles: tris, kept_from_below: nKept, bytes,
                  simplification_error_m: simplErr, texel_m: texel, geometric_error_m: ge });
  }
  log(`${nodes.length} tiles written (${((Date.now() - t2) / 1000).toFixed(0)} s)`);
  const keptList = [...keptBelow].sort((p, q) => p - q);
  if (keptList.length) log(`${keptList.length} primitive(s) kept the geometry of the tile below where the simplifier left them no triangle: ${keptList.join(', ')}`);

  const box = (node) => {
    const lo = [0, 1, 2].map(k => Math.min(...node.items.map(it => it.min[k])));
    const hi = [0, 1, 2].map(k => Math.max(...node.items.map(it => it.max[k])));
    const c = lo.map((v, k) => 0.5 * (v + hi[k])), e = lo.map((v, k) => Math.max(0.5 * (hi[k] - v), 1e-6));
    return [c[0], c[1], c[2], e[0], 0, 0, 0, e[1], 0, 0, 0, e[2]];
  };
  const tile = (node) => ({
    boundingVolume: { box: box(node) }, geometricError: node.ge, refine: 'REPLACE',
    content: { uri: `tile_${node.id}.glb` },
    ...(node.children.length ? { children: node.children.map(tile) } : {}),
  });
  const tileset = { asset: { version: '1.1', gltfUpAxis: 'Z', generator: 'STAC tools/glb/tile_mesh.mjs' },
                    geometricError: tree.ge, root: tile(tree) };
  await writeFile(path.join(part, 'tileset.json'), JSON.stringify(tileset));
  const total = report.reduce((s, r) => s + r.bytes, 0);
  await writeFile(path.join(part, 'tiles_report.json'), JSON.stringify({
    source: path.resolve(inPath), source_bytes: (await stat(inPath)).size, primitives: items.length, triangles: nTri,
    tiles: nodes.length, depth: tree.h, bytes: total, textures_encoded: encodes.size,
    source_primitives_without_triangles: empty, primitives_kept_from_below: keptList,
    ktx: { tool: KTX_BIN, encode: 'uastc', uastc_quality: args.uastcQuality, zstd: 18, mipmaps: true },
    rule: 'leaves = the source primitives at full detail; a tile of height h = its primitives simplified to 1/2^h of '
        + 'their triangles (meshoptimizer, LockBorder, its own absolute error), textures at the source side >> ceil(h/2); '
        + 'a primitive the simplifier leaves with no triangle keeps the geometry and the error of the tile below; '
        + 'geometricError = max(simplifier error, texel size, children)', seconds: (Date.now() - t0) / 1000,
    per_tile: report.sort((p, q) => p.id - q.id) }, null, 1));
  await rm(outDir, { recursive: true, force: true });
  await rename(part, outDir);
  log(`→ ${outDir}: ${nodes.length} tiles, depth ${tree.h}, ${(total / 1e6).toFixed(0)} MB, root error ${tree.ge.toFixed(3)} m (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}

main().catch((err) => {
  console.error('[tile_mesh] FAILED:', err?.stack || err);
  process.exit(1);
});
