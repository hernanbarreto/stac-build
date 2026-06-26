#!/usr/bin/env node
/**
 * STAC Build — GLB compressor
 *
 * Shrinks the TSDF / reconstruction `scene.glb` so the viewer can actually
 * stream it. A textured TSDF scene is typically ~270 MB (243 separate PNG
 * atlases + ~4 M triangles); that floods the browser's connection pool and
 * takes ages to download. This pass brings it down ~6× (e.g. 272 MB → 46 MB):
 *
 *   - textures : PNG → WebP, capped at maxTex px            (~115 MB → ~15 MB)
 *   - geometry : weld + reorder + quantize + EXT_meshopt    (~169 MB → ~30 MB)
 *
 * The output uses EXT_meshopt_compression, so the viewer's GLTFLoader MUST be
 * wired with a Meshopt decoder (see ui/src/components/Viewport.tsx → makeGltfLoader).
 *
 * Usage:
 *   node compress_glb.mjs <in.glb> <out.glb> [maxTex=2048] [quality=85]
 *
 * Writes atomically: builds <out>.tmp then renames over <out>, so a crash never
 * leaves a half-written GLB in place of the original.
 *
 * Hernán Barreto — Ingerop IN3
 */
import { rename } from 'node:fs/promises';
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, EXTMeshoptCompression } from '@gltf-transform/extensions';
import { dedup, prune, weld, quantize, reorder, textureCompress } from '@gltf-transform/functions';
import { MeshoptEncoder, MeshoptDecoder } from 'meshoptimizer';
import sharp from 'sharp';

const [, , inPath, outPath, maxTexArg, qualityArg] = process.argv;
const maxTex = parseInt(maxTexArg || '2048', 10);
const quality = parseInt(qualityArg || '85', 10);

if (!inPath || !outPath) {
  console.error('usage: node compress_glb.mjs <in.glb> <out.glb> [maxTex=2048] [quality=85]');
  process.exit(2);
}

async function main() {
  await MeshoptEncoder.ready;
  await MeshoptDecoder.ready;

  const io = new NodeIO()
    .registerExtensions(ALL_EXTENSIONS)
    .registerDependencies({
      'meshopt.encoder': MeshoptEncoder,
      'meshopt.decoder': MeshoptDecoder,
    });

  const t0 = Date.now();
  const doc = await io.read(inPath);

  await doc.transform(
    dedup(),
    prune(),
    // Textures: cap resolution and recode PNG → WebP (the big win on file size).
    textureCompress({ encoder: sharp, targetFormat: 'webp', resize: [maxTex, maxTex], quality }),
    // Geometry: weld duplicate verts, reorder for the meshopt codec, then quantize.
    // 14-bit positions ≈ sub-mm over a room-scale bbox — invisible for viewing.
    weld(),
    reorder({ encoder: MeshoptEncoder }),
    quantize({ quantizePosition: 14, quantizeNormal: 10, quantizeTexcoord: 12 }),
  );

  doc.createExtension(EXTMeshoptCompression)
    .setRequired(true)
    .setEncoderOptions({ method: EXTMeshoptCompression.EncoderMethod.FILTER });

  // Keep the .glb suffix on the temp file: NodeIO picks GLB vs glTF (JSON +
  // external .bin) by extension, so a bare ".tmp" would silently write a tiny
  // JSON-only file and drop all the binary geometry.
  const tmpPath = outPath + '.tmp.glb';
  await io.write(tmpPath, doc);
  await rename(tmpPath, outPath);

  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[compress_glb] OK ${inPath} -> ${outPath} (maxTex=${maxTex}, q=${quality}) in ${secs}s`);
}

main().catch((err) => {
  console.error('[compress_glb] FAILED:', err?.stack || err);
  process.exit(1);
});
