#!/usr/bin/env python3
"""
Convert 2DGS PLY (GauS-SLAM) → 3DGS PLY (SuperSplat compatible)

Changes:
  1. Adds scale_2 = scale_1 - 5.0  (very thin 3rd dimension, log-space)
  2. Renames r,g,b → f_dc_0,f_dc_1,f_dc_2  (SH DC component: f = (rgb - 0.5)/C0)
  3. Keeps all other properties identical (nx,ny,nz,opacity,rot_*)

Usage:
  python convert_2dgs_to_3dgs.py --input gaussians.ply --output gaussians_3dgs.ply
"""

import argparse
import numpy as np
import struct
import re
from pathlib import Path

C0 = 0.28209479177387814  # SH DC coefficient

def read_ply_header(f):
    """Read PLY header, return (header_bytes, properties, n_verts, end_pos)"""
    header_lines = []
    while True:
        line = f.readline()
        header_lines.append(line)
        if line.strip() == b'end_header':
            break
    header_text = b''.join(header_lines).decode('ascii')
    
    # Parse vertex count
    n_verts = int(re.search(r'element vertex (\d+)', header_text).group(1))
    
    # Parse properties in order
    props = re.findall(r'property (\w+) (\w+)', header_text)
    
    return header_text, props, n_verts


def dtype_from_ply_type(ply_type):
    return {
        'float': np.float32, 'float32': np.float32,
        'double': np.float64,
        'int': np.int32, 'uint': np.uint32,
        'short': np.int16, 'ushort': np.uint16,
        'char': np.int8, 'uchar': np.uint8,
    }[ply_type]


def main():
    parser = argparse.ArgumentParser("2DGS → 3DGS PLY converter for SuperSplat")
    parser.add_argument("--input",  required=True, help="Input 2DGS PLY (from GauS-SLAM)")
    parser.add_argument("--output", required=True, help="Output 3DGS PLY (SuperSplat)")
    parser.add_argument("--thin_margin", type=float, default=5.0,
                        help="Log-space margin for scale_2 below min(scale_0,scale_1). "
                             "Higher = thinner disk. Default: 5.0")
    args = parser.parse_args()

    print(f"[2DGS→3DGS] Reading: {args.input}")
    with open(args.input, 'rb') as f:
        header_text, props, n_verts = read_ply_header(f)
        
        # Build numpy dtype for reading
        read_dtype = np.dtype([(name, dtype_from_ply_type(ptype)) for ptype, name in props])
        data = np.frombuffer(f.read(n_verts * read_dtype.itemsize), dtype=read_dtype)

    print(f"[2DGS→3DGS] Loaded {n_verts:,} Gaussians")
    print(f"[2DGS→3DGS] Input properties: {[n for _,n in props]}")

    # ── Build output record ──────────────────────────────────────────
    # scale_2: very small value in log-space to make 2D disk appear thin in 3D
    scale_2 = np.minimum(data['scale_0'], data['scale_1']) - args.thin_margin

    # SH DC coefficient from linear RGB: f_dc = (rgb - 0.5) / C0
    f_dc_0 = (data['r'].astype(np.float32) - 0.5) / C0
    f_dc_1 = (data['g'].astype(np.float32) - 0.5) / C0
    f_dc_2 = (data['b'].astype(np.float32) - 0.5) / C0

    out_dtype = np.dtype([
        ('x',      np.float32), ('y',      np.float32), ('z',      np.float32),
        ('nx',     np.float32), ('ny',     np.float32), ('nz',     np.float32),
        ('f_dc_0', np.float32), ('f_dc_1', np.float32), ('f_dc_2', np.float32),
        ('opacity',np.float32),
        ('scale_0',np.float32), ('scale_1',np.float32), ('scale_2',np.float32),
        ('rot_0',  np.float32), ('rot_1',  np.float32),
        ('rot_2',  np.float32), ('rot_3',  np.float32),
    ])

    out = np.zeros(n_verts, dtype=out_dtype)
    out['x']       = data['x']
    out['y']       = data['y']
    out['z']       = data['z']
    out['nx']      = data['nx']
    out['ny']      = data['ny']
    out['nz']      = data['nz']
    out['f_dc_0']  = f_dc_0
    out['f_dc_1']  = f_dc_1
    out['f_dc_2']  = f_dc_2
    out['opacity'] = data['opacity']
    out['scale_0'] = data['scale_0']
    out['scale_1'] = data['scale_1']
    out['scale_2'] = scale_2
    out['rot_0']   = data['rot_0']
    out['rot_1']   = data['rot_1']
    out['rot_2']   = data['rot_2']
    out['rot_3']   = data['rot_3']

    # ── Write 3DGS PLY header ────────────────────────────────────────
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(out.tobytes())

    size_mb = out_path.stat().st_size / 1e6
    print(f"[2DGS→3DGS] ✅ Saved: {out_path}  ({size_mb:.1f} MB)")
    print(f"[2DGS→3DGS] SuperSplat-compatible 3DGS PLY ready.")


if __name__ == "__main__":
    main()
