"""
STAC Build — Potree Converter Wrapper
Converts cleaned_cloud.ply → LAS → Potree 2.0 octree

Hernán Barreto — Ingerop IN3 Session IV
"""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional, Awaitable

import numpy as np

logger = logging.getLogger(__name__)

# Path to PotreeConverter binary (compiled from vendor/)
POTREE_BIN = Path(__file__).parent.parent / "vendor" / "PotreeConverter" / "build" / "PotreeConverter"


def _ply_to_las(ply_path: Path, las_path: Path) -> int:
    """Convert binary PLY (with or without origin fields) to LAS 1.4 with RGB.
    
    Returns the number of points converted.
    """
    import laspy

    # ── Parse PLY header to detect fields ──
    with open(ply_path, "rb") as f:
        n_pts = 0
        has_origins = False
        while True:
            line = f.readline()
            if line.startswith(b"element vertex"):
                n_pts = int(line.split()[-1])
            if b"frame_global" in line:
                has_origins = True
            if line.startswith(b"end_header"):
                break

        # Build numpy dtype matching PLY binary layout
        if has_origins:
            ply_dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ('frame_global', '<i4'),
                ('pixel_row', '<i2'), ('pixel_col', '<i2')
            ])
        else:
            ply_dtype = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1')
            ])

        data = np.frombuffer(f.read(), dtype=ply_dtype)

    if len(data) == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")

    logger.info(f"[Potree] Read {len(data):,} points from {ply_path.name}")

    # ── Write LAS 1.4 (point format 2 = XYZ + RGB) ──
    header = laspy.LasHeader(point_format=2, version="1.4")
    header.offsets = [
        float(data['x'].min()),
        float(data['y'].min()),
        float(data['z'].min()),
    ]
    header.scales = [0.001, 0.001, 0.001]

    las = laspy.LasData(header)
    las.x = data['x'].astype(np.float64)
    las.y = data['y'].astype(np.float64)
    las.z = data['z'].astype(np.float64)
    # LAS RGB is 16-bit; PLY is 8-bit → scale up
    las.red = data['r'].astype(np.uint16) * 256
    las.green = data['g'].astype(np.uint16) * 256
    las.blue = data['b'].astype(np.uint16) * 256

    las.write(las_path)
    logger.info(f"[Potree] Written LAS: {las_path} ({las_path.stat().st_size / 1024**2:.1f} MB)")

    return len(data)


def _run_potree_converter(las_path: Path, output_dir: Path) -> bool:
    """Run PotreeConverter 2.1 CLI on a LAS file.
    
    Returns True on success.
    """
    if not POTREE_BIN.exists():
        raise FileNotFoundError(
            f"PotreeConverter not found at {POTREE_BIN}. "
            "Compile it: cd vendor/PotreeConverter && mkdir build && cd build && "
            "cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)"
        )

    # Remove old output if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    cmd = [
        str(POTREE_BIN),
        "-i", str(las_path),
        "-o", str(output_dir),
        "--encoding", "UNCOMPRESSED",
    ]

    logger.info(f"[Potree] Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max
    )

    if result.returncode != 0:
        logger.error(f"[Potree] ❌ PotreeConverter failed:\n{result.stderr}")
        return False

    # Verify output
    metadata = output_dir / "metadata.json"
    if not metadata.exists():
        logger.error(f"[Potree] ❌ No metadata.json in output")
        return False

    logger.info(f"[Potree] ✅ Octree created at {output_dir}")
    return True


def convert_ply_to_potree(session_dir: Path) -> bool:
    """Full pipeline: cleaned_cloud.ply → LAS → Potree octree.
    
    Args:
        session_dir: Path to session dir (e.g. server/scans/live_xxx)
    
    Returns:
        True if potree/ directory was created successfully.
    """
    output_dir = session_dir / "output"
    ply_path = output_dir / "cleaned_cloud.ply"
    potree_dir = output_dir / "potree"

    if not ply_path.exists():
        logger.warning(f"[Potree] No cleaned_cloud.ply found in {output_dir}")
        return False

    # Skip if already converted and PLY hasn't changed
    if potree_dir.exists() and (potree_dir / "metadata.json").exists():
        potree_mtime = (potree_dir / "metadata.json").stat().st_mtime
        ply_mtime = ply_path.stat().st_mtime
        if potree_mtime > ply_mtime:
            logger.info(f"[Potree] Octree already up-to-date, skipping conversion")
            return True

    logger.info(f"[Potree] 🌲 Starting PLY → Potree conversion...")

    # Use temp file for LAS to avoid leaving it around
    las_path = output_dir / "cleaned_cloud.las"

    try:
        # Step 1: PLY → LAS
        n_points = _ply_to_las(ply_path, las_path)
        logger.info(f"[Potree] Converted {n_points:,} points to LAS")

        # Step 2: LAS → Potree octree
        success = _run_potree_converter(las_path, potree_dir)

        return success

    except Exception as e:
        logger.error(f"[Potree] ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary LAS file
        if las_path.exists():
            las_path.unlink()
            logger.info(f"[Potree] Cleaned up temporary LAS file")


async def convert_ply_to_potree_async(
    session_dir: Path,
    on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
) -> bool:
    """Async wrapper — runs conversion in executor to avoid blocking event loop."""
    if on_progress:
        await on_progress("Converting point cloud to LOD octree...")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, convert_ply_to_potree, session_dir)

    if result and on_progress:
        await on_progress("LOD octree ready")

    return result
