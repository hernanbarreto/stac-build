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
    """Convert binary PLY (with or without origin/confidence fields) to LAS 1.4 with RGB.
    
    If PLY contains a `confidence` field, it is mapped to LAS `intensity` (uint16, 0–65535).
    Returns the number of points converted.
    """
    import laspy

    # ── Parse PLY header: build the dtype from the ACTUAL property list ──
    # (never guess the layout — clouds arrive as float32 or float64 xyz, with
    # or without confidence/origins, and in whatever property order the writer
    # used; a hardcoded layout crashed with 'buffer size must be a multiple of
    # element size' the day a float64 cloud showed up)
    _ply_types = {
        b"char": "i1", b"uchar": "u1", b"int8": "i1", b"uint8": "u1",
        b"short": "<i2", b"ushort": "<u2", b"int16": "<i2", b"uint16": "<u2",
        b"int": "<i4", b"uint": "<u4", b"int32": "<i4", b"uint32": "<u4",
        b"float": "<f4", b"float32": "<f4", b"double": "<f8", b"float64": "<f8",
    }
    _aliases = {"red": "r", "green": "g", "blue": "b"}
    with open(ply_path, "rb") as f:
        n_pts = 0
        fields = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            s = line.strip()
            if s.startswith(b"format") and b"binary_little_endian" not in s:
                raise ValueError(f"Unsupported PLY format (not binary LE): {s!r}")
            if s.startswith(b"element"):
                parts = s.split()
                in_vertex = parts[1] == b"vertex"
                if in_vertex:
                    n_pts = int(parts[2])
            elif s.startswith(b"property") and in_vertex:
                parts = s.split()
                if parts[1] == b"list" or parts[1] not in _ply_types:
                    raise ValueError(f"Unsupported PLY property: {s!r}")
                name = parts[2].decode()
                fields.append((_aliases.get(name, name), _ply_types[parts[1]]))
            if s.startswith(b"end_header"):
                break

        ply_dtype = np.dtype(fields)
        data = np.frombuffer(f.read(), dtype=ply_dtype, count=n_pts)

    has_origins = "frame_global" in ply_dtype.names
    has_confidence = "confidence" in ply_dtype.names
    if not {"x", "y", "z", "r", "g", "b"} <= set(ply_dtype.names):
        raise ValueError(f"PLY missing xyz/rgb properties: {ply_dtype.names}")

    if len(data) == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")

    logger.info(f"[Potree] Read {len(data):,} points from {ply_path.name}")
    if has_confidence:
        logger.info(f"[Potree] Confidence field detected — mapping to LAS intensity")

    # ── Write LAS 1.4 (point format 7 = XYZ + RGB + 8-bit classification) ──
    # NOT format 2: its legacy classification field is 5 bits (max 31) — with
    # >31 segmented instances laspy raises OverflowError and the octree build
    # dies (test2 hit it with 46 instances). Format 7 classification is uint8
    # (0..255) and PotreeConverter supports it (formatToExtraIndex has {7,11}).
    header = laspy.LasHeader(point_format=7, version="1.4")
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

    # Per-point segment classification (0=unsegmented, 1..N=segment ID)
    class_npy = ply_path.parent / "classification.npy"
    if class_npy.exists():
        class_arr = np.load(class_npy)
        if len(class_arr) == len(data):
            n_over = int(np.count_nonzero(class_arr > 255))
            if n_over:
                # uint8 cast would WRAP (300 → 44, wrong instance) — clamp to
                # 0 (unsegmented) instead and say so loudly
                logger.warning(f"[Potree] {n_over:,} pts with instance id >255 "
                               f"(LAS classification is uint8) → set to 0/unsegmented")
                class_arr = np.where(class_arr > 255, 0, class_arr)
            las.classification = class_arr.astype(np.uint8)
            logger.info(f"[Potree] Classification loaded: {np.count_nonzero(class_arr):,} classified pts "
                        f"(max instance id {int(class_arr.max())})")
        else:
            logger.warning(f"[Potree] Classification size mismatch: {len(class_arr)} vs {len(data)} pts")

    # Confidence is already normalized to [0, 1] by VGGT-Long — map directly to uint16 intensity.
    if has_confidence:
        conf = data['confidence'].astype(np.float32)
        # NO NaN TOLERATED: a non-finite confidence means an upstream stage shipped
        # corrupted data. Sanitizing here would HIDE that corruption (and non-finite
        # values also break PotreeConverter's metadata.json). Fail loudly instead —
        # the error must die at its source, not travel in disguise.
        _bad = ~np.isfinite(conf)
        if _bad.any():
            raise ValueError(
                f"cleaned cloud has {int(_bad.sum()):,}/{len(conf):,} non-finite "
                f"confidence values — an upstream stage corrupted the cloud; refusing "
                f"to convert corrupted data")
        conf = np.clip(conf, 0.0, 1.0)
        las.intensity = (conf * 65535).astype(np.uint16)
        n = len(conf)
        logger.info(f"[Potree] Confidence [0-1] mapped to intensity ({n:,} pts)")

    # Per-point origin traceability (source keyframe + pixel) as LAS extra
    # dimensions, so PotreeConverter carries them into the octree attributes —
    # otherwise frame_global/pixel_row/pixel_col would be dropped here and live
    # only in cleaned_cloud.ply. (Confidence also kept as 'confidence' extra dim
    # in addition to intensity, so the value is exact, not quantized to uint16.)
    if has_confidence:
        las.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))
        las.confidence = data['confidence'].astype(np.float32)  # verified finite above
    if has_origins:
        # Pick the smallest int type that fits each field's ACTUAL range, so the
        # octree stays compact for typical scans (frame_global ~hundreds-to-low-
        # thousands, pixels = DA3 res) WITHOUT ever overflowing on a large scan.
        def _int_dim(arr):
            hi = int(arr.max()) if arr.size else 0
            lo = int(arr.min()) if arr.size else 0
            if lo >= 0 and hi <= 65535:
                return np.uint16
            return np.int32
        _types = {}
        for name in ("frame_global", "pixel_row", "pixel_col"):
            t = _int_dim(data[name])
            _types[name] = t
            las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=t))
        las.frame_global = data['frame_global'].astype(_types['frame_global'])
        las.pixel_row = data['pixel_row'].astype(_types['pixel_row'])
        las.pixel_col = data['pixel_col'].astype(_types['pixel_col'])
        logger.info(f"[Potree] Origin fields written as LAS extra dims "
                    f"({', '.join(f'{k}:{np.dtype(v).name}' for k, v in _types.items())}) "
                    f"→ propagated to octree")

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
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            logger.warning(f"[Potree] rmtree failed ({e}), falling back to rm -rf")
            subprocess.run(["rm", "-rf", str(output_dir)], check=False)

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
        logger.error(f"[Potree] ❌ PotreeConverter failed (exit {result.returncode}, "
                     f"signal {-result.returncode if result.returncode < 0 else 'n/a'})\n"
                     f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}")
        return False

    # Verify output
    metadata = output_dir / "metadata.json"
    if not metadata.exists():
        logger.error(f"[Potree] ❌ No metadata.json in output")
        return False

    logger.info(f"[Potree] ✅ Octree created at {output_dir}")
    return True


def convert_ply_to_potree(session_dir: Path, force: bool = False, ply_override: Path = None) -> bool:
    """Full pipeline: PLY → LAS → Potree octree.
    
    Args:
        session_dir: Path to session dir (e.g. server/scans/live_xxx)
        force: If True, skip mtime cache check and always reconvert.
        ply_override: Optional path to use instead of cleaned_cloud.ply
    
    Returns:
        True if potree/ directory was created successfully.
    """
    output_dir = session_dir / "output"
    ply_path = Path(ply_override) if ply_override else output_dir / "cleaned_cloud.ply"
    potree_dir = output_dir / "potree"

    if not ply_path.exists():
        logger.warning(f"[Potree] No {ply_path.name} found in {output_dir}")
        return False

    # Skip if already converted and PLY hasn't changed (unless forced)
    if not force and potree_dir.exists() and (potree_dir / "metadata.json").exists():
        potree_mtime = (potree_dir / "metadata.json").stat().st_mtime
        ply_mtime = ply_path.stat().st_mtime
        if potree_mtime > ply_mtime:
            logger.info(f"[Potree] Octree already up-to-date, skipping conversion")
            return True

    logger.info(f"[Potree] 🌲 Starting PLY → Potree conversion using pure Linux I/O...")

    try:

        # Write the intermediate LAS + octree on the big /workspace volume (via
        # output_dir), NOT /tmp: /tmp here is the container overlay (~20GB, ~7.8 free)
        # and a 128M-point LAS (~4.4GB) + UNCOMPRESSED octree overflows it →
        # PotreeConverter crashes. /workspace has tens of GB free.
        import time
        with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmpdir:
            tmpdir_path = Path(tmpdir)
            tmp_las_path = tmpdir_path / "cleaned_cloud.las"
            tmp_potree_dir = tmpdir_path / "potree"
            
            n_points = _ply_to_las(ply_path, tmp_las_path)
            logger.info(f"[Potree] Converted {n_points:,} points to LAS in RAM/tmp")

            success = _run_potree_converter(tmp_las_path, tmp_potree_dir)

            if success:
                # Evitar borrar la carpeta vieja previniendo Errno 22 por bloqueo
                if potree_dir.exists():
                    old_potree = output_dir / f"potree_old_{int(time.time())}"
                    try:
                        potree_dir.rename(old_potree)
                        subprocess.Popen(["rm", "-rf", str(old_potree)]) 
                    except OSError:
                        pass
                
                # Copiado nativo relámpago a través de bash para evitar colapso de Windows
                subprocess.run(["mkdir", "-p", str(potree_dir)]) 
                subprocess.run(["cp", "-a", f"{tmp_potree_dir}/.", str(potree_dir)])
                logger.info(f"[Potree] Successfully transferred chunks to Windows mount {potree_dir} in record time")

            return success

    except Exception as e:
        logger.error(f"[Potree] ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def convert_ply_to_potree_async(
    session_dir: Path,
    on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
    force: bool = False,
    ply_override: Path = None,
) -> bool:
    """Async wrapper — runs conversion in executor to avoid blocking event loop."""
    if on_progress:
        await on_progress("Converting point cloud to LOD octree...")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, lambda: convert_ply_to_potree(session_dir, force, ply_override)
    )

    if result and on_progress:
        await on_progress("LOD octree ready")

    return result


def convert_sabana_to_potree(session_dir: Path, force: bool = False) -> bool:
    """Convert sabana_cloud.ply → LAS → Potree octree in sabana_potree/.

    The sábana PLY lives in the session root (not output/).
    """
    ply_path = session_dir / "sabana_cloud.ply"
    potree_dir = session_dir / "sabana_potree"

    if not ply_path.exists():
        logger.warning(f"[Potree] No sabana_cloud.ply found in {session_dir}")
        return False

    # Skip if already converted and PLY hasn't changed
    if not force and potree_dir.exists() and (potree_dir / "metadata.json").exists():
        potree_mtime = (potree_dir / "metadata.json").stat().st_mtime
        ply_mtime = ply_path.stat().st_mtime
        if potree_mtime > ply_mtime:
            logger.info("[Potree] Sábana octree already up-to-date, skipping conversion")
            return True

    logger.info("[Potree] 🌲 Starting sábana PLY → Potree conversion...")

    las_path = session_dir / "sabana_cloud.las"

    try:
        n_points = _ply_to_las(ply_path, las_path)
        logger.info(f"[Potree] Converted {n_points:,} sábana points to LAS")
        success = _run_potree_converter(las_path, potree_dir)
        return success
    except Exception as e:
        logger.error(f"[Potree] ❌ Sábana conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if las_path.exists():
            las_path.unlink()


async def convert_sabana_to_potree_async(
    session_dir: Path,
    on_progress: Optional[Callable[[str], Awaitable[None]]] = None,
    force: bool = False,
) -> bool:
    """Async wrapper for sábana Potree conversion."""
    if on_progress:
        await on_progress("Converting sábana to LOD octree...")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, convert_sabana_to_potree, session_dir, force)

    if result and on_progress:
        await on_progress("Sábana LOD octree ready")

    return result
