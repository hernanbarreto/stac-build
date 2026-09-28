"""
STAC Build — Potree Converter Wrapper
Converts cleaned_cloud.ply → LAS → Potree 2.0 octree

Hernán Barreto — Ingerop IN3 Session IV
"""

import asyncio
import errno
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Awaitable

import numpy as np

logger = logging.getLogger(__name__)

# Path to PotreeConverter binary (compiled from vendor/)
POTREE_BIN = Path(__file__).parent.parent / "vendor" / "PotreeConverter" / "build" / "PotreeConverter"

# The LAS / octree position quantum is DERIVED from the cloud (2026-09-28; it
# was a fixed 1 mm, then 0.1 mm): the finest power of two for which the
# largest extent stays under 2**LAS_EXTENT_BITS quanta. PotreeConverter's
# computeScaleOffset coarsens any scale finer than extent / 2**30, so 29 bits
# is a 2x margin under that clamp — its "adjusted" branch never fires.
LAS_EXTENT_BITS = 29


def _las_scale(extent_m: float) -> float:
    """Finest power of two ``s`` with ``extent_m / s < 2**LAS_EXTENT_BITS``.
    A power of two keeps the quantisation reproducible: dividing by it is
    exact in float64. Fails on a non-finite extent."""
    if not math.isfinite(extent_m) or extent_m < 0:
        raise ValueError(f"cloud extent {extent_m!r} m is not a finite size — "
                         f"non-finite coordinates cannot be quantised")
    # frexp: extent/2**bits = m·2**e with 0.5 <= m < 1, so 2**e is the smallest
    # power of two strictly above it (and 1.0 for a zero extent)
    return math.ldexp(1.0, math.frexp(extent_m / 2.0 ** LAS_EXTENT_BITS)[1])


# One octree build per session output at a time ACROSS PROCESSES (2026-09-28).
# The in-memory registry below only sees threads of one process; the preview
# build runs in the backend while the clean build runs in the CloudCompy
# worker subprocess, and both used to swap output/potree unserialised — the
# final octree could be the preview, or a mix of both.
LOCK_NAME = ".potree.lock"
# Cadence of the wait — how often the lock is retried and how often the wait is
# logged. Neither is a verdict: the wait has no deadline, it is only made visible.
LOCK_POLL_S = 1.0
LOCK_LOG_EVERY_S = 60.0


@contextmanager
def _potree_lock(output_dir: Path):
    """Exclusive flock on ``output_dir/.potree.lock`` for the body — waits as
    long as it takes (the holder is building the octree this caller needs),
    logging every ``LOCK_LOG_EVERY_S`` how long and on whom (the holder writes
    its pid into the lock file). The kernel drops the lock with the holder's
    process, so it never goes stale."""
    import fcntl
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / LOCK_NAME
    fh = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        t0 = time.time()
        next_log = t0
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
            now = time.time()
            if now >= next_log:
                try:
                    holder = lock_path.read_text().strip() or "unknown holder"
                except OSError as e:
                    holder = f"holder unreadable: {e}"
                logger.info(f"[Potree] waiting {now - t0:.0f}s for {output_dir}/potree "
                            f"— the octree lock is held by {holder}")
                next_log = now + LOCK_LOG_EVERY_S
            time.sleep(LOCK_POLL_S)
        os.ftruncate(fh, 0)
        os.write(fh, f"pid={os.getpid()} since={time.time():.0f}\n".encode())
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            os.close(fh)


def _swap_in(new_dir: Path, potree_dir: Path, output_dir: Path) -> None:
    """Replace ``potree_dir`` with the freshly built ``new_dir`` (same
    filesystem: both live under ``output_dir``). The previous octree moves into
    a UNIQUE directory first (never a name two builds in one second share); the
    new one is renamed in, and only THEN is the old one deleted in the
    background. Any rename error RAISES: if the new octree cannot be moved in,
    the old one is moved back first — a failed swap never leaves the session
    without an octree, nor the old and new mixed in one directory."""
    if not potree_dir.exists():
        new_dir.rename(potree_dir)
        return
    old = Path(tempfile.mkdtemp(prefix="potree_old_", dir=str(output_dir)))
    potree_dir.rename(old / "potree")
    try:
        new_dir.rename(potree_dir)
    except BaseException:
        (old / "potree").rename(potree_dir)
        old.rmdir()
        raise
    subprocess.Popen(["rm", "-rf", str(old)])


def _read_ply_vertices(ply_path: Path) -> np.ndarray:
    """The vertex records of a binary little-endian PLY as a structured array
    (field names as written, red/green/blue aliased to r/g/b)."""
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
        return np.frombuffer(f.read(), dtype=ply_dtype, count=n_pts)


def _ply_to_las(ply_path: Path, las_path: Path) -> int:
    """Convert binary PLY (with or without origin/confidence fields) to LAS 1.4 with RGB.

    If PLY contains a `confidence` field, it is mapped to LAS `intensity` (uint16, 0–65535).
    Returns the number of points converted.
    """
    return _vertices_to_las(_read_ply_vertices(ply_path), las_path, ply_path.parent,
                            ply_path.name)


def _vertices_to_las(data: np.ndarray, las_path: Path, class_dir: Path, label: str) -> int:
    """Write PLY vertex records (see :func:`_read_ply_vertices`) as LAS 1.4 with RGB;
    ``class_dir/classification.npy`` is applied when its length matches."""
    import laspy
    ply_dtype = data.dtype
    has_origins = "frame_global" in ply_dtype.names
    has_confidence = "confidence" in ply_dtype.names
    if not {"x", "y", "z", "r", "g", "b"} <= set(ply_dtype.names):
        raise ValueError(f"PLY missing xyz/rgb properties: {ply_dtype.names}")

    if len(data) == 0:
        raise ValueError(f"Empty point cloud: {label}")

    logger.info(f"[Potree] Read {len(data):,} points from {label}")
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
    # The quantum comes from the cloud (the measurement tools raycast the
    # octree, so its grid IS their resolution): the finest power of two that
    # keeps the stored integers — 0..extent/scale, the offset being the
    # minimum — under 2**LAS_EXTENT_BITS, inside the 30 bits PotreeConverter
    # keeps without coarsening. The check re-verifies it; it can only trip on
    # coordinates that are not finite.
    extent = max(float(data[a].max()) - float(data[a].min()) for a in ("x", "y", "z"))
    scale = _las_scale(extent)
    if not extent / scale < 2 ** LAS_EXTENT_BITS:
        raise ValueError(f"{label}: extent {extent!r} m does not fit "
                         f"{LAS_EXTENT_BITS}-bit integers at scale {scale!r} m")
    header.scales = [scale, scale, scale]
    logger.info(f"[Potree] LAS quantum {scale:.3g} m (2^{math.frexp(scale)[1] - 1}) for "
                f"a {extent:.2f} m extent")

    las = laspy.LasData(header)
    las.x = data['x'].astype(np.float64)
    las.y = data['y'].astype(np.float64)
    las.z = data['z'].astype(np.float64)
    # LAS RGB is 16-bit; PLY is 8-bit → scale up
    las.red = data['r'].astype(np.uint16) * 256
    las.green = data['g'].astype(np.uint16) * 256
    las.blue = data['b'].astype(np.uint16) * 256

    # Per-point segment classification (0=unsegmented, 1..N=segment ID)
    class_npy = Path(class_dir) / "classification.npy"
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

    # Intensity is the FALLBACK channel, for a cloud that carries no `confidence`
    # extra dim. It must be NORMALIZED here: the old code asserted "already
    # normalized to [0,1] by VGGT-Long" and clipped — false for this pipeline,
    # whose clouds arrive raw (pccr: 2.862..70.946), so every point clipped to
    # 1.0 and the whole channel came out saturated at 65535 (verified in the
    # octree metadata). Min-max is the same normalization PotreeLoader.ts:711
    # applies to the `confidence` attribute and Step 1c of gpu_cloud_clean
    # applies to the gate, so all three speak one language.
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
        _lo, _hi = float(conf.min()), float(conf.max())
        _norm = (conf - _lo) / max(_hi - _lo, 1e-6) if _hi > _lo else np.zeros_like(conf)
        las.intensity = (np.clip(_norm, 0.0, 1.0) * 65535).astype(np.uint16)
        n = len(conf)
        logger.info(f"[Potree] Confidence {_lo:.3f}..{_hi:.3f} normalized min-max "
                    f"→ intensity ({n:,} pts)")

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

    # claude_stac.txt §6 / §11: the witness fields (mv_votes, mask_votes,
    # mask_conflicts, status — uint8) ride along as extra dims so the viewer
    # colours by status / mv_votes straight from the octree attributes
    _wit = [n for n in ("mv_votes", "mask_votes", "mask_conflicts", "status")
            if data.dtype.names and n in data.dtype.names]
    for name in _wit:
        las.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.uint8))
        setattr(las, name, np.asarray(data[name]).astype(np.uint8))
    if _wit:
        logger.info(f"[Potree] Witness fields written as LAS extra dims ({', '.join(_wit)}) "
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

    # NO wall-clock timeout (2026-09-28): 600 s on a slow disk discarded a whole
    # correction epoch as if its geometry were wrong. Duration is not a verdict;
    # a hung converter is the heartbeat/cancel path's to detect and kill.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
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


# One Potree build per session at a time (USER 2026-09-06, after an Undo
# raced the session-load rebuild and both wrote output/potree: "ver si se
# está regenerando... pero no con un lock"). This is an IN-MEMORY registry —
# it dies with the process, so it can never go stale on disk. A caller that
# finds a build running for its session WAITS for it; if the finished build
# already came from the same cloud state, the result is reused.
import threading as _threading
_potree_active: dict = {}      # session key → {"since", "ply", "ply_mtime"}
_potree_reg = _threading.Lock()  # guards the dict only (microseconds)

# An octree built from the raw reconstruction chunks (convert_chunks_preview_to_potree)
# carries this file: it is never "up to date" for cleaned_cloud.ply and never reused
# in its place — the clean build always replaces it.
PREVIEW_MARKER = ".preview"


def convert_ply_to_potree(session_dir: Path, force: bool = False, ply_override: Path = None,
                          potree_dir_override: Path = None) -> bool:
    """Full pipeline: PLY → LAS → Potree octree.

    Args:
        session_dir: Path to session dir (e.g. server/scans/live_xxx)
        force: If True, skip mtime cache check and always reconvert.
        ply_override: Optional path to use instead of cleaned_cloud.ply
        potree_dir_override: Optional target directory instead of
            output/potree — used by the correction module to build the octree
            INSIDE its transaction directory before the atomic swap.

    Returns:
        True if potree/ directory was created successfully.
    """
    output_dir = session_dir / "output"
    ply_path = Path(ply_override) if ply_override else output_dir / "cleaned_cloud.ply"
    potree_dir = Path(potree_dir_override) if potree_dir_override else output_dir / "potree"

    if not ply_path.exists():
        logger.warning(f"[Potree] No {ply_path.name} found in {output_dir}")
        return False

    # ── serialize builds per session: in-process registry (reuse) + flock
    # (every process — the preview in the backend, the clean build in the
    # CloudCompy worker, the correction transaction) ──
    key = str(output_dir.resolve())
    waited = False
    while True:
        with _potree_reg:
            active = _potree_active.get(key)
            if active is None:
                _potree_active[key] = {
                    "since": time.time(),
                    "ply": str(ply_path),
                    "ply_mtime": ply_path.stat().st_mtime,
                }
                break
        if not waited:
            logger.info(f"[Potree] another build is running for this "
                        f"session (since {time.time()-active['since']:.0f}s)"
                        f" — waiting for it to finish")
            waited = True
        time.sleep(3)
    try:
        with _potree_lock(output_dir):
            if waited:
                # the build we waited for may have produced exactly what we
                # need: same source PLY, octree newer than it → reuse
                meta = potree_dir / "metadata.json"
                if meta.exists() and ply_path.exists() \
                        and not (potree_dir / PREVIEW_MARKER).exists() \
                        and meta.stat().st_mtime > ply_path.stat().st_mtime:
                    logger.info("[Potree] finished build already covers the "
                                "current cloud — reusing it")
                    return True
            return _convert_ply_to_potree_inner(
                output_dir, ply_path, potree_dir, force)
    finally:
        with _potree_reg:
            _potree_active.pop(key, None)


def _convert_ply_to_potree_inner(output_dir: Path, ply_path: Path,
                                 potree_dir: Path, force: bool) -> bool:

    # Skip if already converted and PLY hasn't changed (unless forced); a preview
    # octree of the raw chunks is never the clean cloud's
    if not force and potree_dir.exists() and (potree_dir / "metadata.json").exists() \
            and not (potree_dir / PREVIEW_MARKER).exists():
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
        with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmpdir:
            tmpdir_path = Path(tmpdir)
            tmp_las_path = tmpdir_path / "cleaned_cloud.las"
            tmp_potree_dir = tmpdir_path / "potree"

            n_points = _ply_to_las(ply_path, tmp_las_path)
            logger.info(f"[Potree] Converted {n_points:,} points to LAS in RAM/tmp")

            success = _run_potree_converter(tmp_las_path, tmp_potree_dir)

            if success:
                _swap_in(tmp_potree_dir, potree_dir, output_dir)
                logger.info(f"[Potree] octree swapped into {potree_dir}")

            return success

    except Exception as e:
        logger.error(f"[Potree] ❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_chunks_preview_to_potree(session_dir: Path) -> bool:
    """Octree of the RAW reconstruction — every ``output/chunk_*.ply`` concatenated —
    so the viewer shows the cloud the moment the reconstruction stage ends instead of
    after VLM + SAM3 + CloudCompy (USER 2026-09-28: "si hay nube el avance debe
    mostrarse de otra forma, no me debe tapar la nube"). Built into ``output/potree``
    with :data:`PREVIEW_MARKER`; the CloudCompy build replaces it. False when there
    are no chunks, their layouts differ, or the CLEAN cloud already exists — the
    preview never lands on top of (or races) the octree of cleaned_cloud.ply: it
    holds the same cross-process lock as the clean build and re-checks under it."""
    output_dir = Path(session_dir) / "output"
    potree_dir = output_dir / "potree"
    cleaned = output_dir / "cleaned_cloud.ply"
    chunks = sorted(output_dir.glob("chunk_*.ply"))
    if not chunks:
        logger.warning(f"[Potree] preview: no chunk_*.ply in {output_dir}")
        return False
    if cleaned.exists():
        logger.info("[Potree] preview skipped: cleaned_cloud.ply exists — its octree is the one")
        return False
    key = str(output_dir.resolve())
    while True:
        with _potree_reg:
            if key not in _potree_active:
                _potree_active[key] = {"since": time.time(), "ply": "preview",
                                       "ply_mtime": 0.0}
                break
        time.sleep(3)
    try:
        with _potree_lock(output_dir):
            if cleaned.exists():
                logger.info("[Potree] preview skipped: cleaned_cloud.ply appeared while "
                            "waiting — its octree is the one")
                return False
            parts = [_read_ply_vertices(c) for c in chunks]
            if len({p.dtype for p in parts}) != 1:
                logger.warning("[Potree] preview: the chunks' PLY layouts differ — no preview")
                return False
            data = np.concatenate(parts)
            del parts
            with tempfile.TemporaryDirectory(dir=str(output_dir)) as tmpdir:
                tmp = Path(tmpdir)
                # classification.npy belongs to the clean cloud — never applied here
                n = _vertices_to_las(data, tmp / "preview.las", tmp,
                                     f"{len(chunks)} chunk PLY(s)")
                del data
                if not _run_potree_converter(tmp / "preview.las", tmp / "potree"):
                    return False
                (tmp / "potree" / PREVIEW_MARKER).write_text(
                    f"{n} points from {len(chunks)} chunks\n")
                if cleaned.exists():
                    logger.info("[Potree] preview discarded: cleaned_cloud.ply appeared "
                                "during the build — its octree is the one")
                    return False
                _swap_in(tmp / "potree", potree_dir, output_dir)
        logger.info(f"[Potree] ✅ preview octree of the raw reconstruction ({n:,} points)")
        return True
    except Exception as e:
        logger.error(f"[Potree] preview failed (non-fatal): {e}")
        return False
    finally:
        with _potree_reg:
            _potree_active.pop(key, None)


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
