"""
STAC Build — Potree Converter Wrapper
Converts cleaned_cloud.ply → LAS → Potree 2.0 octree

Hernán Barreto — Ingerop IN3 Session IV
"""

import asyncio
import errno
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Awaitable, Tuple

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


def _ply_to_las(ply_path: Path, las_path: Path, class_dir: Optional[Path] = None) -> int:
    """Convert binary PLY (with or without origin/confidence fields) to LAS 1.4 with RGB.

    If PLY contains a `confidence` field, it is mapped to LAS `intensity` (uint16, 0–65535).
    ``class_dir`` is where ``classification.npy`` / ``instance_ids.npy`` are read from (the
    PLY's own directory when not given — the projection stages them elsewhere, point 122).
    Returns the number of points converted.
    """
    return _vertices_to_las(_read_ply_vertices(ply_path), las_path,
                            Path(class_dir) if class_dir is not None else ply_path.parent,
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

    # Per-point INSTANCE id as a 32-bit extra dim (docs/plan_determinismo.md point 107: no
    # 255-object ceiling — segmentation.republish.write_classification writes instance_ids.npy
    # next to the uint8 classification.npy; the viewer keys on `instance` when the octree has it)
    inst_npy = Path(class_dir) / "instance_ids.npy"
    if inst_npy.exists():
        inst_arr = np.load(inst_npy)
        if len(inst_arr) == len(data):
            if inst_arr.size and int(inst_arr.max()) > int(np.iinfo(np.uint32).max):
                raise ValueError(f"{inst_npy}: instance id {int(inst_arr.max())} does not fit uint32")
            las.add_extra_dim(laspy.ExtraBytesParams(name="instance", type=np.uint32))
            las.instance = inst_arr.astype(np.uint32)
            logger.info(f"[Potree] Instance ids written as LAS extra dim 'instance' (uint32, max "
                        f"{int(inst_arr.max()) if inst_arr.size else 0})")
        else:
            logger.warning(f"[Potree] instance_ids.npy size mismatch: {len(inst_arr)} vs {len(data)} pts — not written")

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


# ── one thread (docs/plan_determinismo.md point 57) ──────────────────────────────────────
# The vendored converter (vendor/PotreeConverter, built with the STAC patches of 2026-10-07)
# reads POTREE_NUM_THREADS: the chunker's and the indexer's worker count. MEASURED 2026-10-07 on
# a 2.2 M-point synthetic cloud with exact coordinate ties: at the vendor's default (every
# hardware thread) three conversions gave three different octree.bin / hierarchy.bin — the node
# byte offsets are handed out by an atomic fetch_add in the order the threads FINISH; pinned to
# ONE thread, three conversions were byte-identical (4.0 s vs 2.2 s on 16 cores). The octree is a
# product of the epoch (what the viewer shows), so it is built on one thread, always.
POTREE_THREADS_ENV = "POTREE_NUM_THREADS"
POTREE_THREADS = "1"
#: the converter's encoding flag — part of the octree's identity (the stamp below)
POTREE_ENCODING = "UNCOMPRESSED"
# Point 103 (the converter's offsets follow the threads' completion order; the chunker's and
# the samplers' orders too) is CLOSED by point 57's own rule — "si difieren, fijar los hilos":
# the thread count is pinned HERE, in code, on every launch (potree_env), never left to the
# caller's shell, and recorded in the octree's stamp; the vendored source also carries the
# canonical-layout patches (tests/test_p3_potree_canonical.py). No further C++ change.


def potree_env() -> dict:
    """The environment every PotreeConverter run gets: the caller's, with the thread count
    pinned (never inherited from the shell — a stray value would change the bytes)."""
    env = dict(os.environ)
    env[POTREE_THREADS_ENV] = POTREE_THREADS
    return env


# ── the octree's identity (docs/plan_determinismo.md point 123) ───────────────────────────
# An octree is a function of the cloud it was built from, the per-point class / instance
# arrays baked into it, the converter binary and its launch parameters. Its STAMP records the
# sha256 of each, INSIDE the octree directory (`<potree_dir>/stac_stamp.json`), written by the
# build into the temporary directory BEFORE the swap — so it travels with the octree it
# describes wherever that directory is renamed: a correction / precision transaction
# (`_tx_epoch_<N>/potree` → `output/potree`), `select_epoch`, an `_epoch_<N>/` store. A stamp
# beside the directory (`output/potree_stamp.json`, where it first lived on 2026-10-08) was
# discarded by every transaction swap, so the cloud stage rebuilt f6_bend's own octree on
# every production run (zaragoza 114 M points ≈ 4 min on one thread). That name stays as the
# PROJECTION's product record (:data:`POTREE_STAMP_NAME`, the file the pipeline manager
# lists among the projection's products: the stamp of the octree the result was published
# with, a copy). An octree is reused ONLY on an identical stamp. No file date decides
# anything: the mtime tests this replaced (newer metadata.json than the PLY) reused an octree
# that another classification, converter or cloud of the same age had produced, and could not
# tell a copied or restored file from a fresh one.
POTREE_STAMP_NAME = "potree_stamp.json"
OCTREE_STAMP_NAME = "stac_stamp.json"
POTREE_STAMP_VERSION = 1


def potree_stamp_path(potree_dir: Path) -> Path:
    """The stamp file of an octree directory: inside it, :data:`OCTREE_STAMP_NAME`."""
    return Path(potree_dir) / OCTREE_STAMP_NAME


def potree_stamp_for(ply_path: Path, class_dir: Optional[Path] = None,
                     cloud_sha256: Optional[str] = None) -> dict:
    """The stamp a build from ``ply_path`` (with the class arrays of ``class_dir``, the PLY's
    directory by default — what :func:`_vertices_to_las` bakes in) produces. A missing class
    array is recorded as None: an octree without classification is a different octree.
    ``cloud_sha256``: the cloud's digest when the caller already holds it (the projection's
    result stamp hashes the same file) — a multi-gigabyte cloud is not read again for a
    number that is known; never a memo keyed by file dates."""
    from repro import sha256_file
    ply_path = Path(ply_path)
    class_dir = Path(class_dir) if class_dir is not None else ply_path.parent
    if not ply_path.is_file():
        raise FileNotFoundError(f"{ply_path} does not exist — no octree stamp without its cloud")

    def _opt(name: str) -> Optional[str]:
        p = class_dir / name
        return sha256_file(p) if p.is_file() else None

    return {"stamp_version": POTREE_STAMP_VERSION,
            "cloud_sha256": str(cloud_sha256) if cloud_sha256 else sha256_file(ply_path),
            "classification_sha256": _opt("classification.npy"),
            "instance_ids_sha256": _opt("instance_ids.npy"),
            "converter_sha256": sha256_file(POTREE_BIN) if POTREE_BIN.is_file() else None,
            "threads": POTREE_THREADS,
            "encoding": POTREE_ENCODING}


def read_potree_stamp(potree_dir: Path) -> Optional[dict]:
    """The saved stamp of an octree directory, None when there is none or it is unreadable."""
    p = potree_stamp_path(potree_dir)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _write_json_atomic(p: Path, doc: dict) -> Path:
    """``doc`` as sorted, indented JSON at ``p``: a temporary file in the same directory, one
    rename — a reader never sees half a stamp."""
    text = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def write_potree_stamp(potree_dir: Path, stamp: dict) -> Path:
    """Write the stamp of the octree in ``potree_dir`` (inside it, atomically)."""
    potree_dir = Path(potree_dir)
    if not potree_dir.is_dir():
        raise FileNotFoundError(f"{potree_dir} is not a directory — an octree stamp lives inside "
                                f"the octree it describes")
    return _write_json_atomic(potree_stamp_path(potree_dir), stamp)


def write_octree_record(output_dir: Path, stamp: dict) -> Path:
    """The PROJECTION's record of the octree it was published with — ``output/potree_stamp.json``
    (:data:`POTREE_STAMP_NAME`, a product of the projection: removed with the result, listed by
    the pipeline manager). A copy of the octree's own stamp; nothing decides on it."""
    return _write_json_atomic(Path(output_dir) / POTREE_STAMP_NAME, stamp)


def octree_matches(potree_dir: Path, ply_path: Path, class_dir: Optional[Path] = None,
                   cloud_sha256: Optional[str] = None) -> Tuple[bool, str]:
    """(reusable, reason): may the octree in ``potree_dir`` stand for a build from ``ply_path``
    + the class arrays of ``class_dir``? Only when it exists, is not a chunk preview, and its
    saved stamp EQUALS the one such a build would write (``cloud_sha256``: the cloud's digest
    when already known, see :func:`potree_stamp_for`)."""
    potree_dir = Path(potree_dir)
    if not (potree_dir / "metadata.json").is_file():
        return False, "no octree (metadata.json missing)"
    if (potree_dir / PREVIEW_MARKER).exists():
        return False, "the octree is a preview of the raw chunks"
    saved = read_potree_stamp(potree_dir)
    if saved is None:
        return False, f"the octree has no stamp ({potree_dir.name}/{OCTREE_STAMP_NAME} missing or " \
                      f"unreadable — built before 2026-10-08, or by hand)"
    now = potree_stamp_for(ply_path, class_dir, cloud_sha256)
    diffs = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
    if diffs:
        return False, "the octree's stamp differs (" + ", ".join(diffs) + ")"
    return True, "identical stamp (cloud, class arrays, converter, threads, encoding)"


def _run_potree_converter(las_path: Path, output_dir: Path) -> bool:
    """Run PotreeConverter 2.1 CLI on a LAS file, on ONE thread (point 57).

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
        "--encoding", POTREE_ENCODING,
    ]

    logger.info(f"[Potree] Running: {' '.join(cmd)} ({POTREE_THREADS_ENV}={POTREE_THREADS})")

    # NO wall-clock timeout (2026-09-28): 600 s on a slow disk discarded a whole
    # correction epoch as if its geometry were wrong. Duration is not a verdict;
    # a hung converter is the heartbeat/cancel path's to detect and kill.
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=potree_env(),
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
                          potree_dir_override: Path = None, class_dir: Path = None) -> bool:
    """Full pipeline: PLY → LAS → Potree octree.

    Args:
        session_dir: Path to session dir (e.g. server/scans/live_xxx)
        force: If True, always reconvert; False reuses the octree ONLY when its stamp
            equals the one this build would write (:func:`octree_matches`, point 123).
        ply_override: Optional path to use instead of cleaned_cloud.ply
        potree_dir_override: Optional target directory instead of
            output/potree — used by the correction module to build the octree
            INSIDE its transaction directory before the atomic swap.
        class_dir: where classification.npy / instance_ids.npy are read from
            (the PLY's directory by default; the projection stages them, point 122).

    Returns:
        True if potree/ directory was created successfully.
    """
    output_dir = session_dir / "output"
    ply_path = Path(ply_override) if ply_override else output_dir / "cleaned_cloud.ply"
    potree_dir = Path(potree_dir_override) if potree_dir_override else output_dir / "potree"
    class_dir = Path(class_dir) if class_dir is not None else None

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
                # the build we waited for may have produced exactly what we need: the
                # SAME stamp (cloud, class arrays, converter, launch parameters) → its
                # octree is this build's, byte for byte (point 123 — a newer file date
                # used to decide here, even under force=True)
                ok, why = octree_matches(potree_dir, ply_path, class_dir)
                if ok:
                    logger.info(f"[Potree] the finished build already covers this cloud — "
                                f"reusing it ({why})")
                    return True
            return _convert_ply_to_potree_inner(
                output_dir, ply_path, potree_dir, force, class_dir)
    finally:
        with _potree_reg:
            _potree_active.pop(key, None)


def _convert_ply_to_potree_inner(output_dir: Path, ply_path: Path,
                                 potree_dir: Path, force: bool,
                                 class_dir: Optional[Path] = None) -> bool:

    # Reuse ONLY on an identical stamp (point 123): the cloud's bytes, the class arrays
    # baked in, the converter and its launch parameters. A preview octree of the raw
    # chunks, an unstamped octree or any difference → rebuild. No file date decides.
    if not force:
        ok, why = octree_matches(potree_dir, ply_path, class_dir)
        if ok:
            logger.info(f"[Potree] octree reused — {why}")
            return True
        logger.info(f"[Potree] octree rebuilt — {why}")

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

            # the stamp is taken BEFORE the build from the very files it reads: what the
            # octree is a function of, as it was converted
            stamp = potree_stamp_for(ply_path, class_dir)
            n_points = (_ply_to_las(ply_path, tmp_las_path) if class_dir is None
                        else _ply_to_las(ply_path, tmp_las_path, class_dir))
            logger.info(f"[Potree] Converted {n_points:,} points to LAS in RAM/tmp")

            success = _run_potree_converter(tmp_las_path, tmp_potree_dir)

            if success:
                # the stamp is written INTO the new octree before it moves: the directory
                # and its identity are swapped in as one, and travel together through
                # every later rename (a transaction swap, select_epoch); the old octree
                # leaves with its own stamp inside it
                write_potree_stamp(tmp_potree_dir, stamp)
                # the light cloud the viewer draws the instant the session opens (USER
                # 2026-10-09) — the octree's own top levels, written beside it so it travels
                # with it; a viewer convenience, never a reason to lose the octree
                try:
                    from potree_preview import write_preview
                    write_preview(tmp_potree_dir)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[Potree] DECLARED: preview.bin not written ({e}) — the "
                                   f"viewer builds it from the octree when the session opens")
                _swap_in(tmp_potree_dir, potree_dir, output_dir)
                logger.info(f"[Potree] octree swapped into {potree_dir} "
                            f"(stamp {OCTREE_STAMP_NAME} inside it: cloud "
                            f"{stamp['cloud_sha256'][:12]}, classification "
                            f"{str(stamp['classification_sha256'])[:12]})")

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

    # Reuse ONLY on an identical stamp (point 123) — the same rule as the session octree
    if not force:
        ok, why = octree_matches(potree_dir, ply_path)
        if ok:
            logger.info(f"[Potree] Sábana octree reused — {why}")
            return True

    logger.info("[Potree] 🌲 Starting sábana PLY → Potree conversion...")

    las_path = session_dir / "sabana_cloud.las"

    try:
        stamp = potree_stamp_for(ply_path)
        n_points = _ply_to_las(ply_path, las_path)
        logger.info(f"[Potree] Converted {n_points:,} sábana points to LAS")
        # the converter removes the previous directory (its stamp inside it) before it writes
        success = _run_potree_converter(las_path, potree_dir)
        if success:
            write_potree_stamp(potree_dir, stamp)
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
