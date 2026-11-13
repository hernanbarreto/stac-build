"""GenRecon RESUMES AT THE MESH STEP (USER 2026-11-09: "si algo falla se detiene se soluciona y se lanza desde el
punto donde falló"; 2026-11-12: "la regla es que se continúa desde el punto de falla").

run_cloud.sh runs adapter -> reconstruct_scene.py (pccr 2026-08-24: ~40 min of the card) -> the mesh step (fill
holes, remesh, per-chunk bakes, merge) -> publish, and every launch wiped the work folders and began again: on
2026-11-12 the mesh step failed AFTER the reconstruction had written its products and no launch could reuse them.
The reconstruction's products now carry a STAMP (repro.stamp) of what they are made from:

  inputs   the session files the adapter reads: the published cloud, the poses and their frame list, the camera,
           and when they exist the floor transform and the segmentation (the potree metadata when the order
           filters by confidence). cleaned_cloud_raw.ply is loaded by the session reader and never used: not here.
  code     the adapter's own files and every source file of vendor/GenRecon EXCEPT the mesh step's
           (MESH_STEP_FILES: they run after the reconstruction, so a fix there keeps the products)
  config   the order (the adapter's and reconstruct_scene.py's command lines exactly as the script runs them), the
           config.yaml values the adapter reads, the keyframes and the checkpoints by name and size

`check` computes it BEFORE the adapter runs (never after the fact: an input that changed during the run would be
stamped as what made the products), keeps it pending beside the output folder, and says whether the products on
disk carry that same stamp and are whole (the two files the mesh step loads by size, the work folder file by
file). `seal` writes the pending stamp beside the products once reconstruct_scene.py has succeeded. Every
difference is NAMED and the run then starts at the adapter, as always.

AT THE DECODE (USER 2026-11-13: "Debemos hacer entrar pccr2408 hay que ver la forma la técnica para que sea
escalable a nubes masivas"). reconstruct_scene.py is two steps: the GENERATION (chunks, images, the sampling of one
shape and one texture latent per chunk — the long part) and the DECODE of all of them together (the part that grows
with the scene). The sampled latents are written before the decode (latents.py, <O>/latents.pt) and carry a stamp
of their own — the same as above WITHOUT the decode's own code (decode_files(): the vendor's joint_decode.py and
decode_regions.py) and with the generation's command line (the reconstruction's without the decode's parameters:
genrecon.joint_decode_max_voxels). A run whose decode failed, or whose decode code or cap changed, continues at the
decode on those latents; anything the latents are made from that changed is named and the run starts at the adapter.

  python resume.py check <SES> <W> <O> --conf <c> --adapter "<adapter args>" --reconstruct "<command>"
                         [--generation "<command>"] [--dry]
        exit 0 = the products are reusable (the run continues at the mesh step), 4 = they are not but the sampled
        latents are (with --generation: the run continues at the decode), 3 = neither (the reasons printed).
        --dry: nothing is written or removed (asking is then safe beside a live run of the same session)
  python resume.py seal-latents <SES> <W> <O>      once reconstruct_scene.py has EXITED and left its latents
  python resume.py seal         <SES> <W> <O>      once it has succeeded
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent.parent
VENDOR = SERVER.parent / "vendor" / "GenRecon"
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

import repro  # noqa: E402

STAMP_NAME = "reconstruct.stamp.json"
#: exit code of `check` when the products are not reusable (0 = reusable, anything else = the check itself failed)
NOT_REUSABLE = 3
#: exit code of `check --generation` when the products are not reusable and the sampled latents are
AT_THE_DECODE = 4
#: the sampled latents in the reconstruction's output folder (tools/genrecon/latents.py) and their stamp
LATENTS_NAME = "latents.pt"
LATENTS_STAMP_NAME = "latents.stamp.json"
#: vendor files of the DECODE alone (after the sampling): a change there keeps the sampled latents
DECODE_STEP_FILES = ("genrecon/pipelines/joint_decode.py",)
#: what the mesh step loads from the reconstruction's output folder
PRODUCTS = ("to_glb_inputs.pt", "chunk_inputs.pt")
#: vendor files of the MESH step (after the reconstruction): a change there keeps the reconstruction's products
MESH_STEP_FILES = ("chunked_to_glb.py", "inference/chunked_glb.py")
#: genrecon.undistort_device as the worker hands it to GenRecon's own process (workers/genrecon_worker.py
#: UNDISTORT_ENV, tools/genrecon/remap_gpu.py ENV — one name, held equal by tests/test_remap_gpu.py)
UNDISTORT_ENV = "STAC_UNDISTORT_DEVICE"
#: top-level vendor folders that hold no source of the reconstruction
_VENDOR_SKIP = ("work", ".git", "checkpoints", "weights", "assets")
_VENDOR_SOURCE = (".py", ".json", ".yaml", ".yml")
#: the session files the adapter reads when they exist
_OPTIONAL_INPUTS = ("camera_frames.txt", "frame_list.json", "floor_transform.npz", "segmentation_result.json")


def pending_path(out: Path, name: str = STAMP_NAME) -> Path:
    """The stamp computed before the adapter, waiting for the reconstruction to succeed (``name``: the products'
    stamp, or LATENTS_STAMP_NAME for the sampled latents'): beside the output folder (the script wipes that folder
    before reconstruct_scene.py runs)."""
    return out.parent / f"{out.name}.{name}.pending"


def session_inputs(ses: Path, conf: float) -> Dict[str, Path]:
    out = ses / "output"
    ins = {"output/cleaned_cloud.ply": out / "cleaned_cloud.ply",
           "output/camera_poses.txt": out / "camera_poses.txt",
           "output/camera.json": out / "camera.json"}
    for name in _OPTIONAL_INPUTS:
        if (out / name).exists():
            ins[f"output/{name}"] = out / name
    meta = out / "potree" / "metadata.json"                  # the adapter reads it only to filter by confidence
    if conf > 0 and meta.exists():
        ins["output/potree/metadata.json"] = meta
    return ins


def keyframes(ses: Path) -> Dict[str, Optional[int]]:
    """The keyframes the adapter links into the work folder, each by its size (None = missing)."""
    from segmentation.session_io import _load_frame_index_map
    out: Dict[str, Optional[int]] = {}
    for fr in _load_frame_index_map(ses / "output") or []:
        p = ses / "frames" / f"{int(fr):06d}.jpg"
        out[p.name] = p.stat().st_size if p.exists() else None
    return out


def adapter_config() -> Dict[str, Any]:
    """The config.yaml values adapter.py reads, as it reads them (the order's own values — cube, confidence,
    points per voxel, mode — are in the command lines), and the device GenRecon's own process undistorts its
    frames on."""
    import yaml
    cfg = yaml.safe_load((SERVER / "config.yaml").read_text())
    g = cfg.get("genrecon") or {}
    if "adapter_device" not in g:
        raise RuntimeError("config.yaml: genrecon.adapter_device is missing (numpy | cpu | cuda)")
    return {"presence_vote": bool(g.get("presence_vote", False)),
            "presence_min_views": int(g.get("presence_min_views", 2)),
            "presence_device": str(g.get("presence_device", "cuda")),
            "adapter_device": str(g["adapter_device"]),
            # what GenRecon's own process is launched with (the worker writes genrecon.undistort_device of the
            # order's config into its environment; absent = numpy, the vendor's own lines — remap_gpu.device_from_env)
            "undistort_device": str(os.environ.get(UNDISTORT_ENV, "numpy")),
            "support_surfaces": sorted(str(x) for x in cfg["autoprompt"]["support_surfaces"])}


def vendor_sources(vendor: Optional[Path] = None) -> List[Path]:
    """Every source file of the vendor tree the reconstruction can run, the mesh step's left out."""
    vendor = VENDOR if vendor is None else vendor
    out: List[Path] = []
    for root, dirs, files in os.walk(vendor):
        top = Path(root) == vendor
        dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not (top and d in _VENDOR_SKIP))
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix in _VENDOR_SOURCE and p.relative_to(vendor).as_posix() not in MESH_STEP_FILES:
                out.append(p)
    return out


def code_files() -> List[Path]:
    return [HERE / "adapter.py", HERE / "presence.py", HERE / "genrecon_levels.py", HERE / "layout_gpu.py",
            HERE / "remap_gpu.py", SERVER / "jpeg_gpu.py",
            HERE / "cloud_shape.py",     # the shape from the cloud: reconstruct_scene.py imports it (USER 2026-11-12)
            HERE / "latents.py",         # the sampled latents on disk: reconstruct_scene.py writes and reads them
            HERE / "decode_regions.py",  # the decode by regions: the vendor's joint_decode.py loads it (2026-11-13)
            SERVER / "correction" / "session.py", SERVER / "correction" / "floor.py",
            SERVER / "segmentation" / "perfect_object.py", SERVER / "segmentation" / "session_io.py",
            *vendor_sources()]


def decode_files() -> List[Path]:
    """The code that runs at the DECODE alone, after the latents are sampled: a change there keeps the latents."""
    return [HERE / "decode_regions.py", *(VENDOR / m for m in DECODE_STEP_FILES)]


def generation_code_files() -> List[Path]:
    """What the sampled latents are made by: code_files() without the decode's own."""
    late = {p.resolve() for p in decode_files()}
    return [p for p in code_files() if p.resolve() not in late]


def checkpoints(vendor: Optional[Path] = None) -> Dict[str, Optional[int]]:
    """The model files by name and size (None = a dangling link)."""
    vendor = VENDOR if vendor is None else vendor
    out: Dict[str, Optional[int]] = {}
    for sub in ("checkpoints", "weights"):
        for root, dirs, files in os.walk(vendor / sub, followlinks=True):
            dirs.sort()
            for f in sorted(files):
                p = Path(root) / f
                out[p.relative_to(vendor).as_posix()] = p.stat().st_size if p.exists() else None
    return out


def compute(ses: Path, conf: float, adapter_args: str, reconstruct_cmd: str) -> Dict[str, Any]:
    return repro.stamp(inputs=session_inputs(ses, conf), code=code_files(),
                       config={"order": {"adapter": str(adapter_args), "reconstruct": str(reconstruct_cmd)},
                               "adapter_config": adapter_config(), "keyframes": keyframes(ses),
                               "checkpoints": checkpoints()})


def compute_generation(ses: Path, conf: float, adapter_args: str, generation_cmd: str) -> Dict[str, Any]:
    """The stamp of the SAMPLED LATENTS: compute() without the decode's code and with the generation's command."""
    return repro.stamp(inputs=session_inputs(ses, conf), code=generation_code_files(),
                       config={"order": {"adapter": str(adapter_args), "generation": str(generation_cmd)},
                               "adapter_config": adapter_config(), "keyframes": keyframes(ses),
                               "checkpoints": checkpoints()})


def products(work: Path, out: Path, names=PRODUCTS) -> Dict[str, Any]:
    """The reconstruction's products as they are on disk: the files the mesh step loads (None = missing) and the
    work folder file by file — a file by its size, a link by its target. ``names``: the output folder's files to
    list (the mesh step's two, or the sampled latents' one)."""
    got: Dict[str, Any] = {}
    for name in names:
        p = out / name
        got[f"out/{name}"] = p.stat().st_size if p.is_file() else None
    for root, dirs, files in os.walk(work):
        dirs.sort()
        for f in sorted(files):
            p = Path(root) / f
            got["work/" + p.relative_to(work).as_posix()] = ("-> " + os.readlink(p)) if p.is_symlink() else p.stat().st_size
    return got


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


def _reasons(saved_p: Path, now: Dict[str, Any], has: Dict[str, Any], what: str) -> List[str]:
    """Why what the stamp file ``saved_p`` describes (``what``) cannot be reused under the stamp ``now`` with
    ``has`` on disk ([] = reusable)."""
    if not saved_p.is_file():
        return [f"no {saved_p.name} beside {what} ({saved_p.parent})"]
    try:
        saved = json.loads(saved_p.read_text())
    except ValueError as e:
        return [f"{saved_p} is unreadable ({e})"]
    if not isinstance(saved, dict):
        return [f"{saved_p} is not a stamp record"]
    diffs = list(repro.check_stamp(saved.get("stamp"), now))
    was = saved.get("products")
    if not isinstance(was, dict):
        diffs.append("the saved stamp lists no products")
        return diffs
    gone = object()
    changed = sorted(k for k in set(was) | set(has) if was.get(k, gone) != has.get(k, gone))
    for k in changed[:8]:
        a, b = was.get(k, "absent"), has.get(k, "absent")
        diffs.append(f"product '{k}' was {a if a is not None else 'missing'}, is {b if b is not None else 'missing'}")
    if len(changed) > 8:
        diffs.append(f"… and {len(changed) - 8} more products differ")
    return diffs


def check(ses: Path, work: Path, out: Path, conf: float, adapter_args: str, reconstruct_cmd: str,
          dry: bool = False) -> List[str]:
    """Why the products in ``work`` / ``out`` cannot be reused for this order ([] = reusable). The order's stamp is
    left pending for :func:`seal` either way — unless ``dry``: then nothing is written."""
    now = compute(ses, conf, adapter_args, reconstruct_cmd)
    if not dry:
        _write_json(pending_path(out), now)
        # a generation stamp left pending by an earlier launch is not this order's: seal_latents never reads it
        pending_path(out, LATENTS_STAMP_NAME).unlink(missing_ok=True)
    return _reasons(out / STAMP_NAME, now, products(work, out), "the reconstruction's products")


def check_latents(ses: Path, work: Path, out: Path, conf: float, adapter_args: str, generation_cmd: str,
                  dry: bool = False) -> List[str]:
    """Why the sampled latents in ``out`` cannot be decoded for this order ([] = reusable: the run continues at the
    decode). The generation's stamp is left pending for :func:`seal_latents` either way — unless ``dry``. Call it
    AFTER :func:`check` (which clears the pending generation stamp of an earlier launch)."""
    now = compute_generation(ses, conf, adapter_args, generation_cmd)
    if not dry:
        _write_json(pending_path(out, LATENTS_STAMP_NAME), now)
    return _reasons(out / LATENTS_STAMP_NAME, now, products(work, out, (LATENTS_NAME,)), "the sampled latents")


def _seal(work: Path, out: Path, name: str, names) -> Path:
    pend = pending_path(out, name)
    if not pend.is_file():
        raise RuntimeError(f"{pend} is missing — `check` computes the stamp before the adapter runs; sealing without "
                           f"it would stamp the inputs after the fact")
    now = json.loads(pend.read_text())
    got = products(work, out, names)
    missing = [k for k in (f"out/{n}" for n in names) if got.get(k) is None]
    if missing:
        raise RuntimeError(f"the reconstruction left no {', '.join(missing)} in {out} — nothing to stamp")
    _write_json(out / name, {"stamp": now, "products": got})
    pend.unlink()
    return out / name


def seal(work: Path, out: Path) -> Path:
    """Write the pending stamp and the products' listing beside the products (after reconstruct_scene.py succeeded)."""
    return _seal(work, out, STAMP_NAME, PRODUCTS)


def seal_latents(work: Path, out: Path) -> Path:
    """Write the generation's pending stamp and the latents' listing beside the sampled latents — once
    reconstruct_scene.py has EXITED, whether its decode succeeded or not: a failed decode continues from them."""
    return _seal(work, out, LATENTS_STAMP_NAME, (LATENTS_NAME,))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GenRecon: reuse the reconstruction's products — or its sampled latents — "
                                             "when they carry this order's stamp")
    ap.add_argument("action", choices=("check", "seal", "seal-latents"))
    ap.add_argument("ses", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--adapter", default=None, help="the adapter's arguments as the script passes them")
    ap.add_argument("--reconstruct", default=None, help="reconstruct_scene.py's command line as the script runs it")
    ap.add_argument("--generation", default=None,
                    help="reconstruct_scene.py's command line without the decode's parameters: with it `check` also "
                         "says whether the sampled latents can be decoded (exit 4)")
    ap.add_argument("--dry", action="store_true", help="check: write and remove nothing")
    a = ap.parse_args(argv)
    if a.action == "seal":
        print(f"[resume] the reconstruction's products stamped: {seal(a.work, a.out)}", flush=True)
        return 0
    if a.action == "seal-latents":
        print(f"[resume] the sampled latents stamped: {seal_latents(a.work, a.out)}", flush=True)
        return 0
    if a.adapter is None or a.reconstruct is None:
        ap.error("check needs --adapter and --reconstruct")
    diffs = check(a.ses, a.work, a.out, a.conf, a.adapter, a.reconstruct, dry=a.dry)
    if not diffs:
        if not a.dry:
            pending_path(a.out).unlink()                     # nothing will be sealed: the products stay as stamped
        print("[resume] the reconstruction's products carry this order's stamp and are whole — reused, the run "
              "continues at the mesh step", flush=True)
        return 0
    late = None if a.generation is None else check_latents(a.ses, a.work, a.out, a.conf, a.adapter, a.generation,
                                                           dry=a.dry)
    if late is not None and not late:
        if not a.dry:                                        # the latents stay as stamped; the products' pending
            pending_path(a.out, LATENTS_STAMP_NAME).unlink() # stamp waits for `seal`, after the decode
        print("[resume] the reconstruction's products are not reusable:", flush=True)
        for d in diffs:
            print(f"[resume]   {d}", flush=True)
        print("[resume] the sampled latents carry this order's generation stamp and are whole — reused, the run "
              "continues at the decode", flush=True)
        return AT_THE_DECODE
    print("[resume] the reconstruction's products are not reusable — the run starts at the adapter:", flush=True)
    for d in diffs:
        print(f"[resume]   {d}", flush=True)
    if late is not None:
        print("[resume] nor are the sampled latents:", flush=True)
        for d in late:
            print(f"[resume]   {d}", flush=True)
    return NOT_REUSABLE


if __name__ == "__main__":
    sys.exit(main())
