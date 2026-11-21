"""Shared by the version-2 scripts: the cube layout of a session the pipeline reconstructed."""
import json
from pathlib import Path

import yaml


def config_path(out: Path) -> Path:
    """The fork's configuration of the session's run (the pipeline writes it in output/)."""
    for p in (Path(out) / "vggt_omega_config.yaml", Path(out) / "maplong_run" / "vggt_omega_config.yaml"):
        if p.exists():
            return p
    raise FileNotFoundError(f"no vggt_omega_config.yaml under {out}")


def layout_of(out: Path):
    """(cube sets: the keyframe indices each cube holds, in the cube's own order; the owner cube of every keyframe;
    the keyframes' file names in order) — what the pipeline handed the fork for this session."""
    cfg = yaml.safe_load(config_path(out).read_text())
    names = json.loads((Path(out) / "maplong_run" / "frame_list.json").read_text())
    sets = [[int(g) for g in F] for F in cfg["Model"]["chunk_sets"]]
    owner = [int(x) for x in cfg["Model"]["chunk_owner"]]
    if len(owner) != len(names):
        raise RuntimeError(f"{len(owner)} owners for {len(names)} keyframes")
    return sets, owner, names
