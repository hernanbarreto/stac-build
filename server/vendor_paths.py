"""
Vendor Path Resolver — centralised path management for vendored dependencies.
==============================================================================

All external dependencies (SAM3, CloudComPy) live under
  stac-builder/vendor/

This module MUST be imported BEFORE any SAM3 imports. It:
  1. Resolves vendor paths relative to *this* file (portable)
  2. Adds them to sys.path in the correct order
  3. Exports constants for config / subprocess usage

Usage:
    import vendor_paths  # at top of any file that needs SAM3
"""

import sys
import os
from pathlib import Path

# ── Resolve vendor root relative to THIS file (server/vendor_paths.py) ──
_SERVER_DIR = Path(__file__).resolve().parent          # stac-builder/server/
_PROJECT_ROOT = _SERVER_DIR.parent                     # stac-builder/
VENDOR_ROOT = _PROJECT_ROOT / "vendor"                 # stac-builder/vendor/

# ── SAM3 paths (version-switched) ────────────────────────────────────
# `models.segmentation.version` in config.yaml picks the checkout:
#   "sam3"   → vendor/sam3   (pinned SAM 3.0 — the validated baseline)
#   "sam3.1" → vendor/sam31  (upstream main with Object Multiplex)
# Read the YAML directly (NOT via config.py) to avoid an import cycle:
# this module must run before anything else imports the sam3 package.
def _segmentation_version() -> str:
    try:
        import yaml
        with open(_SERVER_DIR / "config.yaml") as f:
            c = yaml.safe_load(f) or {}
        return str(((c.get("models") or {}).get("segmentation") or {})
                   .get("version", "sam3"))
    except Exception:
        return "sam3"


SEGMENTATION_VERSION = _segmentation_version()
SAM3_ROOT = VENDOR_ROOT / ("sam31" if SEGMENTATION_VERSION == "sam3.1" else "sam3")
SAM3_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "sam3"    # 3.0 weights (HF cache fallback)
SAM31_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "sam31"  # sam3.1_multiplex.pt

# ── CloudComPy paths ─────────────────────────────────────────────────
CLOUDCOMPY_ROOT = VENDOR_ROOT / "cloudcompy"
CLOUDCOMPY_BIN = CLOUDCOMPY_ROOT / "bin"
CLOUDCOMPY_LIB = CLOUDCOMPY_ROOT / "lib"

# ── Add to sys.path (order matters!) ─────────────────────────────────
_paths_to_add = [
    str(SAM3_ROOT),        # so `from sam3.model_builder import ...` works
]

for p in _paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

# Verify vendor directory exists (warn but don't fail — allows tests without vendor)
if not VENDOR_ROOT.exists():
    import warnings
    warnings.warn(
        f"Vendor directory not found at {VENDOR_ROOT}. "
        "Run from stac-builder root or check project structure.",
        RuntimeWarning
    )
