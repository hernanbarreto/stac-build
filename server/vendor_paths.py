"""
Vendor Path Resolver — centralised path management for vendored dependencies.
==============================================================================

All external dependencies (DA3, SAM3, CloudComPy) live under
  stac-builder/vendor/

This module MUST be imported BEFORE any DA3/SAM3 imports. It:
  1. Resolves vendor paths relative to *this* file (portable)
  2. Adds them to sys.path in the correct order
  3. Exports constants for config / subprocess usage

Usage:
    import vendor_paths  # at top of any file that needs DA3/SAM3
"""

import sys
import os
from pathlib import Path

# ── Resolve vendor root relative to THIS file (server/vendor_paths.py) ──
_SERVER_DIR = Path(__file__).resolve().parent          # stac-builder/server/
_PROJECT_ROOT = _SERVER_DIR.parent                     # stac-builder/
VENDOR_ROOT = _PROJECT_ROOT / "vendor"                 # stac-builder/vendor/

# ── DA3 paths ────────────────────────────────────────────────────────
DA3_ROOT = VENDOR_ROOT / "depth_anything_3"            # source root
DA3_STREAMING = DA3_ROOT / "da3_streaming"             # streaming pipeline
DA3_SRC = DA3_ROOT / "src"                             # depth_anything_3 package
DA3_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "da3"    # model weights (gitignored)

# ── SAM3 paths ───────────────────────────────────────────────────────
SAM3_ROOT = VENDOR_ROOT / "sam3"                       # contains sam3/ package
SAM3_WEIGHTS_DIR = _PROJECT_ROOT / "weights" / "sam3"  # model weights (gitignored)

# ── CloudComPy paths ─────────────────────────────────────────────────
CLOUDCOMPY_ROOT = VENDOR_ROOT / "cloudcompy"
CLOUDCOMPY_BIN = CLOUDCOMPY_ROOT / "bin"
CLOUDCOMPY_LIB = CLOUDCOMPY_ROOT / "lib"

# ── Add to sys.path (order matters!) ─────────────────────────────────
# DA3: needs DA3_ROOT first (so `da3_streaming` resolves to package dir),
#       then DA3_STREAMING (for loop_utils), then DA3_SRC.

_paths_to_add = [
    str(DA3_SRC),          # depth_anything_3 package
    str(DA3_STREAMING),    # loop_utils, fastloop, etc.
    str(DA3_ROOT),         # MUST be first so da3_streaming/ resolves as package
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
