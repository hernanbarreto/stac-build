# _mast3r_paths.py
# Bootstrap module for MASt3R-SLAM path configuration
# MUST be imported BEFORE any mast3r_slam, mast3r, dust3r, or croco imports
#
# Usage: At the very top of any module that needs mast3r:
#   import _mast3r_paths  # noqa: F401 - sets up sys.path

import sys
from pathlib import Path

MAST3R_SLAM_PATH = Path("/home/hernan/mast3r_slam")

# Detect and handle 'models' namespace collision (DA3/Salad vs Croco)
if 'models' in sys.modules:
    m = sys.modules['models']
    m_file = getattr(m, '__file__', '')
    if 'salad' in m_file or 'da3' in m_file.lower():
        print(f"[MASt3R Paths] ⚠️ Detected conflicting 'models' package from DA3/Salad: {m_file}")
        print("[MASt3R Paths] Unloading it to allow MASt3R (croco) models to load...")
        del sys.modules['models']


# Paths required for MASt3R-SLAM and its dependencies
# Order matters! croco must come first to resolve 'models.dpt_block'
MAST3R_PATHS = [
    str(MAST3R_SLAM_PATH / "thirdparty" / "mast3r" / "dust3r" / "croco"),  # Contains 'models' package
    str(MAST3R_SLAM_PATH / "thirdparty" / "mast3r" / "dust3r"),  # dust3r
    str(MAST3R_SLAM_PATH / "thirdparty" / "mast3r"),  # mast3r
    str(MAST3R_SLAM_PATH),  # mast3r_slam itself
]

# Insert paths at the START of sys.path (in order, so first stays first)
for path in MAST3R_PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)
    else:
        # Move to front if already present but not at the start
        idx = sys.path.index(path)
        if idx > 0:
            sys.path.remove(path)
            sys.path.insert(0, path)

print(f"[MASt3R Paths] Configured sys.path with {len(MAST3R_PATHS)} MASt3R paths")
