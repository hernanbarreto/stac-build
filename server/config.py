# STAC-BUILDER: Configuration Loader
import os
import yaml
from pathlib import Path

# Base directory for the server
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# Runtime data directory (personnel, teams, logs, stac.db)
DATA_DIR = BASE_DIR / "data"

# Projects directory — configurable via config.yaml paths.projects_dir
# Defaults to DATA_DIR / "projects" if not set (backwards compatible)
# NOTE: Do NOT check exists() here — WSL mounts external drives lazily,
# so /mnt/e may not be accessible at import time but works at runtime.
def _resolve_projects_dir():
    """Read projects_dir from config.yaml at import time."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            _raw = yaml.safe_load(f) or {}
        custom = (_raw.get("paths") or {}).get("projects_dir")
        if custom:
            return Path(custom)
    except Exception:
        pass
    return DATA_DIR / "projects"

PROJECTS_DIR = _resolve_projects_dir()

def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_PATH.exists():
        print(f"[Config] ⚠️ Config file not found at {CONFIG_PATH}. Using empty defaults.")
        return {}

    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[Config] Loaded configuration from {CONFIG_PATH}")
        return config
    except Exception as e:
        print(f"[Config] ❌ Error loading config: {e}")
        return {}

# Load global config
_config_data = load_config()

class DictConfig:
    """Helper to access dict keys as attributes (dot notation)."""
    def __init__(self, data):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, DictConfig(v))
            else:
                setattr(self, k, v)
    
    def get(self, key, default=None):
        return getattr(self, key, default)

# Convert to object for easier access (cfg.server.host instead of cfg['server']['host'])
# or just expose the dict. Let's keep it simple as a Dict or Wrapped object.
# Actually, keeping it as a dictionary is safer for robustness, but attribute access is cleaner.
# Let's verify if 'server' exists first to avoid crashes on partial configs.

# Helper function to get nested keys safely
def get_param(path: str, default=None):
    keys = path.split('.')
    val = _config_data
    try:
        for k in keys:
            val = val[k]
        return val
    except KeyError:
        return default

# Expose raw dictionary
cfg = _config_data
