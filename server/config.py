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

class ConfigError(RuntimeError):
    """config.yaml cannot be read, parsed, or is not a mapping: the server does not start."""


def load_config():
    """Load configuration from YAML file.

    DETERMINISM (docs/plan_determinismo.md points 112 / 151, 2026-10-08): a missing,
    unreadable or unparseable config.yaml — or one that is not a mapping — RAISES
    :class:`ConfigError` naming the file and the error. It used to become ``{}`` and every
    ``.get(key, default)`` caller then ran the server on defaults nobody declared; the order
    time already refuses (pipeline_manager.read_config_once) and the server start does the
    same: it never runs on a broken configuration."""
    if not CONFIG_PATH.exists():
        raise ConfigError(f"the configuration {CONFIG_PATH} does not exist — the server does not "
                          f"start on an empty configuration (point 151)")
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"the configuration {CONFIG_PATH} cannot be read ({e}) — the server "
                          f"does not start on defaults (point 151)") from e
    except yaml.YAMLError as e:
        raise ConfigError(f"the configuration {CONFIG_PATH} is not readable YAML ({e}) — the "
                          f"server does not start on defaults (point 151)") from e
    if not isinstance(config, dict):
        raise ConfigError(f"the configuration {CONFIG_PATH} holds a {type(config).__name__}, not "
                          f"a mapping — the server does not start on defaults (point 151)")
    print(f"[Config] Loaded configuration from {CONFIG_PATH}")
    return config

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
