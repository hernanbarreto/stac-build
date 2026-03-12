# STAC-BUILDER: Configuration Loader
import os
import yaml
from pathlib import Path

# Base directory for the server
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

# Runtime data directory (projects, personnel, teams, logs, stac.db)
DATA_DIR = BASE_DIR / "data"

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
