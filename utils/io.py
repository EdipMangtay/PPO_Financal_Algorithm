"""
IO Utilities - Safe file operations
"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def save_json(data: Any, path: Path, indent: int = 2):
    """Save data to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug(f"Saved JSON to {path}")

def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_yaml(data: Any, path: Path):
    """Save data to YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    logger.debug(f"Saved YAML to {path}")

def load_yaml(path: Path) -> Any:
    """Load data from YAML file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_pickle(data: Any, path: Path):
    """Save data to pickle file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.debug(f"Saved pickle to {path}")

def load_pickle(path: Path) -> Any:
    """Load data from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


