"""
Dataset Caching - Cache processed datasets to disk
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_feature_hash(feature_config: Dict) -> str:
    """Compute hash of feature configuration"""
    config_str = json.dumps(feature_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:16]

def compute_dataset_hash(
    coin: str,
    timeframe: str,
    date_range: Dict[str, str],
    feature_hash: str
) -> str:
    """Compute hash for dataset cache key"""
    key = f"{coin}_{timeframe}_{date_range['start']}_{date_range['end']}_{feature_hash}"
    return hashlib.md5(key.encode()).hexdigest()[:16]

def get_cache_path(
    cache_dir: str,
    dataset_hash: str
) -> Path:
    """Get cache file path"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"dataset_{dataset_hash}.pkl"

def load_from_cache(
    cache_dir: str,
    dataset_hash: str
) -> Optional[Dict[str, Any]]:
    """
    Load dataset from cache.
    
    Returns:
        Cached data dict or None if not found
    """
    cache_file = get_cache_path(cache_dir, dataset_hash)
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded dataset from cache: {cache_file}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_file}: {e}")
        return None

def save_to_cache(
    cache_dir: str,
    dataset_hash: str,
    data: Dict[str, Any]
) -> Path:
    """
    Save dataset to cache.
    
    Returns:
        Path to cache file
    """
    cache_file = get_cache_path(cache_dir, dataset_hash)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved dataset to cache: {cache_file}")
        return cache_file
    except Exception as e:
        logger.error(f"Failed to save cache {cache_file}: {e}")
        raise

def is_cache_stale(
    cache_file: Path,
    source_files: list[Path]
) -> bool:
    """
    Check if cache is stale (source files newer than cache).
    
    Args:
        cache_file: Path to cache file
        source_files: List of source file paths
    
    Returns:
        True if cache is stale
    """
    if not cache_file.exists():
        return True
    
    cache_mtime = cache_file.stat().st_mtime
    
    for source_file in source_files:
        if source_file.exists():
            if source_file.stat().st_mtime > cache_mtime:
                return True
    
    return False





