"""
Unified Data Loader - Loads raw data from parquet files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_raw(
    coin: str,
    timeframe: str,
    date_range: Dict[str, str],
    data_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Load raw OHLCV data from parquet file.
    
    Args:
        coin: Coin symbol (e.g., "BTC/USDT")
        timeframe: Timeframe (e.g., "15m", "1h", "4h")
        date_range: Dict with 'start' and 'end' dates (YYYY-MM-DD)
        data_dir: Directory containing parquet files
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    # Convert coin symbol to filename format
    coin_clean = coin.replace('/', '_')
    filename = f"{coin_clean}_{timeframe}.parquet"
    filepath = Path(data_dir) / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Load parquet
    df = pd.read_parquet(filepath)
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        # Try to infer timestamp from index or other columns
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            if 'timestamp' not in df.columns:
                df['timestamp'] = df.index
        else:
            # Try common timestamp column names
            for col in ['time', 'datetime', 'date', 'Time', 'DateTime']:
                if col in df.columns:
                    df = df.rename(columns={col: 'timestamp'})
                    break
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Filter by date range
    start_date = pd.to_datetime(date_range['start'])
    end_date = pd.to_datetime(date_range['end'])
    
    df = df[
        (df['timestamp'] >= start_date) &
        (df['timestamp'] <= end_date)
    ].copy()
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Select only required columns
    df = df[['timestamp'] + required_cols].copy()
    
    logger.info(
        f"Loaded {len(df)} bars for {coin} {timeframe} "
        f"from {df['timestamp'].min()} to {df['timestamp'].max()}"
    )
    
    return df

def load_or_resample(
    coin: str,
    target_timeframe: str,
    date_range: Dict[str, str],
    data_dir: str = "data/raw",
    base_timeframe: str = "15m"
) -> pd.DataFrame:
    """
    Load data for target timeframe, resampling from base if needed.
    
    Args:
        coin: Coin symbol
        target_timeframe: Desired timeframe
        date_range: Date range dict
        data_dir: Data directory
        base_timeframe: Base timeframe to resample from (if target not available)
    
    Returns:
        DataFrame for target timeframe
    """
    # Try to load target timeframe directly
    try:
        return load_raw(coin, target_timeframe, date_range, data_dir)
    except FileNotFoundError:
        logger.info(
            f"{coin} {target_timeframe} not found, resampling from {base_timeframe}"
        )
        
        # Load base timeframe
        df_base = load_raw(coin, base_timeframe, date_range, data_dir)
        
        # Resample
        from .resample import resample_ohlcv
        df_resampled = resample_ohlcv(df_base, target_timeframe)
        
        return df_resampled


