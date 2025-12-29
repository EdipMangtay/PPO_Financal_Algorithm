"""
Timeframe Resampling - Safe OHLCV resampling
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {
    '15m': 15,
    '1h': 60,
    '4h': 240,
    '1d': 1440,
}

def resample_ohlcv(
    df: pd.DataFrame,
    target_timeframe: str,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe using proper OHLCV rules.
    
    Rules:
    - Open: First open of the period
    - High: Maximum high of the period
    - Low: Minimum low of the period
    - Close: Last close of the period
    - Volume: Sum of volumes
    
    Args:
        df: DataFrame with OHLCV columns
        target_timeframe: Target timeframe ('15m', '1h', '4h', '1d')
        timestamp_col: Name of timestamp column
    
    Returns:
        Resampled DataFrame
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found")
    
    if target_timeframe not in TIMEFRAME_MINUTES:
        raise ValueError(f"Unknown timeframe: {target_timeframe}")
    
    # Ensure timestamp is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Set timestamp as index
    df_indexed = df.set_index(timestamp_col)
    
    # Resample
    resampled = df_indexed.resample(target_timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    
    # Drop rows with NaN (incomplete periods)
    resampled = resampled.dropna()
    
    # Reset index
    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={timestamp_col: 'timestamp'})
    
    logger.info(f"Resampled {len(df)} bars to {len(resampled)} {target_timeframe} bars")
    
    return resampled

def resample_if_needed(
    df: pd.DataFrame,
    current_timeframe: str,
    target_timeframe: str,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Resample only if target timeframe is longer than current.
    
    Returns:
        Resampled DataFrame if needed, otherwise original
    """
    if current_timeframe == target_timeframe:
        return df
    
    current_min = TIMEFRAME_MINUTES.get(current_timeframe, 0)
    target_min = TIMEFRAME_MINUTES.get(target_timeframe, 0)
    
    if target_min < current_min:
        logger.warning(
            f"Cannot resample from {current_timeframe} to {target_timeframe} "
            "(target is shorter). Returning original."
        )
        return df
    
    if target_min == current_min:
        return df
    
    return resample_ohlcv(df, target_timeframe, timestamp_col)




