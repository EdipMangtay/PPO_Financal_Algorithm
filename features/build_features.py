"""
Feature Engineering - Robust feature building with error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI with safe division."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD with safe operations."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd.fillna(0), macd_signal.fillna(0), macd_hist.fillna(0)

def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """Calculate ATR with safe operations."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.bfill().fillna(0)

def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper.fillna(prices), sma.fillna(prices), lower.fillna(prices)

def calculate_volume_ma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Volume MA."""
    return volume.rolling(window=period).mean().fillna(volume)

def build_features(
    df: pd.DataFrame,
    timeframe: str,
    feature_config: Dict,
    target_horizon_bars: int = 12
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    Build features and target from raw OHLCV data.
    
    Args:
        df: DataFrame with OHLCV columns
        timeframe: Timeframe string
        feature_config: Feature configuration dict
        target_horizon_bars: Number of bars ahead for target
    
    Returns:
        (features_df, target_series, metadata)
    """
    df = df.copy()
    
    # Get feature lists
    common_features = feature_config.get('features_common', [])
    timeframe_features = feature_config.get('features_by_timeframe', {}).get(timeframe, [])
    all_feature_names = list(set(common_features + timeframe_features))
    
    # Get feature parameters
    params = feature_config.get('feature_params', {})
    
    # Calculate features
    feature_dict = {}
    
    # RSI
    if 'RSI' in all_feature_names:
        rsi_period = params.get('rsi_period', 14)
        feature_dict['RSI'] = calculate_rsi(df['close'], rsi_period)
    
    # MACD
    if any(f in all_feature_names for f in ['MACD', 'MACD_signal', 'MACD_hist']):
        macd_fast = params.get('macd_fast', 12)
        macd_slow = params.get('macd_slow', 26)
        macd_signal = params.get('macd_signal', 9)
        macd, signal, hist = calculate_macd(df['close'], macd_fast, macd_slow, macd_signal)
        if 'MACD' in all_feature_names:
            feature_dict['MACD'] = macd
        if 'MACD_signal' in all_feature_names:
            feature_dict['MACD_signal'] = signal
        if 'MACD_hist' in all_feature_names:
            feature_dict['MACD_hist'] = hist
    
    # ATR
    if 'ATR' in all_feature_names:
        atr_period = params.get('atr_period', 14)
        feature_dict['ATR'] = calculate_atr(df['high'], df['low'], df['close'], atr_period)
    
    # Bollinger Bands
    if any(f in all_feature_names for f in ['BB_upper', 'BB_middle', 'BB_lower']):
        bb_period = params.get('bb_period', 20)
        bb_std = params.get('bb_std', 2.0)
        upper, middle, lower = calculate_bollinger_bands(df['close'], bb_period, bb_std)
        if 'BB_upper' in all_feature_names:
            feature_dict['BB_upper'] = upper
        if 'BB_middle' in all_feature_names:
            feature_dict['BB_middle'] = middle
        if 'BB_lower' in all_feature_names:
            feature_dict['BB_lower'] = lower
    
    # Volume
    if 'Volume' in all_feature_names:
        feature_dict['Volume'] = df['volume']
    
    # Volume MA
    if 'Volume_MA' in all_feature_names:
        vol_ma_period = params.get('volume_ma_period', 20)
        feature_dict['Volume_MA'] = calculate_volume_ma(df['volume'], vol_ma_period)
    
    # Create features DataFrame
    features_df = pd.DataFrame(feature_dict, index=df.index)
    
    # Add base columns
    base_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        base_cols.append('volume')
    
    for col in base_cols:
        if col in df.columns:
            features_df[col] = df[col]
    
    # CRITICAL: Preserve timestamp if it exists (needed for backtest alignment)
    if 'timestamp' in df.columns:
        features_df['timestamp'] = df['timestamp'].values
    
    # Calculate target (forward return)
    future_close = df['close'].shift(-target_horizon_bars)
    target = (future_close - df['close']) / df['close']
    
    # Handle NaN/Inf
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    target = target.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with insufficient history (warmup period)
    max_period = max(
        params.get('rsi_period', 14),
        params.get('macd_slow', 26),
        params.get('atr_period', 14),
        params.get('bb_period', 20),
        params.get('volume_ma_period', 20),
        target_horizon_bars
    )
    
    initial_rows = len(features_df)
    features_df = features_df.iloc[max_period:].copy()
    target = target.iloc[max_period:].copy()
    
    # Drop rows with NaN in features or target
    valid_mask = features_df.notna().all(axis=1) & target.notna()
    features_df = features_df[valid_mask].copy()
    target = target[valid_mask].copy()
    
    dropped_rows = initial_rows - len(features_df)
    
    # Metadata
    metadata = {
        'feature_list': list(features_df.columns),
        'num_features': len(features_df.columns),
        'initial_rows': initial_rows,
        'final_rows': len(features_df),
        'dropped_rows': dropped_rows,
        'missing_counts': features_df.isna().sum().to_dict(),
        'target_stats': {
            'mean': float(target.mean()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
        }
    }
    
    logger.info(
        f"Built {len(features_df.columns)} features, "
        f"{len(features_df)} rows (dropped {dropped_rows} warmup/invalid)"
    )
    
    return features_df, target, metadata

def save_feature_report(
    metadata: Dict,
    output_path: Path,
    timeframe: str
):
    """Save feature engineering report."""
    report = {
        'timeframe': timeframe,
        'metadata': metadata,
        'timestamp': pd.Timestamp.now().isoformat(),
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Saved feature report to {output_path}")

