"""
Data Validation - Strict data contract enforcement
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume']
REQUIRED_DTYPES = {
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',
}

def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame schema (required columns, dtypes).
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check required columns
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check dtypes
    for col in REQUIRED_COLS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} is not numeric: {df[col].dtype}")
    
    return len(errors) == 0, errors

def validate_sorted_unique(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> Tuple[bool, List[str]]:
    """
    Validate timestamps are sorted ascending and unique.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found")
        return False, errors
    
    # Check sorted
    if not df[timestamp_col].is_monotonic_increasing:
        errors.append("Timestamps not sorted ascending")
    
    # Check unique
    if df[timestamp_col].duplicated().any():
        dup_count = df[timestamp_col].duplicated().sum()
        errors.append(f"Found {dup_count} duplicate timestamps")
    
    return len(errors) == 0, errors

def validate_timeframe_spacing(
    df: pd.DataFrame,
    timeframe: str,
    timestamp_col: str = 'timestamp',
    tolerance_pct: float = 0.1,
    min_outlier_pct: float = 0.1,  # Minimum outlier % to trigger error
    max_outlier_count: int = 5      # Maximum outlier count to ignore
) -> Tuple[bool, List[str]]:
    """
    Validate timeframe spacing (expected delta between bars).
    
    Args:
        timeframe: Expected timeframe (e.g., '15m', '1h', '4h')
        tolerance_pct: Allowed deviation from expected spacing
        min_outlier_pct: Minimum outlier percentage to trigger error (default: 0.1%)
        max_outlier_count: Maximum outlier count to ignore (default: 5)
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if timestamp_col not in df.columns:
        errors.append(f"Timestamp column '{timestamp_col}' not found")
        return False, errors
    
    # Expected spacing in minutes
    timeframe_minutes = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
    }
    
    if timeframe not in timeframe_minutes:
        errors.append(f"Unknown timeframe: {timeframe}")
        return False, errors
    
    expected_minutes = timeframe_minutes[timeframe]
    
    # Calculate actual spacing
    if len(df) < 2:
        errors.append("Need at least 2 rows to validate spacing")
        return False, errors
    
    df_sorted = df.sort_values(timestamp_col)
    deltas = df_sorted[timestamp_col].diff().dropna()
    
    # Convert to minutes - handle all Timedelta cases properly
    if len(deltas) == 0:
        errors.append("No timestamp deltas to validate")
        return False, errors
    
    # Check dtype first (most reliable)
    if pd.api.types.is_timedelta64_dtype(deltas):
        # Timedelta64 dtype - convert to minutes
        deltas_minutes = deltas.dt.total_seconds() / 60
    elif hasattr(deltas, 'dt') and hasattr(deltas.dt, 'total_seconds'):
        # TimedeltaIndex or Series with Timedelta objects
        deltas_minutes = deltas.dt.total_seconds() / 60
    elif len(deltas) > 0 and isinstance(deltas.iloc[0], pd.Timedelta):
        # Timedelta objects in Series
        deltas_minutes = deltas.dt.total_seconds() / 60
    elif pd.api.types.is_datetime64_any_dtype(deltas):
        # Datetime64 dtype (shouldn't happen for diffs, but handle it)
        deltas_minutes = deltas.dt.total_seconds() / 60
    else:
        # Numeric - assume already in minutes or seconds
        try:
            # Convert to numeric safely
            deltas_numeric = pd.to_numeric(deltas, errors='coerce')
            max_val = float(deltas_numeric.max())
            # If max > 1000, assume seconds, else assume minutes
            deltas_minutes = deltas_numeric / 60 if max_val > 1000 else deltas_numeric
        except (TypeError, ValueError) as e:
            errors.append(f"Cannot convert deltas to numeric: {e}")
            return False, errors
    
    # Check if within tolerance
    expected_delta = expected_minutes
    tolerance = expected_delta * tolerance_pct
    
    outliers = deltas_minutes[
        (deltas_minutes < expected_delta - tolerance) |
        (deltas_minutes > expected_delta + tolerance)
    ]
    
    if len(outliers) > 0:
        outlier_pct = len(outliers) / len(deltas_minutes) * 100
        
        # CRITICAL FIX: Ignore small outliers (very common in real data)
        # If outlier % is very small (< min_outlier_pct) OR count is very small (< max_outlier_count),
        # don't treat it as an error - just log it as a warning
        if outlier_pct < min_outlier_pct and len(outliers) <= max_outlier_count:
            # This is acceptable - just log it but don't error
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Found {len(outliers)} ({outlier_pct:.2f}%) bars with spacing outside tolerance "
                f"(Expected: {expected_minutes} min ± {tolerance:.1f} min). "
                f"Ignoring due to small count/percentage."
            )
            # Don't add to errors - this is acceptable
        else:
            # Significant outliers - this is a real problem
            errors.append(
                f"Found {len(outliers)} ({outlier_pct:.1f}%) bars with spacing outside tolerance. "
                f"Expected: {expected_minutes} min ± {tolerance:.1f} min"
            )
    
    return len(errors) == 0, errors

def validate_no_leakage(
    df: pd.DataFrame,
    split_points: Dict[str, int],
    feature_cols: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate no forward-looking features (lookahead leakage).
    
    Args:
        split_points: Dict with 'train_end', 'val_end', 'test_start' indices
        feature_cols: List of feature column names to check
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # This is a simplified check - in practice, you'd need to verify
    # that features at index i don't use data from index > i
    
    # Check for common leakage patterns in column names
    leakage_patterns = ['future', 'forward', 'next', 'lead']
    for col in feature_cols:
        col_lower = col.lower()
        for pattern in leakage_patterns:
            if pattern in col_lower:
                errors.append(f"Potential leakage in column: {col}")
    
    return len(errors) == 0, errors

def validate_no_nan_inf(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """
    Validate no NaN or Inf in features and target.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check features
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
        
        if nan_count > 0:
            errors.append(f"Column {col}: {nan_count} NaN values")
        if inf_count > 0:
            errors.append(f"Column {col}: {inf_count} Inf values")
    
    # Check target
    if target_col and target_col in df.columns:
        nan_count = df[target_col].isna().sum()
        inf_count = np.isinf(df[target_col]).sum() if pd.api.types.is_numeric_dtype(df[target_col]) else 0
        
        if nan_count > 0:
            errors.append(f"Target {target_col}: {nan_count} NaN values")
        if inf_count > 0:
            errors.append(f"Target {target_col}: {inf_count} Inf values")
    
    return len(errors) == 0, errors

def validate_dataframe(
    df: pd.DataFrame,
    timeframe: str,
    feature_cols: List[str],
    target_col: Optional[str] = None,
    timestamp_col: str = 'timestamp'
) -> Tuple[bool, List[str]]:
    """
    Run all validations on a DataFrame.
    
    Returns:
        (is_valid, all_error_messages)
    """
    all_errors = []
    
    # Schema validation
    is_valid, errors = validate_schema(df)
    if not is_valid:
        all_errors.extend(errors)
    
    # Timestamp validation
    if timestamp_col in df.columns:
        is_valid, errors = validate_sorted_unique(df, timestamp_col)
        if not is_valid:
            all_errors.extend(errors)
        
        is_valid, errors = validate_timeframe_spacing(df, timeframe, timestamp_col)
        if not is_valid:
            all_errors.extend(errors)
    
    # NaN/Inf validation
    is_valid, errors = validate_no_nan_inf(df, feature_cols, target_col)
    if not is_valid:
        all_errors.extend(errors)
    
    return len(all_errors) == 0, all_errors

