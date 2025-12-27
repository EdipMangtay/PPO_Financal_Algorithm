"""
Unit test for Timedelta validation logic
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from data.validators import validate_timeframe_spacing

def test_validate_timeframe_spacing_15m():
    """Test 15m timeframe spacing validation."""
    # Create test data with 15-minute intervals
    timestamps = pd.date_range('2023-01-01', periods=100, freq='15min')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.random.randn(100),
    })
    
    is_valid, errors = validate_timeframe_spacing(df, '15m', 'timestamp')
    assert is_valid, f"Validation failed: {errors}"
    print("[OK] 15m timeframe validation passed")

def test_validate_timeframe_spacing_1h():
    """Test 1h timeframe spacing validation."""
    timestamps = pd.date_range('2023-01-01', periods=100, freq='1h')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.random.randn(100),
    })
    
    is_valid, errors = validate_timeframe_spacing(df, '1h', 'timestamp')
    assert is_valid, f"Validation failed: {errors}"
    print("[OK] 1h timeframe validation passed")

def test_validate_timeframe_spacing_4h():
    """Test 4h timeframe spacing validation."""
    timestamps = pd.date_range('2023-01-01', periods=100, freq='4h')
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.randn(100),
        'high': np.random.randn(100),
        'low': np.random.randn(100),
        'close': np.random.randn(100),
        'volume': np.random.randn(100),
    })
    
    is_valid, errors = validate_timeframe_spacing(df, '4h', 'timestamp')
    assert is_valid, f"Validation failed: {errors}"
    print("[OK] 4h timeframe validation passed")

if __name__ == "__main__":
    test_validate_timeframe_spacing_15m()
    test_validate_timeframe_spacing_1h()
    test_validate_timeframe_spacing_4h()
    print("\n[OK] All Timedelta validation tests passed!")

