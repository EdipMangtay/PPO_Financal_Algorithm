"""
Smoke test - Tiny sample training and backtest
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_smoke_train_backtest():
    """Smoke test with tiny dataset."""
    # Create tiny synthetic dataset
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    # Generate synthetic OHLCV
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 0.1)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(n_samples) * 1000,
    })
    
    # Save to temp file
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / "test_data.parquet"
    df.to_parquet(temp_file)
    
    # Test feature building
    from features.build_features import build_features
    import yaml
    
    with open("config/features.yaml", 'r') as f:
        feature_config = yaml.safe_load(f)
    
    features_df, target, metadata = build_features(
        df,
        timeframe='15m',
        feature_config=feature_config,
        target_horizon_bars=12
    )
    
    assert len(features_df) > 0
    assert len(target) > 0
    assert not features_df.isna().all().all()
    
    # Test backtest engine
    from backtest.engine import BacktestEngine
    
    signals = pd.Series(np.random.choice([-1, 0, 1], size=len(df)))
    engine = BacktestEngine(initial_balance=10000.0)
    
    results = engine.run_backtest(df, signals)
    
    assert 'metrics' in results
    assert 'trades' in results
    assert 'equity_curve' in results
    assert len(results['equity_curve']) == len(df) + 1
    
    print("✓ Smoke test passed")

if __name__ == "__main__":
    test_smoke_train_backtest()



