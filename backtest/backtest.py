"""
Backtest on Test Data - Uses model predictions
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestEngine, save_backtest_results
from backtest.plots import plot_equity_curve
from utils.io import save_json

logger = logging.getLogger(__name__)

def run_backtest_on_test(
    timeframe: str,
    test_data: pd.DataFrame,
    predictions: np.ndarray,
    config: Dict,
    run_id: str
) -> Dict:
    """
    Run backtest on test data using model predictions.
    
    Args:
        timeframe: Timeframe string
        test_data: Test DataFrame with OHLCV
        predictions: Model predictions (forward returns)
        config: Configuration dict
        run_id: Run ID
    
    Returns:
        Backtest results dict
    """
    # Convert predictions to signals
    # Simple strategy: if predicted return > threshold -> LONG, < -threshold -> SHORT
    backtest_config = config.get('backtest', {})
    threshold = backtest_config.get('signal_threshold', 0.01)  # 1% return threshold
    
    signals = np.zeros(len(predictions))
    signals[predictions > threshold] = 1  # LONG
    signals[predictions < -threshold] = -1  # SHORT
    
    # Ensure signals align with test data
    if len(signals) != len(test_data):
        min_len = min(len(signals), len(test_data))
        signals = signals[:min_len]
        test_data = test_data.iloc[:min_len].copy()
    
    # Run backtest
    engine = BacktestEngine(
        initial_balance=10000.0,
        fee_rate=backtest_config.get('fee_rate', 0.0004),
        slippage=backtest_config.get('slippage', 0.0005),
        position_sizing=backtest_config.get('position_sizing', 'fixed'),
        position_size=backtest_config.get('position_size', 0.1),
        enable_long=backtest_config.get('enable_long', True),
        enable_short=backtest_config.get('enable_short', True),
        max_leverage=backtest_config.get('max_leverage', 1.0),
    )
    
    results = engine.run_backtest(
        df=test_data,
        signals=pd.Series(signals, index=test_data.index),
        price_col='close',
        timestamp_col='timestamp'
    )
    
    # Save results
    artifacts_dir = Path(config.get('paths', {}).get('artifacts_dir', 'artifacts')) / run_id / timeframe
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_backtest_results(results, artifacts_dir, timeframe)
    
    # Plot
    plot_equity_curve(
        results['equity_curve'],
        artifacts_dir / 'equity_curve.png',
        title=f"{timeframe} Backtest Equity Curve"
    )
    
    logger.info(f"Backtest completed for {timeframe}")
    logger.info(f"  Total Return: {results['metrics']['total_return_pct']:.2f}%")
    logger.info(f"  Sharpe: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"  Max DD: {results['metrics']['max_drawdown_pct']:.2f}%")
    logger.info(f"  Win Rate: {results['metrics']['win_rate_pct']:.1f}%")
    logger.info(f"  Trades: {results['metrics']['total_trades']}")
    
    return results

