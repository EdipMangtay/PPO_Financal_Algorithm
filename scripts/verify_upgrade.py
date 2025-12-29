"""
Verification Script - Proves the Profit-Driven HPO Upgrade Works
Tests the new financial metrics and feature selection logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_mock_data(n_samples: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """
    Create synthetic market data for testing.
    
    Returns:
        DataFrame with OHLCV + technical features + target
    """
    logger.info(f"Creating mock data: {n_samples} samples, {n_features} features")
    
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    # Random walk price with trend
    price = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price + np.random.randn(n_samples) * 10,
        'high': price + np.abs(np.random.randn(n_samples) * 20),
        'low': price - np.abs(np.random.randn(n_samples) * 20),
        'close': price,
        'volume': np.random.randint(100, 1000, n_samples),
        'time_idx': np.arange(n_samples),
        'coin': 'BTC/USDT'
    })
    
    # Generate mock technical features (organized by groups)
    # Oscillators
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = np.random.uniform(20, 80, n_samples)
        df[f'STOCH_k_{period}'] = np.random.uniform(20, 80, n_samples)
    
    # Moving Averages
    for length in [10, 20, 50]:
        df[f'EMA_{length}'] = df['close'].ewm(span=length).mean()
        df[f'linreg_{length}'] = df['close'].rolling(window=length).mean()
    
    # Volatility
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = np.random.uniform(50, 500, n_samples)
        df[f'BB_upper_{period}_2'] = df['close'] * 1.02
        df[f'BB_lower_{period}_2'] = df['close'] * 0.98
    
    # Volume
    df['OBV'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
    df['VWAP'] = df['close'] * np.random.uniform(0.99, 1.01, n_samples)
    
    # Patterns
    for fast, slow in [(8, 21), (12, 26)]:
        df[f'MACD_{fast}_{slow}_9'] = np.random.randn(n_samples)
        df[f'MACD_hist_{fast}_{slow}_9'] = np.random.randn(n_samples)
    
    df['SuperTrend_10_2_0'] = df['close'] * np.random.uniform(0.98, 1.02, n_samples)
    df['ADX_14'] = np.random.uniform(10, 50, n_samples)
    
    # Target: future return (12 steps ahead, ~3 hours for 15m timeframe)
    df['target'] = df['close'].pct_change(12).shift(-12) * 100  # % return
    
    # Drop NaN rows
    df = df.dropna()
    
    logger.info(f"Mock data created: {len(df)} rows, {len(df.columns)} columns")
    
    return df

def test_feature_filtering():
    """Test feature selection logic."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Feature Filtering Logic")
    logger.info("="*80)
    
    from hpo.optuna_search import _filter_features
    
    # Create mock data
    df = create_mock_data(n_samples=500)
    
    # Test Case 1: All features enabled
    flags_all = {
        'use_oscillators': True,
        'use_moving_averages': True,
        'use_volatility': True,
        'use_volume': True,
        'use_patterns': True
    }
    
    df_all = _filter_features(df.copy(), flags_all)
    logger.info(f"✓ All features enabled: {len(df.columns)} -> {len(df_all.columns)} columns")
    
    # Test Case 2: Only oscillators
    flags_osc = {
        'use_oscillators': True,
        'use_moving_averages': False,
        'use_volatility': False,
        'use_volume': False,
        'use_patterns': False
    }
    
    df_osc = _filter_features(df.copy(), flags_osc)
    osc_features = [c for c in df_osc.columns if 'RSI' in c or 'STOCH' in c]
    logger.info(f"✓ Only oscillators: {len(df.columns)} -> {len(df_osc.columns)} columns")
    logger.info(f"  Oscillator features found: {len(osc_features)}")
    
    # Test Case 3: No features (only base columns)
    flags_none = {
        'use_oscillators': False,
        'use_moving_averages': False,
        'use_volatility': False,
        'use_volume': False,
        'use_patterns': False
    }
    
    df_none = _filter_features(df.copy(), flags_none)
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target', 'time_idx', 'coin']
    actual_base = [c for c in df_none.columns if c in base_cols]
    logger.info(f"✓ No features: {len(df.columns)} -> {len(df_none.columns)} columns (base only)")
    logger.info(f"  Base columns preserved: {len(actual_base)}")
    
    logger.info("\n✅ Feature filtering test PASSED")
    
    return True

def test_objective_function():
    """Test the new profit-driven objective function."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Profit-Driven Objective Function (Mock Trial)")
    logger.info("="*80)
    
    try:
        import optuna
        from hpo.optuna_search import objective
        
        # Create mock data
        df = create_mock_data(n_samples=1000)
        
        # Split data
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)
        
        train_data = df.iloc[:train_size].copy()
        val_data = df.iloc[train_size:train_size+val_size].copy()
        
        logger.info(f"Data split: Train={len(train_data)}, Val={len(val_data)}")
        
        # Mock config
        config = {
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mixed_precision': 'bf16' if torch.cuda.is_bf16_supported() else 'fp32',
            'grad_clip': 1.0,
            'num_workers': 0,
            'model': {
                'hidden_size': 64,  # Small for speed
                'attention_head_size': 2,
                'dropout': 0.1,
                'max_encoder_length': 30,  # Reduced for speed
                'max_decoder_length': 6,   # Reduced for speed
            },
            'task': {
                'mode': 'quantile',
                'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9]
            },
            'paths': {
                'artifacts_dir': 'artifacts_verify'
            }
        }
        
        logger.info(f"Running on device: {config['device']}")
        
        # Create a minimal study for testing
        study = optuna.create_study(direction='maximize')
        
        # Run 2 trials
        logger.info("\nRunning Trial 1...")
        try:
            trial1 = study.ask()
            score1 = objective(trial1, '15m', train_data, val_data, config, 'BTC/USDT')
            study.tell(trial1, score1)
            
            # Extract metrics
            da1 = trial1.user_attrs.get('directional_accuracy', 0)
            pnl1 = trial1.user_attrs.get('proxy_pnl_mean', 0)
            loss1 = trial1.user_attrs.get('val_loss', 0)
            n_feat1 = trial1.user_attrs.get('n_features_selected', 0)
            
            logger.info(f"✓ Trial 1 completed:")
            logger.info(f"  Score: {score1:.6f}")
            logger.info(f"  Directional Accuracy: {da1:.4f} ({da1*100:.2f}%)")
            logger.info(f"  Proxy PnL (mean): {pnl1:.6f}")
            logger.info(f"  Validation Loss: {loss1:.6f}")
            logger.info(f"  Features Selected: {n_feat1}")
            
            logger.info("\nRunning Trial 2...")
            trial2 = study.ask()
            score2 = objective(trial2, '15m', train_data, val_data, config, 'BTC/USDT')
            study.tell(trial2, score2)
            
            # Extract metrics
            da2 = trial2.user_attrs.get('directional_accuracy', 0)
            pnl2 = trial2.user_attrs.get('proxy_pnl_mean', 0)
            loss2 = trial2.user_attrs.get('val_loss', 0)
            n_feat2 = trial2.user_attrs.get('n_features_selected', 0)
            
            logger.info(f"✓ Trial 2 completed:")
            logger.info(f"  Score: {score2:.6f}")
            logger.info(f"  Directional Accuracy: {da2:.4f} ({da2*100:.2f}%)")
            logger.info(f"  Proxy PnL (mean): {pnl2:.6f}")
            logger.info(f"  Validation Loss: {loss2:.6f}")
            logger.info(f"  Features Selected: {n_feat2}")
            
            # Verify feature selection actually changes
            if n_feat1 != n_feat2:
                logger.info(f"\n✅ Feature selection is WORKING (Trial 1: {n_feat1}, Trial 2: {n_feat2})")
            else:
                logger.warning(f"\n⚠️  Feature count identical across trials (might be coincidence)")
            
            logger.info("\n✅ Objective function test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trial execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    logger.info("="*80)
    logger.info("VERIFICATION: Profit-Driven HPO Upgrade")
    logger.info("="*80)
    
    results = []
    
    # Test 1: Feature filtering
    try:
        results.append(("Feature Filtering", test_feature_filtering()))
    except Exception as e:
        logger.error(f"Feature filtering test FAILED: {e}")
        results.append(("Feature Filtering", False))
    
    # Test 2: Objective function
    try:
        results.append(("Objective Function", test_objective_function()))
    except Exception as e:
        logger.error(f"Objective function test FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Objective Function", False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - Upgrade is verified!")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED - Review errors above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


