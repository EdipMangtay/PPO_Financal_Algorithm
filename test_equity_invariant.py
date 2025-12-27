"""
Unit test for equity invariant: equity_curve[step] should ALWAYS match recomputed equity from state.
Tests 5k steps with random actions to verify invariant holds.
"""

import numpy as np
import pandas as pd
from env.trading_env import TradingEnv
from config import INITIAL_BALANCE
import logging

logging.basicConfig(level=logging.WARNING)  # Suppress info logs during test

def create_test_data(coin: str, n_steps: int = 5000) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(42)
    base_price = 30000.0
    
    dates = pd.date_range(start='2024-01-01', periods=n_steps, freq='15min')
    
    # Random walk for prices
    returns = np.random.normal(0, 0.01, n_steps)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    high_mult = 1 + np.abs(np.random.normal(0, 0.005, n_steps))
    low_mult = 1 - np.abs(np.random.normal(0, 0.005, n_steps))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_steps)),
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_steps),
        'ATR_14': np.random.uniform(100, 500, n_steps)
    }, index=dates)
    
    return df

def test_equity_invariant():
    """Test that equity_curve[step] always matches recomputed equity."""
    print("Creating test environment...")
    data = {'BTC/USDT': create_test_data('BTC/USDT', n_steps=5000)}
    env = TradingEnv(data=data, initial_balance=INITIAL_BALANCE)
    
    # Enable sanity checks
    env.enable_sanity_checks = True
    
    obs, info = env.reset()
    
    print(f"Starting test with {len(data['BTC/USDT'])} steps...")
    print(f"Initial balance: {INITIAL_BALANCE:.2f}")
    print(f"Initial equity: {env._recompute_equity():.2f}")
    
    mismatches = []
    tolerance = 1e-6
    
    for step in range(min(5000, len(data['BTC/USDT']))):
        # Random action
        action = np.random.uniform(-1.0, 1.0, size=(1,))
        confidence = np.random.uniform(0.5, 1.0)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(
            action,
            tft_confidence_15m=confidence,
            tft_confidence_1h=confidence,
            tft_confidence_4h=confidence,
            atr=None
        )
        
        # Check equity invariant: equity_curve[-1] should match recomputed equity
        if len(env.equity_curve) > 0:
            stored_equity = env.equity_curve[-1]
            recomputed_equity = env._recompute_equity()
            mismatch = abs(stored_equity - recomputed_equity)
            
            if mismatch > tolerance:
                mismatches.append({
                    'step': step,
                    'stored': stored_equity,
                    'recomputed': recomputed_equity,
                    'mismatch': mismatch,
                    'balance': env.balance,
                    'unrealized_pnl': env._mark_to_market(),
                    'positions': len(env.positions),
                    'cumulative_fees': env.cumulative_fees
                })
                print(f"\n❌ MISMATCH at step {step}:")
                print(f"   Stored equity: {stored_equity:.9f}")
                print(f"   Recomputed equity: {recomputed_equity:.9f}")
                print(f"   Mismatch: {mismatch:.9f}")
                print(f"   Balance: {env.balance:.9f}")
                print(f"   Unrealized PnL: {env._mark_to_market():.9f}")
                print(f"   Positions: {len(env.positions)}")
                print(f"   Cumulative fees: {env.cumulative_fees:.9f}")
                break  # Stop on first mismatch
        
        if step % 500 == 0:
            print(f"Step {step}: equity={env._recompute_equity():.2f}, balance={env.balance:.2f}, "
                  f"positions={len(env.positions)}, trades={len(env.trades)}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break
    
    # Final check
    if len(env.equity_curve) > 0:
        final_stored = env.equity_curve[-1]
        final_recomputed = env._recompute_equity()
        final_mismatch = abs(final_stored - final_recomputed)
        
        print(f"\n{'='*60}")
        print(f"FINAL CHECK:")
        print(f"  Stored equity: {final_stored:.9f}")
        print(f"  Recomputed equity: {final_recomputed:.9f}")
        print(f"  Mismatch: {final_mismatch:.9f}")
        print(f"  Balance: {env.balance:.9f}")
        print(f"  Unrealized PnL: {env._mark_to_market():.9f}")
        print(f"  Positions: {len(env.positions)}")
        print(f"  Total trades: {len(env.trades)}")
        print(f"  Cumulative fees: {env.cumulative_fees:.9f}")
        print(f"{'='*60}")
        
        if final_mismatch > tolerance:
            mismatches.append({
                'step': 'final',
                'stored': final_stored,
                'recomputed': final_recomputed,
                'mismatch': final_mismatch
            })
    
    # Results
    if len(mismatches) == 0:
        print(f"\n✅ SUCCESS: Equity invariant held for all {len(env.equity_curve)} steps!")
        print(f"   Final equity: {env._recompute_equity():.2f}")
        print(f"   Total trades: {len(env.trades)}")
        return True
    else:
        print(f"\n❌ FAILED: Found {len(mismatches)} equity mismatches")
        for m in mismatches:
            print(f"   Step {m['step']}: mismatch={m['mismatch']:.9f}")
        return False

if __name__ == '__main__':
    success = test_equity_invariant()
    exit(0 if success else 1)



