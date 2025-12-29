"""
Regression tests for equity accounting invariant.
Tests that equity_curve[step] ALWAYS equals recomputed equity from state.
"""

import pytest
import numpy as np
import pandas as pd
from env.trading_env import TradingEnv
from config import INITIAL_BALANCE, TAKER_FEE, SLIPPAGE_PCT
from env.trading_env import PositionType

# Disable logging during tests
import logging
logging.getLogger().setLevel(logging.ERROR)


def create_deterministic_data(coin: str, prices: list, n_steps: int = None) -> pd.DataFrame:
    """Create deterministic OHLCV data from price list."""
    if n_steps is None:
        n_steps = len(prices)
    
    dates = pd.date_range(start='2024-01-01', periods=n_steps, freq='15min')
    
    # Generate OHLC from close prices (deterministic)
    df_data = []
    for i, close_price in enumerate(prices[:n_steps]):
        high = close_price * 1.001  # Small spread
        low = close_price * 0.999
        open_price = close_price * 1.0001
        df_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': 1000.0,
            'ATR_14': close_price * 0.01  # 1% ATR
        })
    
    df = pd.DataFrame(df_data, index=dates[:len(df_data)])
    return df


class TestEquityAccounting:
    """Test suite for equity accounting invariant."""
    
    def test_empty_positions_equity_equals_balance(self):
        """Test 1: No positions. After reset and N steps with no trades, 
        assert equity == balance and equity_curve equals equity each step."""
        print("\n=== Test 1: Empty Positions ===")
        
        # Create simple price series
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]
        data = {'BTC/USDT': create_deterministic_data('BTC/USDT', prices)}
        env = TradingEnv(data=data, initial_balance=INITIAL_BALANCE)
        env.enable_sanity_checks = True
        
        obs, info = env.reset()
        
        # Run 5 steps with neutral actions (no trades)
        for step in range(5):
            action = np.array([0.0])  # Neutral action
            obs, reward, terminated, truncated, info = env.step(
                action,
                tft_confidence_15m=0.5,  # Below threshold, no trades
                tft_confidence_1h=0.5,
                tft_confidence_4h=0.5,
                atr=None
            )
            
            # Assert: equity == balance when no positions
            assert len(env.positions) == 0, f"Step {step}: Should have no positions"
            equity = env._compute_equity_state()
            assert abs(equity - env.balance) < 1e-6, \
                f"Step {step}: equity={equity:.9f} != balance={env.balance:.9f}"
            
            # Assert: equity_curve[-1] == equity
            if len(env.equity_curve) > 0:
                stored_equity = env.equity_curve[-1]
                assert abs(stored_equity - equity) < 1e-6, \
                    f"Step {step}: equity_curve[-1]={stored_equity:.9f} != equity={equity:.9f}"
            
            if terminated or truncated:
                break
        
        print(f"✅ Test 1 passed: {len(env.equity_curve)} steps, equity always equals balance")
    
    def test_open_position_equity_includes_unrealized_pnl(self):
        """Test 2: Open position then hold a few steps with deterministic prices, 
        assert equity = balance + unrealized_pnl each step."""
        print("\n=== Test 2: Open Position ===")
        
        # Create price series with known movement
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
        data = {'BTC/USDT': create_deterministic_data('BTC/USDT', prices)}
        env = TradingEnv(data=data, initial_balance=INITIAL_BALANCE)
        env.enable_sanity_checks = True
        
        obs, info = env.reset()
        
        # Step 0: Open LONG position
        action = np.array([1.0])  # Strong long signal
        obs, reward, terminated, truncated, info = env.step(
            action,
            tft_confidence_15m=0.8,  # Above threshold
            tft_confidence_1h=0.8,
            tft_confidence_4h=0.8,
            atr=None
        )
        
        # Verify position opened
        assert len(env.positions) == 1, "Position should be open"
        position = list(env.positions.values())[0]
        assert position.position_type == PositionType.LONG
        
        # Calculate expected values
        entry_price_with_slippage = prices[0] * (1 + SLIPPAGE_PCT)
        position_notional = position.margin_used * position.leverage
        entry_fee = position_notional * TAKER_FEE
        expected_balance_after_open = INITIAL_BALANCE - entry_fee
        
        assert abs(env.balance - expected_balance_after_open) < 1e-6, \
            f"Balance after open: {env.balance:.9f} != expected: {expected_balance_after_open:.9f}"
        
        # Hold position for a few steps
        for step in range(1, 4):
            action = np.array([1.0])  # Hold long
            obs, reward, terminated, truncated, info = env.step(
                action,
                tft_confidence_15m=0.8,
                tft_confidence_1h=0.8,
                tft_confidence_4h=0.8,
                atr=None
            )
            
            # Calculate expected unrealized PnL
            current_price = prices[step]
            position_notional = position.margin_used * position.leverage
            units = position_notional / entry_price_with_slippage
            expected_unrealized = units * (current_price - entry_price_with_slippage)
            
            # Assert: equity = balance + unrealized_pnl
            equity = env._compute_equity_state()
            expected_equity = env.balance + expected_unrealized
            
            assert abs(equity - expected_equity) < 1e-3, \
                f"Step {step}: equity={equity:.6f} != balance+unrealized={expected_equity:.6f}, " \
                f"balance={env.balance:.6f}, unrealized={expected_unrealized:.6f}"
            
            # Assert: equity_curve[-1] == equity
            if len(env.equity_curve) > 0:
                stored_equity = env.equity_curve[-1]
                assert abs(stored_equity - equity) < 1e-6, \
                    f"Step {step}: equity_curve[-1]={stored_equity:.9f} != equity={equity:.9f}"
            
            if terminated or truncated:
                break
        
        print(f"✅ Test 2 passed: {len(env.equity_curve)} steps, equity = balance + unrealized_pnl")
    
    def test_open_close_position_fees_applied_once(self):
        """Test 3: Open and close position with known price change, 
        assert realized pnl and fees applied exactly once."""
        print("\n=== Test 3: Open and Close ===")
        
        # Create price series: open at 100, close at 105 (5% gain)
        prices = [100.0, 105.0]
        data = {'BTC/USDT': create_deterministic_data('BTC/USDT', prices)}
        env = TradingEnv(data=data, initial_balance=INITIAL_BALANCE)
        env.enable_sanity_checks = True
        
        obs, info = env.reset()
        
        # Step 0: Open LONG position
        action = np.array([1.0])
        obs, reward, terminated, truncated, info = env.step(
            action,
            tft_confidence_15m=0.8,
            tft_confidence_1h=0.8,
            tft_confidence_4h=0.8,
            atr=None
        )
        
        assert len(env.positions) == 1
        position = list(env.positions.values())[0]
        
        # Calculate expected values after open
        entry_price_with_slippage = prices[0] * (1 + SLIPPAGE_PCT)
        position_notional = position.margin_used * position.leverage
        entry_fee = position_notional * TAKER_FEE
        expected_balance_after_open = INITIAL_BALANCE - entry_fee
        
        assert abs(env.balance - expected_balance_after_open) < 1e-6
        
        # Step 1: Close position (agent flat signal)
        action = np.array([0.0])  # Neutral = close
        obs, reward, terminated, truncated, info = env.step(
            action,
            tft_confidence_15m=0.8,
            tft_confidence_1h=0.8,
            tft_confidence_4h=0.8,
            atr=None
        )
        
        # Verify position closed
        assert len(env.positions) == 0, "Position should be closed"
        
        # Calculate expected values after close
        exit_price = prices[1]
        exit_price_with_slippage = exit_price * (1 - SLIPPAGE_PCT)  # Long exit
        units = position_notional / entry_price_with_slippage
        unrealized_pnl = units * (exit_price_with_slippage - entry_price_with_slippage)
        exit_fee = position_notional * TAKER_FEE
        net_pnl = unrealized_pnl - exit_fee
        expected_balance_after_close = expected_balance_after_open + net_pnl
        
        assert abs(env.balance - expected_balance_after_close) < 1e-3, \
            f"Balance after close: {env.balance:.6f} != expected: {expected_balance_after_close:.6f}, " \
            f"net_pnl={net_pnl:.6f}, entry_fee={entry_fee:.6f}, exit_fee={exit_fee:.6f}"
        
        # Assert: equity == balance (no positions)
        equity = env._compute_equity_state()
        assert abs(equity - env.balance) < 1e-6, \
            f"Equity={equity:.9f} != balance={env.balance:.9f}"
        
        # Assert: equity_curve[-1] == equity
        stored_equity = env.equity_curve[-1]
        assert abs(stored_equity - equity) < 1e-6, \
            f"equity_curve[-1]={stored_equity:.9f} != equity={equity:.9f}"
        
        # Verify fees were applied exactly once
        expected_total_fees = entry_fee + exit_fee
        assert abs(env.cumulative_fees - expected_total_fees) < 1e-6, \
            f"Cumulative fees: {env.cumulative_fees:.9f} != expected: {expected_total_fees:.9f}"
        
        print(f"✅ Test 3 passed: Fees applied once, equity = balance after close")
        print(f"   Balance: {INITIAL_BALANCE:.2f} -> {env.balance:.2f}")
        print(f"   Total fees: {env.cumulative_fees:.6f}")
    
    def test_equity_invariant_throughout_step(self):
        """Test 4: Verify equity invariant holds at every point in step()."""
        print("\n=== Test 4: Equity Invariant Throughout Step ===")
        
        prices = [100.0, 101.0, 102.0, 100.0]  # Price goes up then down
        data = {'BTC/USDT': create_deterministic_data('BTC/USDT', prices)}
        env = TradingEnv(data=data, initial_balance=INITIAL_BALANCE)
        env.enable_sanity_checks = True
        
        obs, info = env.reset()
        
        # Run multiple steps with various actions
        for step in range(3):
            # Alternate between long and neutral
            if step % 2 == 0:
                action = np.array([1.0])  # Long
                confidence = 0.8
            else:
                action = np.array([0.0])  # Neutral (close)
                confidence = 0.5
            
            obs, reward, terminated, truncated, info = env.step(
                action,
                tft_confidence_15m=confidence,
                tft_confidence_1h=confidence,
                tft_confidence_4h=confidence,
                atr=None
            )
            
            # After each step, verify invariant
            equity = env._compute_equity_state()
            if len(env.equity_curve) > 0:
                stored_equity = env.equity_curve[-1]
                assert abs(stored_equity - equity) < 1e-6, \
                    f"Step {step}: equity_curve[-1]={stored_equity:.9f} != equity={equity:.9f}"
            
            # When no positions, equity must equal balance
            if len(env.positions) == 0:
                assert abs(equity - env.balance) < 1e-6, \
                    f"Step {step}: No positions but equity={equity:.9f} != balance={env.balance:.9f}"
            
            if terminated or truncated:
                break
        
        print(f"✅ Test 4 passed: Equity invariant held for {len(env.equity_curve)} steps")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])






