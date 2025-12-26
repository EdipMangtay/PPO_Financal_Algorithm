"""
SANITY CHECK: Futures Accounting Verification

Given:
- entry_price = 100
- current_price = 110
- position_type = LONG
- leverage = 2
- margin_used = 1000

Expected Calculation:
- position_notional = margin_used * leverage = 1000 * 2 = 2000
- units = position_notional / entry_price = 2000 / 100 = 20
- unrealized_pnl_usd = units * (current_price - entry_price) = 20 * (110 - 100) = 200
- equity = balance + unrealized_pnl_usd

This code verifies the calculation matches the implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.trading_env import TradingEnv, Position, PositionType
from config import INITIAL_BALANCE, BASE_LEVERAGE, TAKER_FEE
import pandas as pd
import numpy as np

# Test parameters
entry_price = 100.0
current_price = 110.0
margin_used = 1000.0
leverage = 2.0
initial_balance = 10000.0

# Expected calculations
position_notional = margin_used * leverage  # 2000
units = position_notional / entry_price  # 20
unrealized_pnl_usd = units * (current_price - entry_price)  # 20 * 10 = 200

print("=" * 60)
print("SANITY CHECK: Futures Accounting")
print("=" * 60)
print(f"Entry Price: {entry_price}")
print(f"Current Price: {current_price}")
print(f"Margin Used: {margin_used}")
print(f"Leverage: {leverage}x")
print()
print("Expected Calculations:")
print(f"  Position Notional = {margin_used} * {leverage} = {position_notional}")
print(f"  Units = {position_notional} / {entry_price} = {units}")
print(f"  Unrealized PnL = {units} * ({current_price} - {entry_price}) = {unrealized_pnl_usd}")
print()

# Create test environment
test_data = pd.DataFrame({
    'open': [entry_price] * 10,
    'high': [current_price] * 10,
    'low': [entry_price] * 10,
    'close': [current_price] * 10,
    'volume': [1000] * 10
})

env = TradingEnv(data={'TEST/USDT': test_data}, initial_balance=initial_balance)
obs, info = env.reset()

# Create position manually
position = Position(
    coin='TEST/USDT',
    position_type=PositionType.LONG,
    entry_price=entry_price,
    size=position_notional,  # Notional value
    leverage=leverage,
    margin_used=margin_used,
    entry_time=pd.Timestamp.now(),
    entry_step=0,
    stop_loss=90.0,
    take_profit=120.0,
    bars_held=0
)

env.positions['TEST/USDT'] = position
env.balance = initial_balance - margin_used  # Deduct margin

# Calculate unrealized PnL using the method
calculated_pnl = env._calculate_unrealized_pnl(position, current_price)
equity = env.portfolio_value

print("Implementation Results:")
print(f"  Calculated Unrealized PnL: {calculated_pnl:.2f}")
print(f"  Expected Unrealized PnL: {unrealized_pnl_usd:.2f}")
match_pnl = "OK" if abs(calculated_pnl - unrealized_pnl_usd) < 0.01 else "FAIL"
print(f"  Match: {match_pnl}")
print()
print(f"  Balance: {env.balance:.2f}")
print(f"  Equity (Balance + Unrealized PnL): {equity:.2f}")
print(f"  Expected Equity: {initial_balance - margin_used + unrealized_pnl_usd:.2f}")
match_equity = "OK" if abs(equity - (initial_balance - margin_used + unrealized_pnl_usd)) < 0.01 else "FAIL"
print(f"  Match: {match_equity}")
print()

# Verify
if abs(calculated_pnl - unrealized_pnl_usd) < 0.01:
    print("OK SANITY CHECK PASSED: Futures accounting is correct!")
else:
    print("FAIL SANITY CHECK FAILED: Futures accounting mismatch!")
    sys.exit(1)

print("=" * 60)

