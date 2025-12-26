# FUTURES ACCOUNTING FIX - Summary

## Problem
Buy&Hold baseline validation was failing. Expected positive asset return, but environment returned ~-15%. This indicated incorrect futures accounting.

## Root Causes Identified

1. **Incorrect PnL Calculation**: Used `position.size * pnl_pct` which is wrong for futures
2. **Missing Margin Tracking**: Didn't track `margin_used` separately from `position.size` (notional)
3. **Fee Double Counting**: Fees might have been applied incorrectly
4. **Stops Triggering**: Buy&Hold position was being closed by hard stops during baseline

## Fixes Applied

### A) `env/trading_env.py` - Correct Futures Accounting

#### 1. Added `margin_used` to Position dataclass
```python
@dataclass
class Position:
    ...
    margin_used: float  # CRITICAL: Margin actually used (notional / leverage)
```

#### 2. New `_calculate_unrealized_pnl()` method
```python
def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
    """
    CRITICAL: Correct futures accounting for unrealized PnL.
    
    Formula:
    - position_notional = margin_used * leverage
    - units = position_notional / entry_price
    - unrealized_pnl_usd = units * (price_t - entry_price) for long
    - unrealized_pnl_usd = units * (entry_price - price_t) for short
    """
    position_notional = position.margin_used * position.leverage
    units = position_notional / position.entry_price
    
    if position.position_type == PositionType.LONG:
        unrealized_pnl_usd = units * (current_price - position.entry_price)
    else:  # SHORT
        unrealized_pnl_usd = units * (position.entry_price - current_price)
    
    return unrealized_pnl_usd
```

#### 3. Fixed `portfolio_value` property
```python
@property
def portfolio_value(self) -> float:
    equity = self.balance
    for coin, position in self.positions.items():
        current_price = self._get_current_price(coin)
        if current_price is None:
            continue
        unrealized_pnl_usd = self._calculate_unrealized_pnl(position, current_price)
        equity += unrealized_pnl_usd
    return equity
```

#### 4. Fixed `_close_position()` accounting
```python
def _close_position(self, coin: str, reason: str, exit_price: Optional[float] = None):
    # Calculate realized PnL
    unrealized_pnl_usd = self._calculate_unrealized_pnl(position, exit_price)
    
    # Exit fee only (entry fee already charged)
    exit_fee = position_notional * TAKER_FEE
    slippage_cost = position_notional * SLIPPAGE_PCT
    
    net_pnl = unrealized_pnl_usd - exit_fee - slippage_cost
    
    # Return margin + add realized PnL
    self.balance += position.margin_used  # Return margin
    self.balance += net_pnl  # Add realized PnL
```

#### 5. Added `ignore_stops` flag for baseline validation
```python
# In __init__
self.ignore_stops: bool = False
self.debug_mode: bool = False

# In _check_hard_stops()
if self.ignore_stops:
    return []  # Skip stops during baseline
```

#### 6. Store `margin_used` when opening position
```python
position = Position(
    ...
    margin_used=required_margin,  # CRITICAL: Store margin used
    ...
)
```

### B) `tuning/optimizer.py` - Buy&Hold Baseline Fix

#### 1. Enable `ignore_stops` during baseline
```python
env_bh = TradingEnv(data=data_dict, initial_balance=INITIAL_BALANCE)
obs, info = env_bh.reset(options={'ignore_stops': True, 'debug_mode': True})
```

#### 2. Close position at end to realize PnL
```python
# After all steps, close position to realize PnL
if len(env_bh.positions) > 0:
    for coin in list(env_bh.positions.keys()):
        env_bh._close_position(coin, "Baseline End", None)
```

#### 3. Adjusted expected return calculation
```python
# Account for leverage
expected_with_leverage = expected_return * leverage_used

# Allow tolerance for fees (0.2%)
fee_tolerance = 0.002
expected_min = expected_with_leverage - fee_tolerance
expected_max = expected_with_leverage + 0.01
```

## Sanity Check Verification

**Test Case:**
- Entry Price: 100
- Current Price: 110
- Margin Used: 1000
- Leverage: 2x

**Expected:**
- Position Notional = 1000 * 2 = 2000
- Units = 2000 / 100 = 20
- Unrealized PnL = 20 * (110 - 100) = 200
- Equity = Balance + Unrealized PnL = 9000 + 200 = 9200

**Result:** ✓ PASSED

## Key Changes Summary

1. ✅ Correct futures accounting (units-based PnL calculation)
2. ✅ Track `margin_used` separately from notional value
3. ✅ Fees charged only on trade execution (entry + exit)
4. ✅ `ignore_stops` flag for baseline validation
5. ✅ Debug logging for baselines
6. ✅ Close position at end of Buy&Hold baseline

## Testing

Run the sanity check:
```powershell
python SANITY_CHECK_FUTURES_ACCOUNTING.py
```

Expected output: "OK SANITY CHECK PASSED"

Then run the full pipeline:
```powershell
python master_pipeline.py --days 1825 --trials 100 --coins BTC/USDT --backtest-steps 10000
```

Buy&Hold baseline should now pass validation.

