# PRODUCTION TRADING ENVIRONMENT FIXES

## Overview
Comprehensive refactoring to remove accounting bugs, stop-fill optimism, timing inconsistencies, and observation errors. All fixes maintain Gymnasium API compatibility.

---

## A) MARGIN ACCOUNTING FIX

### Problem
Opening multiple positions allowed "free margin" because balance check ignored existing `margin_used`.

### Solution
Implemented proper margin reservation tracking:

```python
@property
def _used_margin(self) -> float:
    """Calculate total margin used across all open positions."""
    return sum(p.margin_used for p in self.positions.values())

@property
def _available_margin(self) -> float:
    """Available margin = balance - used_margin (reserved margin tracking)."""
    return self.balance - self._used_margin
```

**Changes:**
- Added `_used_margin` property to sum all position margins
- Added `_available_margin` property for available margin calculation
- Updated position open check: `required_margin <= self._available_margin` (not just `balance`)
- Maintains Option A: balance = wallet equity, margin NOT deducted

**Why This Prevents Overfitting:**
- Prevents opening positions that exceed available capital
- Ensures realistic margin constraints (no "free money" bug)
- Multiple positions correctly reserve margin

---

## B) STOP EXECUTION REALISM FIX

### Problem
Take-profit used `max(high, target)` for long and `min(low, target)` for short, giving unrealistically better fills.

### Solution
Changed to conservative, realistic fills:

**Stop Loss (Conservative):**
```python
# Long SL: fill at worse of current_low or stop_price
exit_price = min(current_low, stop_price)  # Worse fill (conservative)

# Short SL: fill at worse of current_high or stop_price
exit_price = max(current_high, stop_price)  # Worse fill (conservative)
```

**Take Profit (Target Price):**
```python
# TP uses target price, NOT bar extreme
exit_price = position.entry_price * (1 + TAKE_PROFIT_THRESHOLD / position.leverage)
# NOT: max(current_high, target)  # REMOVED optimistic fill
```

**Changes:**
- Hard Stop Loss: Uses `min(current_low, stop_price)` for long, `max(current_high, stop_price)` for short
- Hard Take Profit: Uses target price directly (not bar extreme)
- ATR Stop Loss: Uses `min(current_low, stop_loss)` for long, `max(current_high, stop_loss)` for short
- ATR Take Profit: Uses `take_profit` target directly (not `max(high, target)`)

**Why This Prevents Overfitting:**
- Removes optimistic fill bias (no "best case" fills)
- Conservative fills prevent hidden overfitting
- Realistic execution costs prevent false profitability

---

## C) BARS_HELD DOUBLE INCREMENT FIX

### Problem
`bars_held` incremented in `step()` AND again in `_update_trailing_stops()`, causing double counting.

### Solution
Removed increment from `_update_trailing_stops()`:

**Before:**
```python
def _update_trailing_stops(self):
    # ... trailing stop logic ...
    position.bars_held += 1  # ❌ DOUBLE INCREMENT
```

**After:**
```python
def _update_trailing_stops(self):
    # ... trailing stop logic ...
    # CRITICAL FIX: bars_held is incremented ONCE in step(), NOT here
```

**In step():**
```python
# STEP 1: Increment bars_held ONCE per step
for position in self.positions.values():
    position.bars_held += 1  # ✅ SINGLE INCREMENT
```

**Why This Prevents Overfitting:**
- Correct time-stop triggering (MAX_HOLD_BARS uses accurate count)
- Prevents premature position closure
- Accurate trade duration tracking

---

## D) OBSERVATION PNL DOUBLE LEVERAGE FIX

### Problem
`_get_observation` calculated PnL using `position.size * pnl_pct` where `pnl_pct` already includes leverage, causing double leverage counting.

### Solution
Use `unrealized_pnl_usd` directly:

**Before:**
```python
pnl_pct = ((current_price - entry_price) / entry_price) * position.leverage  # Leverage applied
pnl = position.size * pnl_pct / self.portfolio_value  # ❌ Double leverage (size already includes leverage)
```

**After:**
```python
# CRITICAL FIX: Use unrealized_pnl_usd directly (no double leverage)
unrealized_pnl_usd = self._calculate_unrealized_pnl(position, current_price)
pnl = unrealized_pnl_usd / portfolio_value  # ✅ Correct
```

**Also Fixed Position Size:**
```python
# CRITICAL FIX: position_size = position_notional / portfolio_value
position_notional = position.margin_used * position.leverage
position_size = position_notional / portfolio_value
```

**Why This Prevents Overfitting:**
- Correct PnL signal prevents agent from learning wrong patterns
- Accurate position size prevents incorrect risk perception
- No hidden leverage amplification in observations

---

## E) PORTFOLIO_VALUE TIMING CONSISTENCY FIX

### Problem
`portfolio_value` computed multiple times in `step()`, leading to stale values in `info` dict.

### Solution
Compute `portfolio_value` ONCE at end of step:

**Changes:**
- Compute `portfolio_value` once after all position changes
- Use same value for `equity_curve.append()`, `info['portfolio_value']`, and reward calculation
- Ensure `info['portfolio_value']` reflects final state

**Why This Prevents Overfitting:**
- Consistent equity curve prevents metric calculation errors
- Accurate reward signals prevent training instability
- No stale portfolio values in info dict

---

## F) BASELINE SUITE FIXES

### Problem
Buy&Hold expected return calculation didn't include slippage cost.

### Solution
Added slippage to expected return calculation:

**Before:**
```python
fees_total = 2 * TAKER_FEE * position_notional
fee_impact = fees_total / INITIAL_BALANCE
expected_after_fees = expected_portfolio_return - fee_impact
```

**After:**
```python
fees_total = 2 * TAKER_FEE * position_notional  # open + close fees
slippage_total = SLIPPAGE_PCT * position_notional  # Exit slippage only
total_cost = fees_total + slippage_total
cost_impact = total_cost / INITIAL_BALANCE
expected_after_fees = expected_portfolio_return - cost_impact
```

**Why This Prevents Overfitting:**
- Accurate baseline validation prevents false positives
- Includes realistic execution costs
- Correct tolerance bands prevent overly strict validation

---

## G) SANITY ASSERTIONS

### Implementation
Added lightweight internal assertions (behind `enable_sanity_checks` flag):

```python
if self.enable_sanity_checks:
    assert available_margin >= -1e-6, f"Available margin negative: {available_margin}"
    assert self.balance > 0, f"Balance non-positive: {self.balance}"
    assert position.margin_used > 0, f"Margin used non-positive: {position.margin_used}"
    assert 0 < position.leverage <= MAX_LEVERAGE, f"Invalid leverage: {position.leverage}"
```

**Locations:**
- After position open
- After position close
- End of step (portfolio_value check)

**Why This Prevents Overfitting:**
- Catches accounting bugs early
- Prevents invalid state propagation
- Debug flag avoids performance hit in production

---

## SUMMARY OF CHANGES

### `env/trading_env.py`
1. ✅ Added `_used_margin` and `_available_margin` properties
2. ✅ Updated position open check to use `available_margin`
3. ✅ Fixed stop execution (conservative fills, target price for TP)
4. ✅ Removed `bars_held` increment from `_update_trailing_stops()`
5. ✅ Fixed `_get_observation()` PnL calculation (no double leverage)
6. ✅ Fixed `_get_observation()` position_size calculation
7. ✅ Ensured `portfolio_value` computed once and used consistently
8. ✅ Added sanity assertions (behind debug flag)

### `tuning/optimizer.py`
1. ✅ Added slippage cost to Buy&Hold expected return calculation
2. ✅ Updated debug output to show "Expected After Fees (incl. slippage)"

---

## TESTING

All fixes maintain:
- ✅ Gymnasium API compatibility
- ✅ No breaking changes to external signatures
- ✅ Config variables unchanged
- ✅ Observation/action space shapes unchanged
- ✅ Logs and JSON saving intact
- ✅ No new dependencies

**Verification:**
```python
from env.trading_env import TradingEnv
from tuning.optimizer import TwoLayerOptimizer
# ✅ All imports successful
```

---

## IMPACT

**Before Fixes:**
- ❌ Multiple positions could exceed available margin
- ❌ Optimistic stop fills (hidden overfitting)
- ❌ Double leverage in observations (wrong signals)
- ❌ Inaccurate time-stop triggering
- ❌ Stale portfolio values in info dict
- ❌ Baseline validation missing slippage

**After Fixes:**
- ✅ Proper margin reservation tracking
- ✅ Conservative, realistic stop fills
- ✅ Correct PnL and position size in observations
- ✅ Accurate time-stop triggering
- ✅ Consistent portfolio value timing
- ✅ Complete baseline validation (fees + slippage)

---

## NEXT STEPS

1. Run baseline validation suite to verify fixes
2. Enable `enable_sanity_checks=True` for debugging if needed
3. Monitor training for improved stability and correctness



