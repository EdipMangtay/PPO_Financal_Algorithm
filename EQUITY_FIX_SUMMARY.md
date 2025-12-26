# Equity Accounting Fix Summary

## Root Cause Analysis

The equity mismatch was caused by **inconsistent equity calculation** between:
1. When `equity_curve[step]` is stored (during `step()`)
2. When equity is recomputed from state (in optimizer check)

### Symptoms
- `equity_curve_value > recomputed_equity` (typically 5-25 USDT mismatch)
- Mismatch occurs even when no positions are open
- Optuna trials get pruned due to invariant failure

### Root Cause
The accounting model uses "Option A": balance = wallet equity (margin NOT removed).
- Equity = balance + unrealized_pnl
- When positions are closed, balance is updated with net_pnl
- However, there was no single source of truth for equity calculation
- Multiple code paths could compute equity slightly differently (floating point, timing)

## Solution

### 1. Single Source of Truth: `_recompute_equity()`

Created a dedicated function that ALWAYS computes equity the same way:

```python
def _recompute_equity(self, mark_price: Optional[float] = None) -> float:
    """
    CRITICAL: Single source of truth for equity calculation.
    Recomputes equity from current state: balance + unrealized_pnl.
    """
    unrealized_pnl = 0.0
    for coin, position in self.positions.items():
        if mark_price is not None:
            current_price = mark_price
        else:
            current_price = self._get_current_price(coin)
        
        if current_price is not None:
            unrealized_pnl += self._calculate_unrealized_pnl(position, current_price)
    
    equity = self.balance + unrealized_pnl
    return equity
```

### 2. Per-Step Equity Invariant Assertion

Added assertion in `step()` that catches mismatches immediately:

```python
# After appending to equity_curve
stored_equity = self.equity_curve[-1]
post_append_recomputed = self._recompute_equity()
if abs(stored_equity - post_append_recomputed) > 1e-6:
    # Dump debug snapshot and raise error
```

### 3. Fixed Duplicate Fee Calculation

Removed duplicate `exit_fee` calculation in `_close_position()`:
- Before: `exit_fee` calculated twice (lines 474 and 490)
- After: Calculated once, after slippage adjustment

### 4. Updated Optimizer

Optimizer now uses `_recompute_equity()` for consistency:
```python
if hasattr(env, '_recompute_equity'):
    recomputed_equity = env._recompute_equity()
```

## Accounting Contract

### Equity Formula
```
equity = cash_balance + unrealized_pnl

Where:
- cash_balance = realized balance AFTER all fees (entry + exit)
- unrealized_pnl = sum of unrealized PnL for all open positions
- unrealized_pnl = 0 when no positions are open
```

### Balance Updates
- **On Open**: `balance -= entry_fee` (margin NOT removed, stays in balance)
- **On Close**: `balance += net_pnl` where `net_pnl = unrealized_pnl - exit_fee - funding`
- **Margin**: Tracked separately via `_used_margin`, NOT deducted from balance

### Fees
- Entry fee: Charged on open, deducted from balance
- Exit fee: Charged on close, deducted from net_pnl
- Slippage: Applied to fill price (not separate cost)

## Testing

Created `test_equity_invariant.py`:
- Runs 5k steps with random actions
- Verifies `equity_curve[step] == recomputed_equity` at every step
- Tolerance: 1e-6 (floating point precision)

## Files Changed

1. **env/trading_env.py**:
   - Added `_recompute_equity()` method
   - Updated `_equity()` to delegate to `_recompute_equity()`
   - Added per-step equity invariant assertion
   - Fixed duplicate `exit_fee` calculation in `_close_position()`

2. **tuning/optimizer.py**:
   - Updated to use `_recompute_equity()` for consistency
   - Improved debug snapshot with more details

3. **test_equity_invariant.py** (new):
   - Unit test with 5k random steps
   - Verifies invariant holds throughout

## Expected Outcome

- Equity invariant holds for all steps: `equity_curve[step] == recomputed_equity` (within 1e-6)
- Optuna trials no longer pruned due to equity mismatch
- Debug snapshots provide detailed accounting breakdown when mismatches occur

