# LEVERAGE DOUBLE COUNTING FIX

## Root Cause Identified

**Problem:** Leverage was being counted TWICE in expected return calculation.

**Old (Wrong) Code:**
```python
expected_with_leverage = expected_return * leverage_used
```

**Why This is Wrong:**
- `position_notional = margin_used * leverage` (leverage already included)
- Buy&Hold opens with `position_notional = 975` on `initial_balance = 10000`
- This means only 9.75% of capital is used
- Multiplying by leverage again = DOUBLE COUNTING

**Example:**
- Asset return: 100%
- Position notional: 975 (9.75% of 10000)
- Old calculation: `100% * 3x leverage = 300%` ❌ WRONG
- Correct calculation: `100% * (975/10000) = 9.75%` ✓ CORRECT

---

## Fix Applied

### `tuning/optimizer.py` - Buy&Hold Baseline Validation

**New (Correct) Code:**
```python
# Get actual position_notional used
position_notional = env_bh.trades[0].size  # or from position.margin_used * leverage

# Correct expected return (NO leverage multiplication)
position_fraction = position_notional / INITIAL_BALANCE
expected_portfolio_return = expected_return * position_fraction

# Fees calculation
fees_total = 2 * TAKER_FEE * position_notional  # open + close
fee_impact = fees_total / INITIAL_BALANCE
expected_after_fees = expected_portfolio_return - fee_impact
```

**Key Changes:**
1. ✅ Get actual `position_notional` from trade/position
2. ✅ Calculate `position_fraction = position_notional / initial_balance`
3. ✅ `expected_portfolio_return = asset_return * position_fraction` (NO leverage multiply)
4. ✅ Fees based on actual notional: `fees_total = 2 * TAKER_FEE * position_notional`
5. ✅ `fee_impact = fees_total / initial_balance`
6. ✅ Updated debug output with all required fields

---

## Updated Debug Output

**New Format:**
```
Buy&Hold: Return=X.XX%, 
Expected Asset Return (unlevered)=X.XX%, 
Position Fraction=X.XX%, 
Expected Portfolio Return=X.XX%, 
Expected After Fees=X.XX%
```

**Fields Explained:**
- **Expected Asset Return (unlevered):** Raw asset price movement (e.g., 100% if price doubles)
- **Position Fraction:** What % of capital is actually used (e.g., 9.75% if notional=975, balance=10000)
- **Expected Portfolio Return:** Asset return scaled by position fraction (e.g., 100% * 9.75% = 9.75%)
- **Expected After Fees:** Portfolio return minus fee impact

---

## Verification

After this fix:
1. Buy&Hold should match `expected_after_fees` within 0.5% tolerance
2. Debug logs show correct position fraction
3. No more leverage double counting

---

## Example Calculation

**Given:**
- Initial balance: 10000
- Position notional: 975 (9.75% of capital)
- Asset return: 100% (price doubles)
- TAKER_FEE: 0.0004

**Calculation:**
- Position fraction: 975 / 10000 = 0.0975 (9.75%)
- Expected portfolio return: 100% * 0.0975 = 9.75%
- Fees total: 2 * 0.0004 * 975 = 0.78
- Fee impact: 0.78 / 10000 = 0.000078 (0.0078%)
- Expected after fees: 9.75% - 0.0078% = 9.7422%

**Buy&Hold return should be ~9.74%** (not 300%!)

