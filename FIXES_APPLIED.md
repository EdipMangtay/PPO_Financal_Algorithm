# CRITICAL FIXES APPLIED

## Summary of Changes

### 1. Sortino Ratio Calculation (FIXED)
- **File:** `tuning/optimizer.py`
- **Issue:** Returned 10.0 when no downside returns, inflating scores
- **Fix:** Capped at 5.0 for no-downside case, max 10.0 overall
- **Impact:** Prevents absurd Sortino ratios (1000+ → realistic <10)

### 2. Annualization Factor (FIXED)
- **File:** `tuning/optimizer.py`
- **Issue:** Using sqrt(252) for 15m timeframe (should be ~sqrt(35040))
- **Fix:** Added `_periods_per_year()` method with correct factors per timeframe
- **Impact:** Sharpe/Sortino now correctly annualized (24 → ~2-3)

### 3. Slippage & Spread (PARTIALLY FIXED)
- **File:** `env/trading_env.py`, `config.py`
- **Issue:** No slippage or spread costs
- **Fix:** Added SLIPPAGE_PCT (0.05%) and SPREAD_PCT (0.02%) to config
- **Status:** Needs refinement in execution price calculation
- **Impact:** More realistic execution costs

### 4. Feature Shift Consistency (TO BE FIXED)
- **Issue:** Features may be shifted twice or inconsistently
- **Action Required:** Ensure single canonical shift location

### 5. Walk-Forward Split (TO BE IMPLEMENTED)
- **Issue:** No train/validation split, all data used for optimization
- **Action Required:** Implement time-based split in optimizer

### 6. Reproducibility (TO BE IMPLEMENTED)
- **Issue:** No deterministic seeds
- **Action Required:** Add seeds for numpy, random, environment

### 7. Data Alignment Assertions (TO BE IMPLEMENTED)
- **Issue:** No validation of data alignment
- **Action Required:** Add assertions for monotonic timestamps, no NaNs, etc.

## Next Steps

1. Complete slippage implementation (fix double-counting)
2. Implement walk-forward train/val split
3. Add reproducibility seeds
4. Add data alignment assertions
5. Fix feature shift consistency
6. Create acceptance test suite

