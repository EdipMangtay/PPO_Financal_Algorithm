# ROOT CAUSE ANALYSIS: Absurd Trading Metrics

## Top 10 Root Causes (Ranked by Likelihood)

### 1. **Sortino Ratio Calculation Bug** (CRITICAL)
**Location:** `tuning/optimizer.py:178`
**Issue:** Returns `10.0` when no downside returns exist
**Impact:** Inflates scores massively when strategy has only positive returns
**Fix:** Cap at reasonable maximum (e.g., 5.0) or use proper formula

### 2. **Wrong Annualization Factor** (CRITICAL)
**Location:** `tuning/optimizer.py:185, 363`
**Issue:** Using `sqrt(252)` for 15m timeframe (should be ~sqrt(35040) for 15m candles/year)
**Impact:** Sharpe/Sortino are 10x+ too high
**Fix:** Calculate correct periods per year based on timeframe

### 3. **No Slippage/Spread Costs** (HIGH)
**Location:** `env/trading_env.py:457`
**Issue:** Only fees applied, no slippage or bid-ask spread
**Impact:** Unrealistic execution costs, inflates returns
**Fix:** Add slippage (0.05-0.1% per trade) and spread costs

### 4. **Random Policy in Backtest** (HIGH)
**Location:** `tuning/optimizer.py:292`
**Issue:** Using random actions with fixed high confidence (0.7)
**Impact:** Doesn't reflect real trading, generates unrealistic trade patterns
**Fix:** Use proper policy or at least realistic action distribution

### 5. **No Train/Validation Split** (HIGH)
**Location:** `tuning/optimizer.py:200-426`
**Issue:** All data used for optimization, no OOS evaluation
**Impact:** Overfitting, unrealistic in-sample performance
**Fix:** Implement walk-forward or time-based split

### 6. **Returns Compounding Error** (MEDIUM)
**Location:** `tuning/optimizer.py:314`
**Issue:** Step-by-step returns may compound incorrectly
**Impact:** Total return calculation may be wrong
**Fix:** Use proper portfolio value series, not step returns

### 7. **Feature Shift Inconsistency** (MEDIUM)
**Location:** `tuning/optimizer.py:250-257` vs `data_engine/features.py:477`
**Issue:** Features shifted in optimizer but maybe double-shifted or inconsistent
**Impact:** Look-ahead bias or missing data
**Fix:** Single canonical shift location

### 8. **No Position Sizing Limits** (MEDIUM)
**Location:** `env/trading_env.py:310-311`
**Issue:** Kelly can create very large positions relative to portfolio
**Impact:** Unrealistic leverage and risk
**Fix:** Cap position size at reasonable % of portfolio

### 9. **Reward Calculation Issues** (MEDIUM)
**Location:** `env/trading_env.py:671`
**Issue:** Using cumulative portfolio return as reward
**Impact:** Reward doesn't reflect per-trade performance
**Fix:** Use per-step return or risk-adjusted metric

### 10. **No Reproducibility Seeds** (LOW)
**Location:** Multiple files
**Issue:** Random actions, no deterministic seeds
**Impact:** Results not reproducible
**Fix:** Set seeds for numpy, random, environment

