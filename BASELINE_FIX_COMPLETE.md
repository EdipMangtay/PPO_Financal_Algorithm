# BASELINE FIX - Complete Solution

## 1. TOP 8 FAILURE MODES

1. **Per-Step Fee Drain**: Fees charged every step instead of only on open/close
2. **Funding Rate Misapplied**: Funding fees charged when they shouldn't be (default should be 0)
3. **Wrong Notional Calculation**: Using `position.size` instead of `margin_used * leverage`
4. **Wrong Units Calculation**: Dividing by wrong price or using wrong notional source
5. **Wrong Price Used**: Using high/low instead of close, or future price (lookahead)
6. **Double Counting**: Adding both `pnl_pct` and `pnl_usd`, or counting fees twice
7. **Entry/Exit Fee Mismatch**: Entry fee charged but not properly accounted, or exit fee charged twice
8. **Forced Close by Stop**: Hard stop, time stop, or risk engine closing position early (even with ignore_stops)

---

## 2. DEBUG CHECKLIST LOG FORMAT

The debug log will print at step 0, 1, 5, and last step with these exact fields:

```
BASELINE_DEBUG step=X | timestamp=... | close_price=X.XXXX | entry_price=X.XXXX | 
side=LONG | margin_used=XXXX.XX | leverage=X.Xx | position_notional=XXXX.XX | 
units=X.XXXX | unrealized_pnl_usd=XXX.XX | realized_pnl_usd=XXX.XX | 
fee_open=X.XXXX | fee_close=X.XXXX | cumulative_fees=X.XXXX | 
balance=XXXX.XX | equity=XXXX.XX | is_position_open=True | stop_triggered_reason=None
```

---

## 3. PATCH FILES CREATED

### A) `PATCH_config.py`
- Update TAKER_FEE to 0.0004 (Binance USDT-M default)
- Add MAKER_FEE = 0.0002
- Add FUNDING_RATE = 0.0 (default no funding)

### B) `PATCH_env_trading_env.py`
- Add `funding_rate` and `cumulative_fees` to `__init__`
- Add `_log_baseline_debug()` method
- Update `_close_position()` to ensure no funding charges
- Add debug logging calls in `step()`
- Initialize `cumulative_fees` in `reset()`

### C) `PATCH_tuning_optimizer.py`
- Replace Buy&Hold baseline with debug logging
- Add calls to `_log_baseline_debug()` at steps 0, 1, 5, last
- Fix expected return calculation (account for fees properly)
- Close position at end to realize PnL

---

## 4. SELF-TEST FUNCTION

**File:** `test_futures_accounting_standalone.py`

**Run:** `python test_futures_accounting_standalone.py`

**Expected Output:**
```
OK Self-test PASSED:
  Notional: 2000.00 (expected: 2000.00)
  Units: 20.0000 (expected: 20.0000)
  Unrealized PnL: 200.00 (expected: 200.00)
  Equity: 9200.00 (expected: 9200.00)
```

---

## 5. APPLICATION INSTRUCTIONS

### Step 1: Apply config.py patch
```python
# In config.py, replace lines 40-44 with content from PATCH_config.py
```

### Step 2: Apply env/trading_env.py patch
```python
# Follow instructions in PATCH_env_trading_env.py
# 5 separate changes to apply
```

### Step 3: Apply tuning/optimizer.py patch
```python
# Replace Buy&Hold baseline section (lines 167-230) with content from PATCH_tuning_optimizer.py
```

### Step 4: Run self-test
```powershell
python test_futures_accounting_standalone.py
```

### Step 5: Run pipeline
```powershell
python master_pipeline.py --days 1825 --trials 1 --coins BTC/USDT --backtest-steps 10000
```

---

## 6. VERIFICATION

After applying patches, check:

1. **Self-test passes**: `test_futures_accounting_standalone.py` outputs "OK Self-test PASSED"
2. **Debug logs appear**: Look for `BASELINE_DEBUG` lines at steps 0, 1, 5, last
3. **Buy&Hold passes**: Should match expected return within 0.5% tolerance
4. **No funding charges**: All `funding_cost = 0.0` in logs
5. **Fees correct**: `fee_open` and `fee_close` calculated on notional value

---

## 7. ASSUMPTIONS MADE

1. **Fee Rate**: Using Binance USDT-M default taker fee (0.04%) instead of 0.06%
2. **Funding**: Default to 0 (no funding charges unless explicitly enabled)
3. **Price Source**: Using `close` price for all calculations (no high/low)
4. **Baseline Close**: Closing position at end to realize PnL (consistent accounting)
5. **Equity Formula**: `equity = balance + unrealized_pnl_usd` (no double counting)

---

## 8. TROUBLESHOOTING

If Buy&Hold still fails after applying patches:

1. **Check debug logs**: Compare logged values with expected
2. **Verify fee calculation**: `fee = notional * TAKER_FEE` (notional = margin * leverage)
3. **Check equity formula**: `equity = balance + unrealized_pnl_usd`
4. **Verify no funding**: All funding costs should be 0.0
5. **Check position close**: Position should close exactly once at end

---

## NEXT STEPS

1. Apply all patches
2. Run self-test (should pass)
3. Run pipeline with `--trials 1` to test baseline
4. Check debug logs for exact values
5. If still failing, compare logged values with expected to identify exact bug

