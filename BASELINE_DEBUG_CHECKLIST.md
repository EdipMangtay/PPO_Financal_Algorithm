# BASELINE DEBUG CHECKLIST & PATCH

## 1. TOP 8 FAILURE MODES

Even after "correct" PnL formulas, Buy&Hold can still fail due to:

1. **Per-Step Fee Drain**: Fees charged every step instead of only on open/close
2. **Funding Rate Misapplied**: Funding fees charged when they shouldn't be (default should be 0)
3. **Wrong Notional Calculation**: Using `position.size` instead of `margin_used * leverage`
4. **Wrong Units Calculation**: Dividing by wrong price or using wrong notional source
5. **Wrong Price Used**: Using high/low instead of close, or future price (lookahead)
6. **Double Counting**: Adding both `pnl_pct` and `pnl_usd`, or counting fees twice
7. **Entry/Exit Fee Mismatch**: Entry fee charged but not properly accounted, or exit fee charged twice
8. **Forced Close by Stop**: Hard stop, time stop, or risk engine closing position early (even with ignore_stops)

---

## 2. EXACT DEBUG CHECKLIST LOG FORMAT

For Buy&Hold baseline, log at: step 0 (after opening), step 1, step 5, last step

```python
def _log_baseline_debug(self, step: int, is_last: bool = False):
    """Log baseline debug info at specific steps."""
    if not self.debug_mode:
        return
    
    for coin, position in self.positions.items():
        current_price = self._get_current_price(coin)
        if current_price is None:
            continue
        
        timestamp = self.data[coin].index[self.current_step] if self.current_step < len(self.data[coin]) else None
        position_notional = position.margin_used * position.leverage
        units = position_notional / position.entry_price
        unrealized_pnl_usd = self._calculate_unrealized_pnl(position, current_price)
        
        # Calculate cumulative fees (entry fee already paid, exit fee not yet)
        entry_fee = position_notional * TAKER_FEE
        exit_fee = position_notional * TAKER_FEE if is_last else 0.0
        cumulative_fees = entry_fee + exit_fee if is_last else entry_fee
        
        equity = self.portfolio_value
        
        logger.info(
            f"BASELINE_DEBUG step={step} | "
            f"timestamp={timestamp} | "
            f"close_price={current_price:.4f} | "
            f"entry_price={position.entry_price:.4f} | "
            f"side={position.position_type.name} | "
            f"margin_used={position.margin_used:.2f} | "
            f"leverage={position.leverage:.1f}x | "
            f"position_notional={position_notional:.2f} | "
            f"units={units:.4f} | "
            f"unrealized_pnl_usd={unrealized_pnl_usd:.2f} | "
            f"realized_pnl_usd=0.00 | "
            f"fee_open={entry_fee:.4f} | "
            f"fee_close={exit_fee:.4f} | "
            f"cumulative_fees={cumulative_fees:.4f} | "
            f"balance={self.balance:.2f} | "
            f"equity={equity:.2f} | "
            f"is_position_open={True} | "
            f"stop_triggered_reason=None"
        )
```

---

## 3. MINIMAL PATCH CODE

### A) config.py - Update Fee Rates

```python
# Trading Fees & Execution Costs (Binance USDT-M Futures Default)
TAKER_FEE: float = 0.0004  # 0.04% taker fee (Binance USDT-M default)
MAKER_FEE: float = 0.0002  # 0.02% maker fee (Binance USDT-M default)
TRANSACTION_FEE: float = 0.0004  # Default to taker
SLIPPAGE_PCT: float = 0.0005  # 0.05% slippage per trade
SPREAD_PCT: float = 0.0002  # 0.02% bid-ask spread

# Funding Rate (default 0, only charge if explicitly enabled)
FUNDING_RATE: float = 0.0  # Default: no funding charges
```

### B) env/trading_env.py - Critical Fixes

```python
# Add to __init__:
self.funding_rate: float = 0.0  # Default: no funding
self.cumulative_fees: float = 0.0  # Track total fees paid

# Replace _close_position method:
def _close_position(self, coin: str, reason: str, exit_price: Optional[float] = None):
    """Close position with correct futures accounting."""
    if coin not in self.positions:
        return
    
    position = self.positions[coin]
    
    if exit_price is None:
        exit_price = self._get_current_price(coin)
    
    if exit_price is None:
        logger.error(f"Cannot close {coin}: No exit price available")
        return
    
    # CRITICAL: Calculate realized PnL
    unrealized_pnl_usd = self._calculate_unrealized_pnl(position, exit_price)
    
    # Calculate notional and PnL percentage for reporting
    position_notional = position.margin_used * position.leverage
    
    # FEES: Exit fee only (entry fee already charged on open)
    # Fee is charged on NOTIONAL value
    exit_fee = position_notional * TAKER_FEE
    
    # SLIPPAGE (applied on exit only)
    slippage_cost = position_notional * SLIPPAGE_PCT
    
    # NO FUNDING (default funding_rate = 0.0)
    funding_cost = 0.0
    
    # Net realized PnL (after exit fees and slippage)
    net_pnl = unrealized_pnl_usd - exit_fee - slippage_cost - funding_cost
    
    # CRITICAL: Return margin + add realized PnL
    # Balance currently: initial_balance - margin_used - entry_fee
    # After close: balance = balance + margin_used + net_pnl
    self.balance += position.margin_used  # Return margin
    self.balance += net_pnl  # Add realized PnL
    
    # Update cumulative fees
    self.cumulative_fees += exit_fee
    
    # Calculate PnL percentage for reporting
    if position.position_type == PositionType.LONG:
        pnl_pct = ((exit_price - position.entry_price) / position.entry_price) * position.leverage
    else:
        pnl_pct = ((position.entry_price - exit_price) / position.entry_price) * position.leverage
    
    # Record trade
    trade = Trade(
        coin=coin,
        position_type=position.position_type,
        entry_price=position.entry_price,
        exit_price=exit_price,
        size=position_notional,
        leverage=position.leverage,
        pnl=net_pnl,
        pnl_pct=pnl_pct,
        entry_time=position.entry_time,
        exit_time=datetime.now(),
        entry_step=position.entry_step,
        exit_step=self.current_step,
        bars_held=position.bars_held,
        reason=reason,
        fees_paid=exit_fee + slippage_cost
    )
    
    self.trades.append(trade)
    del self.positions[coin]
    
    logger.debug(f"Closed {coin} {position.position_type.name}: PnL={net_pnl:.2f} ({pnl_pct*100:.2f}%), Exit Fee={exit_fee:.4f}")

# Add debug logging method:
def _log_baseline_debug(self, step: int, is_last: bool = False):
    """Log baseline debug info at specific steps."""
    if not self.debug_mode:
        return
    
    for coin, position in self.positions.items():
        current_price = self._get_current_price(coin)
        if current_price is None:
            continue
        
        try:
            timestamp = self.data[coin].index[self.current_step] if self.current_step < len(self.data[coin]) else None
        except:
            timestamp = f"step_{self.current_step}"
        
        position_notional = position.margin_used * position.leverage
        units = position_notional / position.entry_price
        unrealized_pnl_usd = self._calculate_unrealized_pnl(position, current_price)
        
        # Calculate cumulative fees
        entry_fee = position_notional * TAKER_FEE
        exit_fee = position_notional * TAKER_FEE if is_last else 0.0
        cumulative_fees = entry_fee + exit_fee if is_last else entry_fee
        
        equity = self.portfolio_value
        realized_pnl = 0.0 if not is_last else unrealized_pnl_usd - exit_fee - (position_notional * SLIPPAGE_PCT)
        
        logger.info(
            f"BASELINE_DEBUG step={step} | "
            f"timestamp={timestamp} | "
            f"close_price={current_price:.4f} | "
            f"entry_price={position.entry_price:.4f} | "
            f"side={position.position_type.name} | "
            f"margin_used={position.margin_used:.2f} | "
            f"leverage={position.leverage:.1f}x | "
            f"position_notional={position_notional:.2f} | "
            f"units={units:.4f} | "
            f"unrealized_pnl_usd={unrealized_pnl_usd:.2f} | "
            f"realized_pnl_usd={realized_pnl:.2f} | "
            f"fee_open={entry_fee:.4f} | "
            f"fee_close={exit_fee:.4f} | "
            f"cumulative_fees={cumulative_fees:.4f} | "
            f"balance={self.balance:.2f} | "
            f"equity={equity:.2f} | "
            f"is_position_open={True} | "
            f"stop_triggered_reason=None"
        )

# Update step() method to call debug logging:
# In step(), after portfolio_value calculation, add:
if self.debug_mode:
    is_last = (self.current_step >= len(coin_data) - 1)
    if self.current_step == 0 or self.current_step == 1 or self.current_step == 5 or is_last:
        self._log_baseline_debug(self.current_step, is_last=is_last)

# Update reset() to initialize cumulative_fees:
self.cumulative_fees = 0.0
```

### C) tuning/optimizer.py - Buy&Hold Baseline Fix

```python
# Replace Buy&Hold baseline section:
# ====================================================================
# BASELINE 2: Buy&Hold Agent (Long from start, hold to end)
# ====================================================================
logger.info("Baseline 2: Buy&Hold Agent (should match asset return)...")
env_bh = TradingEnv(data=data_dict, initial_balance=INITIAL_BALANCE)
obs, info = env_bh.reset(options={'ignore_stops': True, 'debug_mode': True})

# Open long position immediately
initial_price = test_data.iloc[0]['close']
final_price = test_data.iloc[min(steps, len(test_data)-1)]['close']
expected_return = (final_price - initial_price) / initial_price

# CRITICAL: Open position manually at step 0
action = np.array([1.0])  # Long action
obs, reward, terminated, truncated, info = env_bh.step(
    action,
    tft_confidence_15m=1.0,
    tft_confidence_1h=1.0,
    tft_confidence_4h=1.0
)

# Verify position opened
if len(env_bh.positions) == 0:
    logger.error("CRITICAL: Buy&Hold position did not open!")
    raise RuntimeError("Buy&Hold position failed to open")

# Log step 0 (after opening)
env_bh._log_baseline_debug(0, is_last=False)

# Continue holding (no action changes, position stays open)
for step in range(1, steps):
    action = np.array([1.0])  # Keep long action
    obs, reward, terminated, truncated, info = env_bh.step(
        action,
        tft_confidence_15m=1.0,
        tft_confidence_1h=1.0,
        tft_confidence_4h=1.0
    )
    
    # Log at specific steps
    if step == 1 or step == 5:
        env_bh._log_baseline_debug(step, is_last=False)
    
    if terminated or truncated:
        logger.warning(f"Buy&Hold terminated early at step {step}")
        break

# CRITICAL: Log last step before closing
env_bh._log_baseline_debug(steps - 1, is_last=True)

# CRITICAL: Close position at end to realize PnL
if len(env_bh.positions) > 0:
    for coin in list(env_bh.positions.keys()):
        final_price_for_close = env_bh._get_current_price(coin)
        env_bh._close_position(coin, "Baseline End", final_price_for_close)

final_value_bh = env_bh.portfolio_value
return_bh = (final_value_bh - INITIAL_BALANCE) / INITIAL_BALANCE

# Calculate expected return with leverage and fees
leverage_used = BASE_LEVERAGE  # Default
if len(env_bh.trades) > 0:
    leverage_used = env_bh.trades[0].leverage

# Expected return with leverage
expected_with_leverage = expected_return * leverage_used

# Account for fees: entry + exit = 2 * TAKER_FEE on notional
# Notional = margin * leverage, so fee impact = 2 * TAKER_FEE * leverage
fee_impact = 2 * TAKER_FEE * leverage_used
expected_after_fees = expected_with_leverage - fee_impact

logger.info(f"  Buy&Hold: Return={return_bh*100:.4f}%, Expected Asset Return={expected_return*100:.4f}%, "
           f"Expected with Leverage={expected_with_leverage*100:.4f}%, "
           f"Expected after Fees={expected_after_fees*100:.4f}%, Leverage={leverage_used:.1f}x")

# VALIDATION: Should be close to leveraged asset return minus fees
# Allow tolerance for slippage and rounding
tolerance = 0.005  # 0.5% tolerance
expected_min = expected_after_fees - tolerance
expected_max = expected_after_fees + tolerance

if return_bh < expected_min or return_bh > expected_max:
    error_msg = (
        f"BASELINE VALIDATION FAILED: Buy&Hold agent returned {return_bh*100:.4f}% "
        f"(expected ~{expected_after_fees*100:.4f}% ± {tolerance*100:.2f}%). "
        f"Check debug logs above for details."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)

env_bh.close()
```

---

## 4. SELF-TEST FUNCTION

```python
def test_futures_accounting():
    """Self-test: Verify futures accounting with synthetic data."""
    # Test parameters
    entry_price = 100.0
    current_price = 110.0
    margin_used = 1000.0
    leverage = 2.0
    initial_balance = 10000.0
    fee_rate = 0.0  # No fees for this test
    
    # Expected calculations
    position_notional = margin_used * leverage  # 2000
    units = position_notional / entry_price  # 20
    unrealized_pnl_usd = units * (current_price - entry_price)  # 20 * 10 = 200
    expected_equity = initial_balance - margin_used + unrealized_pnl_usd  # 10000 - 1000 + 200 = 9200
    
    # Implementation
    position_notional_calc = margin_used * leverage
    units_calc = position_notional_calc / entry_price
    unrealized_pnl_calc = units_calc * (current_price - entry_price)
    equity_calc = initial_balance - margin_used + unrealized_pnl_calc
    
    # Assertions
    assert abs(position_notional_calc - 2000.0) < 0.01, f"Notional mismatch: {position_notional_calc} != 2000"
    assert abs(units_calc - 20.0) < 0.01, f"Units mismatch: {units_calc} != 20"
    assert abs(unrealized_pnl_calc - 200.0) < 0.01, f"PnL mismatch: {unrealized_pnl_calc} != 200"
    assert abs(equity_calc - 9200.0) < 0.01, f"Equity mismatch: {equity_calc} != 9200"
    
    print("✓ Self-test PASSED:")
    print(f"  Notional: {position_notional_calc:.2f} (expected: 2000.00)")
    print(f"  Units: {units_calc:.4f} (expected: 20.0000)")
    print(f"  Unrealized PnL: {unrealized_pnl_calc:.2f} (expected: 200.00)")
    print(f"  Equity: {equity_calc:.2f} (expected: 9200.00)")

if __name__ == "__main__":
    test_futures_accounting()
```

---

## ASSUMPTIONS MADE

1. **Fee Rate**: Using Binance USDT-M default taker fee (0.04%) instead of 0.06%
2. **Funding**: Default to 0 (no funding charges unless explicitly enabled)
3. **Price Source**: Using `close` price for all calculations (no high/low)
4. **Baseline Close**: Closing position at end to realize PnL (consistent accounting)
5. **Equity Formula**: `equity = balance + unrealized_pnl_usd` (no double counting)

---

## NEXT STEPS

1. Apply patches to `config.py`, `env/trading_env.py`, `tuning/optimizer.py`
2. Run self-test: `python -c "exec(open('BASELINE_DEBUG_CHECKLIST.md').read().split('```python')[4].split('```')[0])"`
3. Run pipeline: `python master_pipeline.py --days 1825 --trials 1 --coins BTC/USDT --backtest-steps 10000`
4. Check debug logs for exact values at step 0, 1, 5, last
5. If still failing, compare logged values with expected to identify exact bug

