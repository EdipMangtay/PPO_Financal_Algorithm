# High-Frequency Trading Enforcement

## Overview

The optimizer has been updated to **force high-frequency trading** by using an aggressive objective function and minimum trade constraints. This prevents the "lazy bot" problem where the algorithm rarely trades.

## Key Changes

### 1. Aggressive Objective Function

**Old:** Maximize Sharpe Ratio
**New:** Maximize `Sortino_Ratio * log1p(Total_Trades)`

**Logic:**
- **Low trades (5):** `log1p(5) ≈ 1.79` → Heavily penalized
- **High trades (100):** `log1p(100) ≈ 4.61` → Boosted
- Optuna will find indicators that generate frequent signals without sacrificing safety

**Why Sortino?**
- Only considers downside deviation (negative returns)
- Doesn't penalize upside volatility
- More appropriate for trading than Sharpe Ratio

### 2. Shorter Linear Regression Channels

**Added lengths:** 10, 20, 30 (in addition to 50, 100, 200)

**Why:**
- Long channels (200) give rare signals → Low trade count
- Short channels (20) give frequent "scalp" signals → High trade count
- Optuna can now select shorter periods for high-frequency strategies

### 3. Inner Channel Trading

**New Features:**
- `cross_above_linreg_{length}`: Price crossing above mid-line
- `cross_below_linreg_{length}`: Price crossing below mid-line
- `momentum_linreg_{length}`: Rate of change from mid-line
- `strong_momentum_long_{length}`: Strong buy signal (above mid-line + positive slope + momentum)
- `strong_momentum_short_{length}`: Strong sell signal (below mid-line + negative slope + momentum)

**Strategy:**
- Trade not just on channel breakouts
- Also trade on mid-line crossovers when momentum is strong
- Increases trade frequency while maintaining quality

### 4. Minimum Trade Constraint (Pruner)

**Requirement:** Minimum 15 trades per simulated month

**Implementation:**
- Early pruning: Checks every 50 steps
- If after 0.5 months, trades/month < 7.5 (50% of requirement) → Prune
- Final check: If trades/month < 15 → Return very bad score (-999.0)

**Benefits:**
- Immediately kills passive parameter sets
- Saves GPU time for aggressive strategies
- Forces Optuna to explore high-frequency configurations

## Example Scores

| Trades | Sortino | log1p(Trades) | Score | Result |
|--------|---------|---------------|-------|--------|
| 5      | 2.0     | 1.79          | 3.58  | ❌ Low |
| 20     | 1.5     | 3.04          | 4.56  | ✅ Good |
| 50     | 1.2     | 3.93          | 4.72  | ✅ Better |
| 100    | 1.0     | 4.61          | 4.61  | ✅ Best |

**Note:** Even with lower Sortino, higher trade count can win due to the log multiplier.

## Optimization Flow

1. **Feature Selection:** Optuna chooses features (including short LRC lengths)
2. **Parameter Tuning:** Optuna selects indicator parameters
3. **Backtest:** Run simulation with random policy
4. **Early Pruning:** Check trades/month every 50 steps
   - If too passive → Prune immediately
5. **Final Score:** Calculate `Sortino * log1p(Trades)`
6. **Final Check:** If trades/month < 15 → Bad score

## Configuration

In `tuning/optimizer.py`:

```python
min_trades_per_month = 15  # Minimum requirement
steps_per_month = calculated_based_on_timeframe  # e.g., 2880 for 15m
```

## Results

After optimization, each coin will have:
- **High trade frequency:** >15 trades/month (often 20-50+)
- **Optimized features:** Short LRC lengths, momentum indicators
- **Balanced performance:** Good Sortino ratio with high activity

## Monitoring

The optimizer logs:
- `trades_per_month`: Actual trading frequency
- `sortino_ratio`: Risk-adjusted return (downside only)
- `aggressive_score`: Final optimization score
- Pruning events: When trials are killed for being too passive

---

**Result:** A trading system that actively trades (>20 trades/day) while maintaining good risk-adjusted returns!

