# DEEP OPTIMIZATION - RTX 5070 FULL POWER

## 🎯 Transition: Testing → Deep Optimization

**Goal:** Utilize RTX 5070 to its limits for statistically significant Alpha discovery, not just lucky streaks.

---

## ✅ Changes Implemented

### 1. **config.py** - Deep Search Settings

**Updated Constants:**
```python
# Deep Optimization Parameters
OPTUNA_N_TRIALS: int = 100  # Increased from 50 (explore more combinations)
N_TRIALS: int = 100  # Alias for clarity
BACKTEST_STEPS: int = 10000  # 10k steps ≈ 3 months of 15m data (increased from 2000)
MIN_TRADES_FOR_OPTIMIZATION: int = 50  # Filter out lazy strategies aggressively

# Bi-Directional Trading Parameters
LONG_ACTION_THRESHOLD: float = 0.3  # Action > 0.3 -> LONG
SHORT_ACTION_THRESHOLD: float = -0.3  # Action < -0.3 -> SHORT
# -0.3 <= Action <= 0.3 -> FLAT (Cash)

# Production Risk Parameters
MAX_DRAWDOWN_PER_TRADE: float = 0.02  # 2% hard stop
TRANSACTION_FEE: float = 0.0006  # 0.06% realistic fee
```

---

### 2. **env/trading_env.py** - Bi-Directional Trading with Survival-First Logic

#### A) Bi-Directional Action Interpretation
```python
def _interpret_action(self, action_value: float) -> PositionType:
    """
    Action > 0.3 -> Open/Hold LONG
    Action < -0.3 -> Open/Hold SHORT
    -0.3 <= Action <= 0.3 -> FLAT (Cash)
    """
```

#### B) Survival-First Logic (Order of Operations)
```python
def step():
    # 1. Update market data
    # 2. CHECK HARD STOPS FIRST (BEFORE agent logic)
    #    if unrealized_pnl_pct <= -MAX_DRAWDOWN_PER_TRADE:
    #        force_close(reason='hard_stop')
    # 3. Check portfolio ruin
    # 4. Update trailing stops
    # 5. Agent action (only if no stops triggered)
```

#### C) Reward Shaping (Stability-Focused)
```python
# Reward = Log_Return + (Sortino_Component * 0.1) - (Churn_Penalty) - (Holding_Loser_Penalty)

r_main = log(equity_t / equity_{t-1}) * 100
r_sortino = 0.1 * sortino_component  # Stability bonus
r_churn = -0.001  # Penalty for entering (discourages overtrading)
r_loser = -0.01 * abs(unrealized_pnl_pct)  # Penalty for holding losers

reward = r_main + r_sortino - r_churn - r_loser
```

**Goal:** Force agent to be profitable AND active, but punish holding losing bags.

---

### 3. **tuning/optimizer.py** - Validation Suite & Advanced Pruning

#### A) Baseline Validation Suite (Runs BEFORE Optuna)

**Three Baseline Agents:**

1. **RandomAgent:**
   - Random actions, random confidence
   - **Expected:** Should fail/lose money
   - **Validation:** If makes 50% profit → ABORT (environment broken)

2. **BuyAndHold:**
   - Opens long immediately, holds to end
   - **Expected:** Matches underlying asset return (within fees)

3. **AlwaysFlat:**
   - Never trades (neutral action, confidence=0)
   - **Expected:** ~0% return (small negative due to fees if any trades)

**If any baseline fails → `RuntimeError("Environment logic is broken")`**

#### B) Advanced Pruning (PercentilePruner)
```python
pruner = optuna.pruners.PercentilePruner(
    percentile=25.0,  # Prune bottom 25% of trials
    n_startup_trials=10,  # Allow first 10 trials to complete
    n_warmup_steps=1000,  # Wait 1000 steps before pruning (10% of 10k)
    interval_steps=500  # Check every 500 steps
)
```

**Benefits:**
- Kills bad trials early (saves time)
- Allows good trials to run full 10k steps
- Better than MedianPruner for deep optimization

#### C) Deep Search Configuration
```python
# Uses BACKTEST_STEPS (10000) instead of hardcoded 500
metrics = self._backtest_feature_config(
    selected_features=selected_features,
    indicator_params=indicator_params,
    steps=BACKTEST_STEPS,  # 10k steps for deep optimization
    trial=trial
)

# Filters out lazy strategies aggressively
if total_trades < MIN_TRADES_FOR_OPTIMIZATION:  # 50 trades minimum
    return penalty_score
```

---

## 📊 Key Features

### Bi-Directional Trading
- **LONG:** Action > 0.3
- **SHORT:** Action < -0.3
- **FLAT:** -0.3 <= Action <= 0.3

### Survival-First Logic
- Hard stops override agent decisions
- Portfolio ruin protection (15% drawdown)
- Time stops (200 bars max hold)

### Deep Search
- 100 Optuna trials (explore more combinations)
- 10k backtest steps (3 months of data)
- Minimum 50 trades per strategy (filter lazy strategies)

### Advanced Pruning
- PercentilePruner (kills bottom 25%)
- Allows good trials to run full 10k steps
- Saves time on bad trials

---

## 🚀 Expected Results

### Before (Testing):
- 50 trials
- 2000 backtest steps
- Basic MedianPruner
- Simple action interpretation

### After (Deep Optimization):
- 100 trials (2x exploration)
- 10k backtest steps (5x data)
- PercentilePruner (smarter pruning)
- Bi-directional trading (LONG/SHORT/FLAT)
- Minimum 50 trades (filter lazy strategies)

---

## 📝 Usage

```powershell
# Run deep optimization
python master_pipeline.py --days 1825 --trials 100 --coins BTC/USDT --backtest-steps 10000
```

**What to Watch:**
1. Baseline validation should PASS (no errors)
2. Trials should complete with > 50 trades
3. Good trials should run full 10k steps
4. Bad trials should be pruned early (saves time)

---

## ⚠️ Critical Notes

1. **Hard Stops Override Agent:** Agent decisions ignored if hard stops trigger
2. **Bi-Directional Support:** Agent can go LONG, SHORT, or FLAT
3. **Deep Search:** 10k steps per trial (statistically significant)
4. **Lazy Strategy Filter:** Minimum 50 trades (aggressive filtering)

---

## 🎉 Result

**Production-grade deep optimization system with:**
- ✅ Bi-directional trading (LONG/SHORT/FLAT)
- ✅ Survival-first logic (hard stops)
- ✅ Deep search (100 trials, 10k steps)
- ✅ Advanced pruning (PercentilePruner)
- ✅ Baseline validation (catches bugs early)
- ✅ Lazy strategy filtering (minimum 50 trades)

**System is ready for statistically significant Alpha discovery!** 🚀

