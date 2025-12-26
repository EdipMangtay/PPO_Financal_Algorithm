# Auto-Feature Engineering System

## Overview

The GOD_LEVEL_TRADER_FINAL implements **Dynamic Feature Selection** using Optuna. Instead of hardcoding indicator parameters, the system automatically discovers the perfect combination of indicators and their settings for **each coin and each timeframe independently**.

## Architecture

### 1. Feature Pool Generation (`data_engine/features.py`)

The system generates a **massive pool** of candidate features:

- **Linear Regression Channels:** Core indicators with multiple lengths (50, 100, 200)
  - `linreg`, `slope`, `intercept`, `r-squared`
  - Upper/Lower channels with distance metrics
  - **Strategy:** Price touching Lower Channel + Positive Slope = Strong Buy Signal

- **Oscillators:** RSI, Stochastic, CCI, Williams%R (periods: 7, 14, 21, 50)

- **Trend Indicators:** EMA Ribbons, SuperTrend, ADX, MACD (multiple configurations)

- **Volatility:** ATR, Bollinger Bands, Keltner Channels (various periods)

- **Volume:** OBV, MFI, VWAP, Volume Ratios

- **Price Features:** Price changes, high/low ratios

**Total:** 100+ candidate features per coin

### 2. Two-Layer Optuna Optimization (`tuning/optimizer.py`)

#### Layer 1: Feature Selection
Optuna chooses **which features** to use:
- Toggles feature categories on/off
- Selects subset of features from each category
- Example: "For SOL-15m, use [RSI_9, LinReg_50_Slope, VWAP]. Ignore MACD."

#### Layer 2: Parameter Tuning
Optuna selects **parameter values** for chosen indicators:
- `rsi_period = trial.suggest_int('rsi_period', 5, 30)`
- `linreg_length = trial.suggest_int('linreg_len', 20, 200)`
- `atr_period = trial.suggest_int('atr_period', 7, 21)`

**Result:** Each coin gets its own "DNA" - optimal feature set and parameters

### 3. High-Frequency Training

- **5 Million Steps:** PPO training increased to 5M steps for high-frequency learning
- **Reward Shaping:**
  - **LinReg Extremes Bonus:** Small bonus for entering when price is at LinReg channels (high statistical probability)
  - **Hesitation Penalty:** Penalizes staying in cash too long (encourages >20 trades/day)

### 4. Pipeline Execution (`main.py`)

#### Full Pipeline Mode:
```bash
python main.py --mode pipeline --days 90 --trials 100
```

**Steps:**
1. **Data Download:** Fetch extensive history (90 days)
2. **Feature Race:** Run Optuna (100 trials per coin) to find optimal features
3. **Deep Training:** Train TFT + PPO using ONLY optimized features
4. **Save Configs:** Feature configs saved to `feature_configs/` as JSON

#### Optimization Only:
```bash
python main.py --mode optimize --trials 100
```

Runs only the feature optimization step.

#### Live Trading:
```bash
python main.py --mode live
```

Loads optimized feature configs and uses only those features during live trading.

## Feature Configuration Format

Each coin/timeframe gets a JSON config file:
```json
{
  "coin": "SOL/USDT",
  "timeframe": "15m",
  "selected_features": [
    "RSI_14",
    "linreg_50",
    "slope_50",
    "dist_lower_50",
    "VWAP",
    "ATR_14",
    ...
  ],
  "indicator_params": {
    "rsi_period": 14,
    "linreg_length": 50,
    "atr_period": 14,
    ...
  },
  "performance": {
    "sharpe_ratio": 1.85,
    "total_return": 0.23,
    "max_drawdown": 0.12
  }
}
```

## Benefits

1. **Coin-Specific Optimization:** Each coin gets features tailored to its behavior
2. **Timeframe-Specific:** 15m features differ from 4h features
3. **No Hardcoding:** All parameters discovered automatically
4. **High Trade Count:** Reward shaping encourages >20 trades/day
5. **High Win Rate:** LinReg extremes provide high-probability entries

## Example: SOL/USDT 15m

After optimization, SOL might use:
- **RSI_9** (faster than default 14)
- **LinReg_50_Slope** (medium-term trend)
- **dist_lower_50** (channel distance)
- **VWAP** (volume-weighted entry)
- **ATR_7** (shorter period for 15m)

While ETH/USDT 15m might use:
- **RSI_21** (slower)
- **LinReg_100** (longer-term)
- **SuperTrend_14_2.5**
- **MFI_14**

Each coin finds its unique "DNA"!

## Performance Metrics

The optimizer maximizes **Sharpe Ratio** to find the sweet spot:
- Allows volatility (doesn't exit too early)
- Exits on trend breaks (protects capital)
- Balances trade frequency with win rate

## File Structure

```
PPO/
├── main.py                    # Main pipeline orchestrator
├── data_engine/
│   └── features.py           # Feature pool generator
├── tuning/
│   └── optimizer.py          # Two-layer Optuna optimizer
├── feature_configs/          # Saved feature configs (JSON)
│   ├── feature_config_SOL_USDT_15m.json
│   ├── feature_config_ETH_USDT_15m.json
│   └── ...
└── ...
```

## Usage Workflow

1. **Initial Setup:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Pipeline:**
   ```bash
   python main.py --mode pipeline --days 90 --trials 100
   ```
   This will:
   - Download 90 days of data
   - Optimize features for each coin (100 trials each)
   - Train models with optimized features (5M steps)
   - Save everything

3. **Live Trading:**
   ```bash
   python main.py --mode live
   ```
   Automatically loads optimized feature configs and uses only those features.

## Advanced Options

```bash
# Optimize specific coins only
python main.py --mode pipeline --coins SOL/USDT ETH/USDT

# Use different timeframe
python main.py --mode pipeline --timeframe 4h

# More trials for better optimization (slower)
python main.py --mode pipeline --trials 200
```

## Notes

- **First Run:** Feature optimization can take several hours (100 trials × 20 coins)
- **Training:** 5M steps training takes significant time (use GPU recommended)
- **Storage:** Feature configs are small JSON files (~10KB each)
- **Live Trading:** Only calculates optimized features (faster than full pool)

---

**Result:** A trading system that automatically discovers the best features for each coin, maximizing both trade frequency and win rate!

