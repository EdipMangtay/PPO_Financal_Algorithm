# Dynamic Trailing Stop (Chandelier Exit) Implementation

## Overview

The trading environment now uses a **Dynamic Trailing Stop** mechanism (Chandelier Exit) instead of fixed take profits. This maximizes trend capture while protecting profits.

## How It Works

### 1. Initialization
When a trade opens:
- **LONG:** Initial SL = `Entry_Price - (ATR * 1.5)`
- **SHORT:** Initial SL = `Entry_Price + (ATR * 1.5)`

### 2. Dynamic Update (Every Candle)
The stop loss is recalculated at every step:

**For LONG positions:**
```
New_SL = max(Previous_SL, Current_High - (ATR * trailing_mult))
```

**For SHORT positions:**
```
New_SL = min(Previous_SL, Current_Low + (ATR * trailing_mult))
```

**Key Constraint:** The SL can ONLY move in the direction of profit. It never widens risk.

### 3. Profit Locking

#### Breakeven Trigger (1.5% profit)
When unrealized PnL exceeds 1.5%:
- Stop loss automatically moves to `Entry_Price` (breakeven)
- Protects against losses after initial profit

#### Aggressive Trailing (5.0% profit)
When unrealized PnL exceeds 5.0%:
- Trailing multiplier tightens from `2.5` to `1.5`
- More aggressive protection against reversals
- Locks in larger gains

### 4. Stop Execution
- Uses **High/Low prices** (not Close) for immediate execution
- **LONG:** If `Low <= SL` → Market close at SL price
- **SHORT:** If `High >= SL` → Market close at SL price

## Configuration Parameters

In `config.py`:

```python
INITIAL_SL_ATR_MULTIPLIER: float = 1.5  # Initial stop distance
TRAILING_STOP_ATR_MULTIPLIER: float = 2.5  # Trailing distance
ATR_PERIOD: int = 14  # ATR calculation period

BREAKEVEN_TRIGGER_PCT: float = 1.5  # Breakeven at 1.5% profit
AGGRESSIVE_TRAILING_TRIGGER_PCT: float = 5.0  # Tighten at 5.0% profit
AGGRESSIVE_TRAILING_MULTIPLIER: float = 1.5  # Tightened multiplier
```

## Optimization

Use `tuner.py` to find optimal parameters:

```bash
python tuner.py --trials 50 --steps 1000 --objective sharpe_ratio
```

The tuner optimizes:
- **ATR Period:** 7, 14, or 21
- **Trailing Multiplier:** 1.5 to 3.5 (in steps of 0.25)

**Objective:** Find the "sweet spot" where the bot:
- Allows coins to "breathe" (handles volatility)
- Exits immediately when trend breaks

## Benefits

1. **Maximizes Trend Capture:** Stays in winning trades longer
2. **Protects Profits:** Automatic breakeven and aggressive trailing
3. **Reduces Whipsaws:** Only moves in profit direction
4. **Immediate Execution:** Uses High/Low for faster exits

## Example Flow

**LONG Position:**
1. Entry at $100, Initial SL at $98.50 (ATR=1.0, mult=1.5)
2. Price rises to $102 → SL moves to $99.50 (High=$102, ATR*2.5=2.5)
3. PnL reaches 1.5% → SL moves to $100 (breakeven)
4. Price rises to $105 → SL moves to $102.50
5. PnL reaches 5.0% → Trailing tightens to 1.5x
6. Price hits $108 → SL moves to $105.90 (more aggressive)
7. Price reverses, Low hits $105.90 → Exit at $105.90

## Technical Details

- **Position Tracking:** Each position tracks:
  - `highest_price` (LONG) or `lowest_price` (SHORT)
  - `stop_loss` (current dynamic stop)
  - `trailing_mult` (current multiplier)
  - `breakeven_set` (whether breakeven triggered)

- **Update Frequency:** Trailing stops updated every step before checking for hits

- **Market Close:** Executes immediately when High/Low touches stop

