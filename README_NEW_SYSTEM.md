# Trading System - Clean Architecture

## Overview

This is a production-ready trading system that trains **3 independent models** (15m, 1h, 4h) and runs separate backtests for each.

## Structure

```
├── config/              # Configuration files (YAML)
│   ├── train.yaml      # Training configuration
│   ├── features.yaml   # Feature engineering config
│   └── paths.yaml     # All file paths
├── scripts/            # Executable scripts
│   ├── doctor.py      # Environment validation
│   └── run_all.py     # Main orchestrator
├── data/               # Data loading and validation
│   ├── loader_new.py  # Unified data loader
│   ├── validators.py   # Data contract validation
│   ├── resample.py    # Timeframe resampling
│   └── cache.py       # Dataset caching
├── features/           # Feature engineering
│   └── build_features.py
├── training/           # Model training
│   ├── trainer.py     # Safe training wrapper
│   └── train_one.py   # Single timeframe training
├── signals/            # Signal generators
│   ├── signal_base.py
│   ├── signal_15m.py
│   ├── signal_1h.py
│   └── signal_4h.py
├── backtest/           # Backtesting engine
│   ├── engine.py      # Event-driven backtest
│   └── plots.py       # Visualization
├── tests/              # Tests
└── artifacts/          # Output directory (created automatically)
```

## Quick Start

### 1. Check Environment

```bash
python scripts/doctor.py
```

### 2. Run Complete Pipeline

```bash
python scripts/run_all.py --config config/train.yaml
```

This will:
- Train 3 models (15m, 1h, 4h)
- Generate signals for each
- Run separate backtests
- Save all results to `artifacts/{run_id}/`

## Configuration

Edit `config/train.yaml` to change:
- Coin symbol
- Date range
- Training hyperparameters
- Batch sizes per timeframe

## Output

After running, check:
- `artifacts/{run_id}/run_summary.json` - Complete summary
- `artifacts/{run_id}/{timeframe}/backtest_metrics.json` - Per-timeframe metrics
- `artifacts/{run_id}/{timeframe}/trades.csv` - All trades
- `artifacts/{run_id}/{timeframe}/equity.csv` - Equity curve
- `artifacts/{run_id}/{timeframe}/equity_curve.png` - Plot

## Metrics

Each backtest includes:
- Total Return (%)
- CAGR (%)
- Max Drawdown (%)
- Sharpe Ratio
- Sortino Ratio
- Win Rate (%)
- Profit Factor
- Average Trade Return
- Expectancy
- Total Trades
- Exposure (%)
- Turnover

## Requirements

See `requirements.txt` for full list. Key packages:
- torch
- pytorch-forecasting
- pandas
- numpy
- pyyaml
- matplotlib

## Notes

- Models are trained **independently** (no ensemble)
- Each timeframe gets its own backtest
- All results are saved with exact numerical values
- System handles OOM, NaN, and other errors gracefully


