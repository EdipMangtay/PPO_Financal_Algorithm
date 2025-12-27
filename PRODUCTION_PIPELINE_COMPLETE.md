# Production Pipeline - Complete Implementation

## ✅ Implementation Complete

All required modules have been implemented:

### Directory Structure

```
├── config/
│   ├── train.yaml          # Main config with HPO settings
│   ├── features.yaml       # Feature engineering config
│   └── paths.yaml         # All file paths
├── scripts/
│   ├── doctor.py           # Environment validation
│   ├── preflight.py        # Preflight checks + auto-fix
│   └── run_all_new.py      # MAIN ORCHESTRATOR (use this)
├── utils/
│   ├── seed.py            # Deterministic seeding
│   ├── logging.py         # Structured logging
│   ├── io.py              # Safe file operations
│   └── device.py          # GPU/CPU detection
├── hpo/
│   └── optuna_search.py   # Optuna HPO per timeframe
├── training/
│   ├── trainer.py         # Safe training wrapper (OOM/NaN guards)
│   ├── train_one.py       # Single timeframe training (legacy)
│   └── train_final.py     # Final training with best params
├── evaluation/
│   └── metrics.py         # Regression/classification metrics
├── backtest/
│   ├── engine.py          # Event-driven backtest
│   ├── backtest.py        # Backtest on test data
│   └── plots.py           # Visualization
└── artifacts/             # Output directory (auto-created)
    └── {run_id}/
        ├── summary.json   # Global summary
        └── {timeframe}/
            ├── optuna.db
            ├── optuna_best.json
            ├── model.pt
            ├── metrics_test.json
            ├── preds_test.parquet
            ├── backtest_metrics.json
            ├── trades.csv
            ├── equity.csv
            └── equity_curve.png
```

## 🚀 Commands

### Main Command (THE ONLY ONE YOU NEED)

```bash
python scripts/run_all_new.py --config config/train.yaml --hpo_trials 50
```

### Options

```bash
# Skip HPO (use config defaults)
python scripts/run_all_new.py --config config/train.yaml --skip_hpo

# Resume existing Optuna study
python scripts/run_all_new.py --config config/train.yaml --resume_hpo

# Set HPO timeout
python scripts/run_all_new.py --config config/train.yaml --hpo_trials 50 --hpo_timeout_minutes 120

# Continue on error (don't stop if one timeframe fails)
python scripts/run_all_new.py --config config/train.yaml --continue_on_error
```

## 📋 Pipeline Steps (Per Timeframe)

1. **Preflight Checks** → Auto-fix common issues
2. **Load Data** → Load and validate OHLCV data
3. **Build Features** → Feature engineering with validation
4. **Split Data** → Time-based split (train/val/test)
5. **Optuna HPO** → Hyperparameter optimization (if enabled)
6. **Train Final Model** → Train with best params on train+val
7. **Evaluate on Test** → Compute metrics (MAE, RMSE, R2)
8. **Backtest on Test** → Run backtest with exact metrics
9. **Save Results** → All artifacts saved

## 📊 Outputs Per Timeframe

- `optuna.db` - Optuna study database
- `optuna_best.json` - Best hyperparameters
- `model.pt` - Trained model weights
- `metrics_test.json` - Test set metrics
- `preds_test.parquet` - Test predictions
- `backtest_metrics.json` - Backtest metrics
- `trades.csv` - All trades
- `equity.csv` - Equity curve
- `equity_curve.png` - Plot

## 🔧 Features

✅ **Three Independent Models** - 15m, 1h, 4h trained separately
✅ **Optuna HPO** - Per timeframe with MedianPruner
✅ **Preflight Auto-Fix** - Handles NaNs, duplicates, OOM
✅ **OOM/NaN Guards** - Automatic error recovery
✅ **Deterministic** - Same seed = same results
✅ **No Lookahead** - Strict time-based splits
✅ **Exact Metrics** - Precise numerical calculations
✅ **Robust Error Handling** - No silent failures

## ⚙️ Configuration

Edit `config/train.yaml` to customize:
- Coin symbol
- Date range
- HPO settings (trials, timeout)
- Model architecture
- Training hyperparameters
- Backtest parameters

## 📝 Deliverables Checklist

After running, verify:
- ✅ Preflight OK
- ✅ Optuna DB created per tf
- ✅ optuna_best.json created per tf
- ✅ model.pt saved per tf
- ✅ metrics_test.json saved per tf
- ✅ backtest_summary.json saved per tf
- ✅ global summary.json saved

## 🎯 Next Steps

1. Install missing packages: `pip install torch pytorch-forecasting optuna pyyaml`
2. Run: `python scripts/run_all_new.py --config config/train.yaml --hpo_trials 50`
3. Check `artifacts/{run_id}/summary.json` for results


