# Final Commands - Production Pipeline

## ✅ Implementation Complete

All modules implemented. System is ready to run.

## 🚀 EXACT COMMAND TO RUN

```bash
python scripts/run_all_new.py --config config/train.yaml --hpo_trials 50
```

## 📋 What Happens

For each timeframe (15m, 1h, 4h):

1. **Preflight** → Validates environment and data, auto-fixes issues
2. **Optuna HPO** → Finds best hyperparameters (50 trials)
3. **Train Final** → Trains model with best params on train+val
4. **Evaluate** → Tests on test set, computes metrics
5. **Backtest** → Runs backtest on test period with exact metrics

## 📊 Output Structure

```
artifacts/
└── {run_id}/
    ├── summary.json              # Global summary
    ├── run.log                   # Global log
    ├── 15m/
    │   ├── run_15m.log
    │   ├── optuna.db             # Optuna study
    │   ├── optuna_best.json      # Best hyperparameters
    │   ├── model.pt              # Trained model
    │   ├── metrics_test.json     # Test metrics (MAE, RMSE, R2)
    │   ├── preds_test.parquet   # Test predictions
    │   ├── backtest_metrics.json # Backtest metrics
    │   ├── trades.csv            # All trades
    │   ├── equity.csv            # Equity curve
    │   └── equity_curve.png     # Plot
    ├── 1h/ (same structure)
    └── 4h/ (same structure)
```

## ⚙️ Configuration Options

Edit `config/train.yaml`:
- `coin`: Coin symbol
- `date_range`: Training data range
- `hpo.n_trials`: Number of Optuna trials
- `hpo.timeout_minutes`: HPO timeout
- `hpo.skip`: Skip HPO entirely
- `backtest.signal_threshold`: Signal generation threshold

## 🔧 Command Options

```bash
# Skip HPO (use config defaults)
--skip_hpo

# Resume existing Optuna study
--resume_hpo

# Set HPO timeout (minutes)
--hpo_timeout_minutes 120

# Continue if one timeframe fails
--continue_on_error
```

## 📈 Metrics Provided

**Test Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R2 (R-squared)

**Backtest Metrics:**
- Total Return (%)
- CAGR (%)
- Max Drawdown (%)
- Sharpe Ratio
- Sortino Ratio
- Win Rate (%)
- Profit Factor
- Total Trades
- Average Trade Return
- Expectancy
- Exposure (%)
- Turnover

## ✅ Deliverables Checklist

After running, verify all files exist:
- ✅ `artifacts/{run_id}/summary.json`
- ✅ `artifacts/{run_id}/{tf}/optuna.db` (if HPO enabled)
- ✅ `artifacts/{run_id}/{tf}/optuna_best.json` (if HPO enabled)
- ✅ `artifacts/{run_id}/{tf}/model.pt`
- ✅ `artifacts/{run_id}/{tf}/metrics_test.json`
- ✅ `artifacts/{run_id}/{tf}/backtest_metrics.json`

## 🎯 Ready to Run

System is production-ready. Install dependencies and run:

```bash
pip install torch pytorch-forecasting optuna pyyaml matplotlib pandas numpy scikit-learn
python scripts/run_all_new.py --config config/train.yaml --hpo_trials 50
```


