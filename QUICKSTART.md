# Quick Start Guide

## 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## 2. Training (First Time)

Train the models from scratch:

```bash
python run_bot.py --mode train
```

This will:
- Fetch 30 days of historical data
- Optimize TFT hyperparameters with Optuna (optional, can skip with `--no-optuna`)
- Pretrain TFT model
- Train PPO agent
- Save models to `models/checkpoints/`

**Expected time:** 1-3 hours (depending on Optuna trials)

## 3. Live Trading

Once models are trained, start live trading:

```bash
python run_bot.py --mode live
```

The bot will:
- Load pre-trained models
- Start trading with real-time dashboard
- Automatically retrain weekly (Sunday 03:00 UTC)

## 4. Configuration

Edit `config.py` to adjust:
- Trading parameters (leverage, Kelly fraction, etc.)
- Model hyperparameters
- Exchange settings (sandbox vs live)

## 5. Monitor Performance

- **Live Dashboard:** Real-time metrics in terminal
- **Logs:** Check `logs/trading_bot.log`
- **Models:** Saved in `models/checkpoints/`

## Troubleshooting

**Models not found?**
→ Run training mode first: `python run_bot.py --mode train`

**Data fetch fails?**
→ Check internet connection and API keys

**CUDA errors?**
→ Models will fall back to CPU automatically

## Next Steps

1. Start with sandbox mode (default)
2. Monitor performance for 1-2 weeks
3. Adjust parameters in `config.py` if needed
4. Switch to live mode when confident

