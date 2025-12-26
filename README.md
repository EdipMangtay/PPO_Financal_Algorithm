# GOD_LEVEL_TRADER_V5
## Context-Aware High-Frequency Trading System

A fully autonomous, self-learning trading bot for Top 20 Crypto Futures with context-aware market regime understanding.

### 🎯 Target Metrics
- **Monthly ROI:** +20%
- **Trades/Day:** 15-25
- **Max Drawdown:** < 15%

### 🏗️ Architecture

#### Model A: TFT Oracle (Temporal Fusion Transformer)
- **Purpose:** Predict price probability for next 12 candles
- **Static Covariates:** Coin embeddings (SOL, AVAX, ETH...)
- **Dynamic Covariates (CRITICAL):** BTC_Close, BTC_RSI, USDT_Dominance injected into every coin's input stream
- **Key Feature:** Context-aware - understands market regime (e.g., "If BTC dumps, ignore Bullish altcoin patterns")

#### Model B: PPO Commander (Recurrent PPO)
- **Input:** TFT Confidence Score + Volatility (ATR) + PnL State
- **Action:** Continuous Box(-1, 1) → Long/Short/Neutral
- **Architecture:** LSTM-based for temporal dependencies

### 📊 Financial Parameters (Hard-coded)

All parameters are strictly defined in `config.py`:

- **Leverage:** Dynamic cap at **5x**
- **Risk Management:** Fractional Kelly Criterion (`kelly_fraction = 0.15`)
- **Confidence Threshold:** **0.65** (ensures ~20 trades/day)
- **Dynamic Trailing Stop (Chandelier Exit):**
  - Initial SL: `Entry_Price ± (ATR * 1.5)`
  - Trailing: `High/Low ± (ATR * 2.5)` - moves only in profit direction
  - Breakeven Lock: Moves SL to entry price at **1.5%** profit
  - Aggressive Trailing: Tightens to `ATR * 1.5` at **5.0%** profit
- **Trading Fees:** **0.06%** (Taker fee) simulated in environment

### 🔄 Continuous Learning Pipeline

**Weekly Routine (Sunday 03:00 UTC):**
1. Pause Trading
2. Fetch recent 7 days of data
3. Fine-tune models (5 epochs, lr=1e-5) - gentle update to prevent catastrophic forgetting
4. Validate: Compare New_Model Sharpe vs Old_Model Sharpe
5. Hot Swap if new model performs better
6. Resume Trading

### 🚀 Installation

```bash
# Clone repository
cd PPO

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional, for live trading)
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET="your_secret_key"
```

### 💻 Usage

#### Training Mode
Train models from scratch:

```bash
python run_bot.py --mode train
```

With Optuna hyperparameter optimization (recommended):
```bash
python run_bot.py --mode train
```

Skip Optuna (faster, uses default hyperparameters):
```bash
python run_bot.py --mode train --no-optuna
```

#### Live Trading Mode
Run the 24/7 trading bot with weekly retraining:

```bash
python run_bot.py --mode live
```

The bot will:
- Load pre-trained models
- Start live trading with rich terminal dashboard
- Automatically retrain weekly (Sunday 03:00 UTC)
- Display real-time PnL, positions, and trades
- Use dynamic trailing stops (Chandelier Exit) for optimal exits

#### Exit Parameter Tuning
Optimize ATR period and trailing multiplier:

```bash
python tuner.py --trials 50 --steps 1000 --objective sharpe_ratio
```

This will find the optimal balance between allowing volatility and exiting on trend breaks.

### 📁 Project Structure

```
PPO/
├── config.py              # Central configuration (hard-coded parameters)
├── run_bot.py            # Main CLI entry point
├── scheduler.py          # Weekly retraining scheduler
├── trainer.py            # Model training and fine-tuning
├── validator.py          # Model validation and comparison
├── data/
│   └── loader.py         # Async data loader with BTC/USDT injection
├── models/
│   ├── tft.py           # Temporal Fusion Transformer
│   └── ppo.py           # Recurrent PPO agent
├── env/
│   └── trading_env.py   # Trading environment with Kelly Criterion
├── logs/                # Log files
├── models/checkpoints/  # Saved model checkpoints
└── data/raw/            # Cached market data
```

### 🔧 Configuration

All trading parameters are in `config.py`. Key settings:

- `TOP_20_COINS`: List of coins to trade
- `MAX_LEVERAGE`: Maximum leverage (5x)
- `KELLY_FRACTION`: Fractional Kelly (0.15)
- `CONFIDENCE_THRESHOLD`: Minimum confidence to trade (0.65)
- `STOP_LOSS_ATR_MULTIPLIER`: Stop loss multiplier (1.5)
- `TAKE_PROFIT_ATR_MULTIPLIER`: Take profit multiplier (3.0)
- `TAKER_FEE`: Trading fee (0.0006 = 0.06%)

### 🎨 Dashboard Features

The live trading dashboard displays:
- **Portfolio Metrics:** Value, PnL, Drawdown, Total Trades
- **Active Positions:** Real-time position tracking with PnL
- **Recent Trades:** Last 10 completed trades
- **Status:** Uptime, current coin, trading state

### ⚠️ Important Notes

1. **Hardware:** Optimized for NVIDIA RTX 5070 with `torch.set_float32_matmul_precision('medium')`
2. **Sandbox Mode:** Default is sandbox mode. Set `EXCHANGE_SANDBOX = False` in `config.py` for live trading
3. **USDT Dominance:** Currently uses mock data. Integrate CoinGecko API for production:
   ```python
   # In data/loader.py, replace _fetch_usdt_dominance() with real API call
   ```
4. **Risk Warning:** This is experimental software. Use at your own risk. Start with small amounts.

### 🔍 Key Features

- **Context-Aware:** BTC and USDT dominance injected into every coin's prediction
- **Fractional Kelly:** Conservative risk management (15% of full Kelly)
- **Dynamic Trailing Stops (Chandelier Exit):** 
  - Maximizes trend capture while protecting profits
  - Automatically locks breakeven and tightens trailing at profit thresholds
  - Uses High/Low prices for immediate stop execution
- **Continuous Learning:** Weekly retraining with hot-swap capability
- **Rich Dashboard:** Beautiful terminal UI with real-time metrics
- **Parameter Optimization:** Optuna-based tuning for exit parameters

### 📈 Performance Monitoring

Monitor performance through:
- Terminal dashboard (live)
- Log files in `logs/trading_bot.log`
- Model checkpoints in `models/checkpoints/`

### 🐛 Troubleshooting

**Issue:** Models not found
- **Solution:** Run `python run_bot.py --mode train` first

**Issue:** Data fetch fails
- **Solution:** Check internet connection and API keys (if using live exchange)

**Issue:** CUDA out of memory
- **Solution:** Reduce batch size in `config.py` or use CPU mode

### 📝 License

This project is for educational purposes. Use at your own risk.

### 🤝 Contributing

This is a production trading system. Contributions should focus on:
- Bug fixes
- Performance optimizations
- Additional context features
- Risk management improvements

---

**Built with:** PyTorch, Stable-Baselines3, pytorch-forecasting, ccxt, rich

**Author:** Lead AI Quant Architect & Python MLOps Engineer

