"""
GOD_LEVEL_TRADER_V5 - Central Configuration
Hard-coded financial parameters for production trading system.
"""

from typing import List, Dict
from dataclasses import dataclass

# ============================================================================
# TRADING PARAMETERS (Hard-coded)
# ============================================================================

# Top 20 Crypto Futures (sorted by market cap)
TOP_20_COINS: List[str] = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "USDC/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "SHIB/USDT",
    "TON/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "TRX/USDT",
    "BCH/USDT", "NEAR/USDT", "LTC/USDT", "ICP/USDT", "UNI/USDT"
]

# Leverage Configuration
MAX_LEVERAGE: int = 5  # Dynamic cap at 5x
BASE_LEVERAGE: float = 3.0  # Default leverage

# Risk Management (Fractional Kelly Criterion)
KELLY_FRACTION: float = 0.15  # Conservative 15% of full Kelly
CONFIDENCE_THRESHOLD: float = 0.65  # Lowered from 0.90 for ~20 trades/day

# Stop Loss & Dynamic Trailing Stop (Chandelier Exit)
INITIAL_SL_ATR_MULTIPLIER: float = 1.5  # Initial SL: Entry ± (ATR * 1.5)
TRAILING_STOP_ATR_MULTIPLIER: float = 2.5  # Trailing distance: High/Low ± (ATR * 2.5)
ATR_PERIOD: int = 14  # ATR calculation period

# Profit Locking Thresholds
BREAKEVEN_TRIGGER_PCT: float = 1.5  # Move SL to breakeven at 1.5% profit
AGGRESSIVE_TRAILING_TRIGGER_PCT: float = 5.0  # Tighten trailing at 5.0% profit
AGGRESSIVE_TRAILING_MULTIPLIER: float = 1.5  # Tightened trailing multiplier

# Trading Fees
TAKER_FEE: float = 0.0006  # 0.06% taker fee

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# TFT Oracle Parameters
TFT_PREDICTION_HORIZON: int = 12  # Predict next 12 candles
TFT_MAX_ENCODER_LENGTH: int = 60  # Look back 60 candles
TFT_MAX_DECODER_LENGTH: int = 12
TFT_HIDDEN_SIZE: int = 64
TFT_ATTENTION_HEAD_SIZE: int = 4
TFT_DROPOUT: float = 0.1

# PPO Commander Parameters
PPO_HIDDEN_SIZE: int = 256
PPO_LEARNING_RATE: float = 3e-4
PPO_N_STEPS: int = 2048
PPO_BATCH_SIZE: int = 64
PPO_N_EPOCHS: int = 10
PPO_GAMMA: float = 0.99
PPO_GAE_LAMBDA: float = 0.95
PPO_CLIP_RANGE: float = 0.2
PPO_ENT_COEF: float = 0.01
PPO_VF_COEF: float = 0.5
PPO_TOTAL_TIMESTEPS: int = 5000000  # 5M steps for high-frequency training

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Timeframes
SCALP_TIMEFRAME: str = "15m"  # 15-minute candles
SWING_TIMEFRAME: str = "4h"  # 4-hour candles

# Data Fetching
DATA_FETCH_INTERVAL_SECONDS: int = 60  # Fetch new data every minute
API_RATE_LIMIT_DELAY: float = 0.1  # Sleep between requests (100ms)

# Context-Aware Features (CRITICAL)
CONTEXT_FEATURES: List[str] = [
    "BTC_Close",
    "BTC_RSI",
    "USDT_Dominance"
]

# Technical Indicators
TECHNICAL_INDICATORS: List[str] = [
    "RSI",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "ATR",
    "BB_upper",
    "BB_middle",
    "BB_lower",
    "Volume",
    "Volume_MA"
]

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Continuous Learning
RETRAIN_DAY: str = "sunday"  # Weekly retraining
RETRAIN_HOUR: int = 3  # 03:00 UTC
RETRAIN_MINUTE: int = 0

FINE_TUNE_EPOCHS: int = 5
FINE_TUNE_LEARNING_RATE: float = 1e-5
RETRAIN_LOOKBACK_DAYS: int = 7

# Optuna Optimization
OPTUNA_N_TRIALS: int = 50
OPTUNA_TIMEOUT_SECONDS: int = 3600  # 1 hour

# ============================================================================
# ENVIRONMENT PARAMETERS
# ============================================================================

INITIAL_BALANCE: float = 10000.0  # Starting capital (USDT)
MIN_TRADE_SIZE: float = 10.0  # Minimum trade size in USDT
MAX_POSITION_SIZE: float = 0.3  # Max 30% of portfolio per position

# ============================================================================
# EXCHANGE CONFIGURATION
# ============================================================================

EXCHANGE_NAME: str = "binance"  # Using Binance Futures
EXCHANGE_API_KEY: str = ""  # Set via environment variable
EXCHANGE_SECRET: str = ""  # Set via environment variable
EXCHANGE_SANDBOX: bool = True  # Start in sandbox mode

# ============================================================================
# LOGGING & MONITORING
# ============================================================================

LOG_LEVEL: str = "INFO"
LOG_FILE: str = "logs/trading_bot.log"
DASHBOARD_REFRESH_RATE: float = 1.0  # Update dashboard every second

# ============================================================================
# DATACLASS CONFIG
# ============================================================================

@dataclass
class TradingConfig:
    """Centralized configuration dataclass."""
    top_20_coins: List[str] = None
    max_leverage: int = None
    kelly_fraction: float = None
    confidence_threshold: float = None
    initial_sl_atr_multiplier: float = None
    trailing_stop_atr_multiplier: float = None
    atr_period: int = None
    taker_fee: float = None
    initial_balance: float = None
    
    def __post_init__(self):
        if self.top_20_coins is None:
            self.top_20_coins = TOP_20_COINS
        if self.max_leverage is None:
            self.max_leverage = MAX_LEVERAGE
        if self.kelly_fraction is None:
            self.kelly_fraction = KELLY_FRACTION
        if self.confidence_threshold is None:
            self.confidence_threshold = CONFIDENCE_THRESHOLD
        if self.initial_sl_atr_multiplier is None:
            self.initial_sl_atr_multiplier = INITIAL_SL_ATR_MULTIPLIER
        if self.trailing_stop_atr_multiplier is None:
            self.trailing_stop_atr_multiplier = TRAILING_STOP_ATR_MULTIPLIER
        if self.atr_period is None:
            self.atr_period = ATR_PERIOD
        if self.taker_fee is None:
            self.taker_fee = TAKER_FEE
        if self.initial_balance is None:
            self.initial_balance = INITIAL_BALANCE

# Global config instance
config = TradingConfig()

