"""
GOD_LEVEL_TRADER_V5 - Deep Optimization Configuration
RTX 5070 Full Power - Production Trading System
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
TAKE_PROFIT_ATR_MULTIPLIER: float = 3.0  # Take profit: Entry ± (ATR * 3.0)
ATR_PERIOD: int = 14  # ATR calculation period

# Profit Locking Thresholds
BREAKEVEN_TRIGGER_PCT: float = 1.5  # Move SL to breakeven at 1.5% profit
AGGRESSIVE_TRAILING_TRIGGER_PCT: float = 5.0  # Tighten trailing at 5.0% profit
AGGRESSIVE_TRAILING_MULTIPLIER: float = 1.5  # Tightened trailing multiplier

# Trading Fees & Execution Costs (Binance USDT-M Futures Default)
TAKER_FEE: float = 0.0004  # 0.04% taker fee (Binance USDT-M default)
MAKER_FEE: float = 0.0002  # 0.02% maker fee (Binance USDT-M default)
TRANSACTION_FEE: float = 0.0004  # Default to taker
SLIPPAGE_PCT: float = 0.0005  # 0.05% slippage per trade (realistic for crypto)
SPREAD_PCT: float = 0.0002  # 0.02% bid-ask spread (typical for major pairs)

# Funding Rate (default 0, only charge if explicitly enabled)
FUNDING_RATE: float = 0.0  # Default: no funding charges

# ============================================================================
# PRODUCTION RISK ENGINE PARAMETERS (CRITICAL - HARD STOPS)
# ============================================================================

# Hard Stop Loss per Trade (CRITICAL: Overrides agent decisions)
MAX_DRAWDOWN_PER_TRADE: float = 0.02  # 2% max loss per trade (hard stop)
TAKE_PROFIT_THRESHOLD: float = 0.05  # 5% take profit (hard stop)

# Maximum Position Hold Time (Time Stop)
MAX_HOLD_BARS: int = 200  # Maximum bars to hold a position (force close)

# Portfolio-Level Risk Limits
MAX_PORTFOLIO_DRAWDOWN: float = 0.15  # 15% max portfolio drawdown (episode termination)
MIN_PORTFOLIO_VALUE: float = 0.5  # 50% of initial balance (hard liquidation)

# Position Size Limits
MIN_TRADE_SIZE: float = 10.0  # Minimum trade size in USDT
MAX_POSITION_SIZE: float = 0.3  # Max 30% of portfolio per position

# ============================================================================
# BI-DIRECTIONAL TRADING PARAMETERS
# ============================================================================

# Action Thresholds for Bi-Directional Trading
LONG_ACTION_THRESHOLD: float = 0.3  # Action > 0.3 -> LONG
SHORT_ACTION_THRESHOLD: float = -0.3  # Action < -0.3 -> SHORT
# -0.3 <= Action <= 0.3 -> FLAT (Cash)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# TFT Oracle Parameters - RTX 5070 Optimized
TFT_PREDICTION_HORIZON: int = 12  # Predict next 12 candles
TFT_MAX_ENCODER_LENGTH: int = 60  # Look back 60 candles
TFT_MAX_DECODER_LENGTH: int = 12
TFT_HIDDEN_SIZE: int = 128  # RTX 5070: 64 → 128 (2x güç)
TFT_ATTENTION_HEAD_SIZE: int = 8  # RTX 5070: 4 → 8 (2x attention)
TFT_DROPOUT: float = 0.1

# PPO Commander Parameters - RTX 5070 Optimized
PPO_HIDDEN_SIZE: int = 512  # RTX 5070: 256 → 512 (2x güç)
PPO_LEARNING_RATE: float = 3e-4
PPO_N_STEPS: int = 4096  # RTX 5070: 2048 → 4096 (daha fazla data)
PPO_BATCH_SIZE: int = 256  # RTX 5070: 64 → 256 (4x batch - GPU'yu doldurur)
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

# ============================================================================
# DEEP OPTIMIZATION PARAMETERS (RTX 5070 Full Power)
# ============================================================================

# Optuna Optimization - Deep Search
OPTUNA_N_TRIALS: int = 100  # Deep search: 100 trials (increased from 50)
N_TRIALS: int = 100  # Alias for clarity
OPTUNA_TIMEOUT_SECONDS: int = 14400  # 4 hours (for 100 trials with 10k steps)
OPTUNA_TRAIN_VAL_SPLIT: float = 0.7  # 70% train, 30% validation (walk-forward)
OPTUNA_WALK_FORWARD_WINDOWS: int = 3  # Number of walk-forward windows

# Backtest Configuration - Deep Search
BACKTEST_STEPS: int = 10000  # 10k steps ≈ 3 months of 15m data (increased from 2000)
MIN_TRADES_FOR_OPTIMIZATION: int = 50  # Filter out lazy strategies aggressively

# Reproducibility
RANDOM_SEED: int = 42  # Global seed for reproducibility

# ============================================================================
# ENVIRONMENT PARAMETERS
# ============================================================================

INITIAL_BALANCE: float = 10000.0  # Starting capital (USDT)

# ============================================================================
# EXCHANGE CONFIGURATION
# ============================================================================

EXCHANGE_NAME: str = "binance"  # Using Binance Futures
EXCHANGE_API_KEY: str = "UN28uaVerSyRAljui0tJaXZ93B5z3eJy3CySrIbi2bfnufaawwJfXL17GnSnpQKx"
EXCHANGE_SECRET: str = "YhqcPjCLl656z9yeA2EbodxkeN7Emu8IfbalKzCY4rRf19gTKX0JOOa5xAK4mJNe"
EXCHANGE_SANDBOX: bool = True  # Start in sandbox mode

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

import torch
import os
import multiprocessing

# ============================================================================
# CPU OPTIMIZATION - i5 Ultra 245KF (14 Çekirdek) - TAM GÜÇ
# ============================================================================

# CPU çekirdek sayısını otomatik algıla
CPU_COUNT = multiprocessing.cpu_count()  # 14 çekirdek için

# PyTorch thread ayarları (GPU ile birlikte çalışırken 12 thread kullan, 2 çekirdek GPU için bırak)
# GPU yoksa tüm çekirdekleri kullan
PYTORCH_NUM_THREADS = CPU_COUNT - 2 if torch.cuda.is_available() else CPU_COUNT
torch.set_num_threads(PYTORCH_NUM_THREADS)

# OpenMP ve MKL thread ayarları (NumPy, pandas için)
os.environ['OMP_NUM_THREADS'] = str(PYTORCH_NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(PYTORCH_NUM_THREADS)

# DataLoader için optimal worker sayısı (Windows'ta max 8 worker önerilir)
# 14 çekirdek için 8 worker optimal (GPU data loading için yeterli)
OPTIMAL_NUM_WORKERS = min(CPU_COUNT, 8)  # 14 çekirdek → 8 worker

# Optuna paralel çalışma (CPU çekirdeklerinin yarısı kadar paralel trial)
# 14 çekirdek için 4-6 paralel trial optimal
OPTUNA_N_JOBS = min(6, CPU_COUNT // 2)  # 14 çekirdek → 6 paralel trial

# ============================================================================
# RTX 5070 GPU Optimizations
# ============================================================================

if torch.cuda.is_available():
    # Mixed precision için optimizasyonlar
    torch.backends.cudnn.benchmark = True  # Daha hızlı convolution
    torch.backends.cudnn.deterministic = False  # Hız için
    # GPU bellek optimizasyonu
    torch.cuda.empty_cache()  # Cache'i temizle

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # Force CUDA for RTX 5070

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
