"""Configuration package - Exports from root config.py"""

import sys
import os
import ast

# Get parent directory (project root)
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_config_path = os.path.join(_parent_dir, 'config.py')

# Parse config.py as AST to extract values without executing torch imports
if os.path.exists(_config_path):
    with open(_config_path, 'r', encoding='utf-8') as f:
        _config_code = f.read()
    
    # Parse and extract variable assignments
    _tree = ast.parse(_config_code)
    _config_vars = {}
    
    for node in ast.walk(_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        # Try to evaluate simple literals
                        if isinstance(node.value, ast.Constant):
                            # Python 3.8+ - Constant node
                            _config_vars[target.id] = node.value.value
                        elif isinstance(node.value, ast.Str):
                            # Python < 3.8 - String literal
                            _config_vars[target.id] = node.value.s
                        elif isinstance(node.value, ast.Num):
                            # Python < 3.8 - Number literal
                            _config_vars[target.id] = node.value.n
                        elif isinstance(node.value, ast.NameConstant):
                            # Python < 3.8 - True/False/None
                            _config_vars[target.id] = node.value.value
                        elif isinstance(node.value, (ast.List, ast.Tuple)):
                            # List or tuple - use literal_eval on the source
                            _config_vars[target.id] = ast.literal_eval(node.value)
                    except:
                        pass
    
    # Set defaults from config.py values (hardcoded to avoid torch dependency)
    TFT_PREDICTION_HORIZON = _config_vars.get('TFT_PREDICTION_HORIZON', 12)
    TFT_MAX_ENCODER_LENGTH = _config_vars.get('TFT_MAX_ENCODER_LENGTH', 60)
    TFT_MAX_DECODER_LENGTH = _config_vars.get('TFT_MAX_DECODER_LENGTH', 12)
    TFT_HIDDEN_SIZE = _config_vars.get('TFT_HIDDEN_SIZE', 128)
    TFT_ATTENTION_HEAD_SIZE = _config_vars.get('TFT_ATTENTION_HEAD_SIZE', 8)
    TFT_DROPOUT = _config_vars.get('TFT_DROPOUT', 0.1)
    
    # Features (defaults)
    CONTEXT_FEATURES = _config_vars.get('CONTEXT_FEATURES', [
        "BTC_Close",
        "BTC_RSI",
        "USDT_Dominance"
    ])
    TECHNICAL_INDICATORS = _config_vars.get('TECHNICAL_INDICATORS', [
        "RSI", "MACD", "MACD_signal", "MACD_hist",
        "ATR", "BB_upper", "BB_middle", "BB_lower",
        "Volume", "Volume_MA"
    ])
    
    # Trading parameters
    TOP_20_COINS = _config_vars.get('TOP_20_COINS', [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "USDC/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "SHIB/USDT",
        "TON/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "TRX/USDT",
        "BCH/USDT", "NEAR/USDT", "LTC/USDT", "ICP/USDT", "UNI/USDT"
    ])
    SCALP_TIMEFRAME = _config_vars.get('SCALP_TIMEFRAME', "15m")
    SWING_TIMEFRAME = _config_vars.get('SWING_TIMEFRAME', "4h")
    
    # Exchange configuration
    API_RATE_LIMIT_DELAY = _config_vars.get('API_RATE_LIMIT_DELAY', 0.1)
    EXCHANGE_NAME = _config_vars.get('EXCHANGE_NAME', "binance")
    EXCHANGE_API_KEY = _config_vars.get('EXCHANGE_API_KEY', "")
    EXCHANGE_SECRET = _config_vars.get('EXCHANGE_SECRET', "")
    EXCHANGE_SANDBOX = _config_vars.get('EXCHANGE_SANDBOX', True)
    
    # Trading parameters
    INITIAL_BALANCE = _config_vars.get('INITIAL_BALANCE', 10000.0)
    MAX_LEVERAGE = _config_vars.get('MAX_LEVERAGE', 5)
    BASE_LEVERAGE = _config_vars.get('BASE_LEVERAGE', 3.0)
    KELLY_FRACTION = _config_vars.get('KELLY_FRACTION', 0.15)
    CONFIDENCE_THRESHOLD = _config_vars.get('CONFIDENCE_THRESHOLD', 0.65)
    INITIAL_SL_ATR_MULTIPLIER = _config_vars.get('INITIAL_SL_ATR_MULTIPLIER', 1.5)
    TRAILING_STOP_ATR_MULTIPLIER = _config_vars.get('TRAILING_STOP_ATR_MULTIPLIER', 2.5)
    TAKE_PROFIT_ATR_MULTIPLIER = _config_vars.get('TAKE_PROFIT_ATR_MULTIPLIER', 3.0)
    ATR_PERIOD = _config_vars.get('ATR_PERIOD', 14)
    BREAKEVEN_TRIGGER_PCT = _config_vars.get('BREAKEVEN_TRIGGER_PCT', 1.5)
    AGGRESSIVE_TRAILING_TRIGGER_PCT = _config_vars.get('AGGRESSIVE_TRAILING_TRIGGER_PCT', 5.0)
    AGGRESSIVE_TRAILING_MULTIPLIER = _config_vars.get('AGGRESSIVE_TRAILING_MULTIPLIER', 1.5)
    TAKER_FEE = _config_vars.get('TAKER_FEE', 0.0004)
    SLIPPAGE_PCT = _config_vars.get('SLIPPAGE_PCT', 0.0005)
    SPREAD_PCT = _config_vars.get('SPREAD_PCT', 0.0002)
    MIN_TRADE_SIZE = _config_vars.get('MIN_TRADE_SIZE', 10.0)
    MAX_POSITION_SIZE = _config_vars.get('MAX_POSITION_SIZE', 0.3)
    MAX_DRAWDOWN_PER_TRADE = _config_vars.get('MAX_DRAWDOWN_PER_TRADE', 0.02)
    TAKE_PROFIT_THRESHOLD = _config_vars.get('TAKE_PROFIT_THRESHOLD', 0.05)
    MAX_HOLD_BARS = _config_vars.get('MAX_HOLD_BARS', 200)
    MAX_PORTFOLIO_DRAWDOWN = _config_vars.get('MAX_PORTFOLIO_DRAWDOWN', 0.15)
    MIN_PORTFOLIO_VALUE = _config_vars.get('MIN_PORTFOLIO_VALUE', 0.5)
    LONG_ACTION_THRESHOLD = _config_vars.get('LONG_ACTION_THRESHOLD', 0.3)
    SHORT_ACTION_THRESHOLD = _config_vars.get('SHORT_ACTION_THRESHOLD', -0.3)
    
    # Workers (default - calculated value, use 8 for Windows)
    # OPTIMAL_NUM_WORKERS is calculated in config.py, default to 8
    import multiprocessing
    _cpu_count = multiprocessing.cpu_count()
    OPTIMAL_NUM_WORKERS = min(_cpu_count, 8)  # Windows safe default
    DEVICE = 'cuda'
else:
    # Fallback defaults if config.py not found
    TFT_PREDICTION_HORIZON = 12
    TFT_MAX_ENCODER_LENGTH = 60
    TFT_MAX_DECODER_LENGTH = 12
    TFT_HIDDEN_SIZE = 128
    TFT_ATTENTION_HEAD_SIZE = 8
    TFT_DROPOUT = 0.1
    CONTEXT_FEATURES = ["BTC_Close", "BTC_RSI", "USDT_Dominance"]
    TECHNICAL_INDICATORS = ["RSI", "MACD", "MACD_signal", "MACD_hist", "ATR", 
                           "BB_upper", "BB_middle", "BB_lower", "Volume", "Volume_MA"]
    TOP_20_COINS = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "USDC/USDT", "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "SHIB/USDT",
        "TON/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "TRX/USDT",
        "BCH/USDT", "NEAR/USDT", "LTC/USDT", "ICP/USDT", "UNI/USDT"
    ]
    SCALP_TIMEFRAME = "15m"
    SWING_TIMEFRAME = "4h"
    API_RATE_LIMIT_DELAY = 0.1
    EXCHANGE_NAME = "binance"
    EXCHANGE_API_KEY = ""
    EXCHANGE_SECRET = ""
    EXCHANGE_SANDBOX = True
    INITIAL_BALANCE = 10000.0
    MAX_LEVERAGE = 5
    BASE_LEVERAGE = 3.0
    KELLY_FRACTION = 0.15
    CONFIDENCE_THRESHOLD = 0.65
    INITIAL_SL_ATR_MULTIPLIER = 1.5
    TRAILING_STOP_ATR_MULTIPLIER = 2.5
    TAKE_PROFIT_ATR_MULTIPLIER = 3.0
    ATR_PERIOD = 14
    BREAKEVEN_TRIGGER_PCT = 1.5
    AGGRESSIVE_TRAILING_TRIGGER_PCT = 5.0
    AGGRESSIVE_TRAILING_MULTIPLIER = 1.5
    TAKER_FEE = 0.0004
    SLIPPAGE_PCT = 0.0005
    SPREAD_PCT = 0.0002
    MIN_TRADE_SIZE = 10.0
    MAX_POSITION_SIZE = 0.3
    MAX_DRAWDOWN_PER_TRADE = 0.02
    TAKE_PROFIT_THRESHOLD = 0.05
    MAX_HOLD_BARS = 200
    MAX_PORTFOLIO_DRAWDOWN = 0.15
    MIN_PORTFOLIO_VALUE = 0.5
    LONG_ACTION_THRESHOLD = 0.3
    SHORT_ACTION_THRESHOLD = -0.3
    OPTIMAL_NUM_WORKERS = 8
    DEVICE = 'cuda'

__all__ = [
    'TFT_PREDICTION_HORIZON',
    'TFT_MAX_ENCODER_LENGTH',
    'TFT_MAX_DECODER_LENGTH',
    'TFT_HIDDEN_SIZE',
    'TFT_ATTENTION_HEAD_SIZE',
    'TFT_DROPOUT',
    'CONTEXT_FEATURES',
    'TECHNICAL_INDICATORS',
    'TOP_20_COINS',
    'SCALP_TIMEFRAME',
    'SWING_TIMEFRAME',
    'API_RATE_LIMIT_DELAY',
    'EXCHANGE_NAME',
    'EXCHANGE_API_KEY',
    'EXCHANGE_SECRET',
    'EXCHANGE_SANDBOX',
    'INITIAL_BALANCE',
    'MAX_LEVERAGE',
    'BASE_LEVERAGE',
    'KELLY_FRACTION',
    'CONFIDENCE_THRESHOLD',
    'INITIAL_SL_ATR_MULTIPLIER',
    'TRAILING_STOP_ATR_MULTIPLIER',
    'TAKE_PROFIT_ATR_MULTIPLIER',
    'ATR_PERIOD',
    'BREAKEVEN_TRIGGER_PCT',
    'AGGRESSIVE_TRAILING_TRIGGER_PCT',
    'AGGRESSIVE_TRAILING_MULTIPLIER',
    'TAKER_FEE',
    'SLIPPAGE_PCT',
    'SPREAD_PCT',
    'MIN_TRADE_SIZE',
    'MAX_POSITION_SIZE',
    'MAX_DRAWDOWN_PER_TRADE',
    'TAKE_PROFIT_THRESHOLD',
    'MAX_HOLD_BARS',
    'MAX_PORTFOLIO_DRAWDOWN',
    'MIN_PORTFOLIO_VALUE',
    'LONG_ACTION_THRESHOLD',
    'SHORT_ACTION_THRESHOLD',
    'OPTIMAL_NUM_WORKERS',
    'DEVICE',
]
