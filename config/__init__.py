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
                        if isinstance(node.value, (ast.Constant, ast.Num, ast.Str)):
                            _config_vars[target.id] = ast.literal_eval(node.value)
                        elif isinstance(node.value, ast.List):
                            _config_vars[target.id] = ast.literal_eval(node.value)
                        elif isinstance(node.value, ast.Constant):
                            _config_vars[target.id] = node.value.value
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
    'OPTIMAL_NUM_WORKERS',
    'DEVICE',
]
