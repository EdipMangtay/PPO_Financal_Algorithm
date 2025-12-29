"""
1h Signal Generator - Loads and uses 1h TFT model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.signal_15m import Signal15m
import logging

logger = logging.getLogger(__name__)

class Signal1h(Signal15m):
    """1h timeframe signal generator (inherits from 15m structure)."""
    
    def load_model(self, model_path: str):
        """Load trained 1h TFT model."""
        super().load_model(model_path)
        logger.info(f"Loaded 1h model from {model_path}")




