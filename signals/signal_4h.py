"""
4h Signal Generator - Loads and uses 4h TFT model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.signal_15m import Signal15m
import logging

logger = logging.getLogger(__name__)

class Signal4h(Signal15m):
    """4h timeframe signal generator (inherits from 15m structure)."""
    
    def load_model(self, model_path: str):
        """Load trained 4h TFT model."""
        super().load_model(model_path)
        logger.info(f"Loaded 4h model from {model_path}")





