"""
15m Signal Generator - Loads and uses 15m TFT model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

from signals.signal_base import SignalBase
from models.tft import TFTModel
from pytorch_forecasting import TimeSeriesDataSet

logger = logging.getLogger(__name__)

class Signal15m(SignalBase):
    """15m timeframe signal generator."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize 15m signal generator."""
        self.model = None
        self.model_path = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained 15m TFT model."""
        self.model_path = model_path
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model (simplified - in production would need full config)
        self.model = TFTModel(
            prediction_horizon=12,
            max_encoder_length=60,
            max_decoder_length=12,
            device=self.device
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's the state dict directly
            self.model.model.load_state_dict(checkpoint)
        
        self.model.model.eval()
        logger.info(f"Loaded 15m model from {model_path}")
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probability using 15m model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Prepare data for TFT
        # This is simplified - in production would need proper TimeSeriesDataSet creation
        with torch.no_grad():
            # For now, return dummy probabilities
            # In production, would run through TFT model
            n_samples = len(features)
            prob = np.random.rand(n_samples, 2)
            prob = prob / prob.sum(axis=1, keepdims=True)  # Normalize
        
        return prob
    
    def to_signal(
        self,
        prob: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Convert probabilities to signals."""
        if thresholds is None:
            thresholds = {
                'long_threshold': 0.6,
                'short_threshold': 0.4,
            }
        
        prob_up = prob[:, 1] if prob.shape[1] > 1 else prob.flatten()
        
        signals = np.zeros(len(prob_up))
        signals[prob_up > thresholds['long_threshold']] = 1  # LONG
        signals[prob_up < thresholds['short_threshold']] = -1  # SHORT
        
        return signals




