"""
Signal Base Interface - Base class for signal generators
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from pathlib import Path

class SignalBase(ABC):
    """Base interface for signal generators."""
    
    @abstractmethod
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        pass
    
    @abstractmethod
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of price movement.
        
        Returns:
            Array of probabilities (shape: [n_samples, 2]) for [down, up]
        """
        pass
    
    @abstractmethod
    def to_signal(
        self,
        prob: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Convert probabilities to trading signals.
        
        Args:
            prob: Probability array [n_samples, 2] for [down, up]
            thresholds: Dict with 'long_threshold' and 'short_threshold'
        
        Returns:
            Array of signals: 1 (LONG), -1 (SHORT), 0 (FLAT)
        """
        pass
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        thresholds: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generate complete signal DataFrame.
        
        Returns:
            DataFrame with columns: timestamp, prob_up, prob_down, confidence, signal
        """
        prob = self.predict_proba(features)
        signals = self.to_signal(prob, thresholds)
        
        prob_up = prob[:, 1] if prob.shape[1] > 1 else prob.flatten()
        prob_down = prob[:, 0] if prob.shape[1] > 1 else 1 - prob_up
        confidence = np.abs(prob_up - prob_down)
        
        result = pd.DataFrame({
            'timestamp': features['timestamp'].values if 'timestamp' in features.columns else features.index,
            'prob_up': prob_up,
            'prob_down': prob_down,
            'confidence': confidence,
            'signal': signals,
        })
        
        return result




