"""
TFT Ensemble Model
Manages 3 TFT models for different timeframes (15m, 1h, 4h) and combines predictions.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tft import TFTModel
from config import DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TFTEnsemble:
    """
    Ensemble of 3 TFT models for multi-timeframe analysis:
    - 15m: High-frequency scalping signals
    - 1h: Medium-term trend confirmation
    - 4h: Long-term trend context
    """
    
    def __init__(
        self,
        device: str = DEVICE
    ):
        """Initialize TFT Ensemble with 3 models."""
        self.device = device
        self.timeframes = ['15m', '1h', '4h']
        
        # Initialize 3 TFT models (one per timeframe)
        self.models: Dict[str, TFTModel] = {}
        for tf in self.timeframes:
            self.models[tf] = TFTModel(device=device)
            logger.info(f"Initialized TFT model for {tf} timeframe")
        
        logger.info(f"TFT Ensemble initialized with {len(self.models)} models on {device}")
    
    def train_model(
        self,
        timeframe: str,
        data: Dict[str, pd.DataFrame],
        epochs: int = 20,
        batch_size: int = 64
    ):
        """Train a specific TFT model for a timeframe."""
        if timeframe not in self.models:
            raise ValueError(f"Unknown timeframe: {timeframe}. Must be one of {self.timeframes}")
        
        logger.info(f"Training TFT model for {timeframe} timeframe...")
        model = self.models[timeframe]
        
        # Create dataset
        training_data = model.create_dataset(data)
        
        # Build model
        model.build_model(training_data)
        
        # Train
        history = model.train(
            training_data=training_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        logger.info(f"TFT model for {timeframe} trained successfully")
        return history
    
    def train_all(
        self,
        data_15m: Dict[str, pd.DataFrame],
        data_1h: Dict[str, pd.DataFrame],
        data_4h: Dict[str, pd.DataFrame],
        epochs: int = 20,
        batch_size: int = 64
    ):
        """Train all 3 TFT models."""
        logger.info("=" * 60)
        logger.info("TRAINING TFT ENSEMBLE (3 MODELS)")
        logger.info("=" * 60)
        
        # Train each model
        self.train_model('15m', data_15m, epochs, batch_size)
        self.train_model('1h', data_1h, epochs, batch_size)
        self.train_model('4h', data_4h, epochs, batch_size)
        
        logger.info("=" * 60)
        logger.info("TFT ENSEMBLE TRAINING COMPLETED")
        logger.info("=" * 60)
    
    def predict_ensemble(
        self,
        data_15m: pd.DataFrame,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        coin: str
    ) -> Tuple[np.ndarray, float, Dict[str, float]]:
        """
        Get ensemble prediction from all 3 TFT models.
        
        Returns:
            ensemble_predictions: Combined prediction (12 candles)
            ensemble_confidence: Weighted average confidence
            individual_confidences: Dict of confidence scores per timeframe
        """
        predictions = {}
        confidences = {}
        
        # Get predictions from each model
        pred_15m, conf_15m = self.models['15m'].predict(data_15m, coin)
        pred_1h, conf_1h = self.models['1h'].predict(data_1h, coin)
        pred_4h, conf_4h = self.models['4h'].predict(data_4h, coin)
        
        predictions['15m'] = pred_15m
        predictions['1h'] = pred_1h
        predictions['4h'] = pred_4h
        
        confidences['15m'] = conf_15m
        confidences['1h'] = conf_1h
        confidences['4h'] = conf_4h
        
        # Weighted ensemble: Higher weight for shorter timeframes (more responsive)
        weights = {
            '15m': 0.5,  # Highest weight for high-frequency signals
            '1h': 0.3,   # Medium weight for trend confirmation
            '4h': 0.2   # Lower weight for long-term context
        }
        
        # Combine predictions (weighted average)
        # Resample longer timeframes to match 15m prediction length
        ensemble_pred = (
            weights['15m'] * pred_15m +
            weights['1h'] * np.interp(np.linspace(0, len(pred_15m)-1, len(pred_15m)), 
                                     np.linspace(0, len(pred_1h)-1, len(pred_1h)), pred_1h) +
            weights['4h'] * np.interp(np.linspace(0, len(pred_15m)-1, len(pred_15m)),
                                     np.linspace(0, len(pred_4h)-1, len(pred_4h)), pred_4h)
        )
        
        # Weighted average confidence
        ensemble_confidence = (
            weights['15m'] * conf_15m +
            weights['1h'] * conf_1h +
            weights['4h'] * conf_4h
        )
        
        return ensemble_pred, ensemble_confidence, confidences
    
    def predict(
        self,
        data_15m: pd.DataFrame,
        data_1h: pd.DataFrame,
        data_4h: pd.DataFrame,
        coin: str
    ) -> Tuple[np.ndarray, float]:
        """
        Simplified predict interface (returns ensemble prediction and confidence).
        """
        ensemble_pred, ensemble_conf, _ = self.predict_ensemble(data_15m, data_1h, data_4h, coin)
        return ensemble_pred, ensemble_conf
    
    def save(self, base_path: str):
        """Save all 3 TFT models."""
        for tf in self.timeframes:
            model_path = f"{base_path}_tft_{tf}.pt"
            self.models[tf].save(model_path)
            logger.info(f"Saved TFT model for {tf} to {model_path}")
    
    def load(self, base_path: str, training_data_dict: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None):
        """
        Load all 3 TFT models.
        
        Args:
            training_data_dict: Optional dict mapping timeframes to data dicts for model reconstruction
        """
        for tf in self.timeframes:
            model_path = f"{base_path}_tft_{tf}.pt"
            if training_data_dict and tf in training_data_dict:
                # Rebuild model with training data
                data = training_data_dict[tf]
                training_data = self.models[tf].create_dataset(data)
                self.models[tf].load(model_path, training_data)
            else:
                logger.warning(f"Cannot fully load {tf} model without training data. Model structure may be incomplete.")
            logger.info(f"Loaded TFT model for {tf} from {model_path}")


if __name__ == "__main__":
    # Test TFT Ensemble
    logger.info("Testing TFT Ensemble...")
    ensemble = TFTEnsemble()
    logger.info("TFT Ensemble created successfully")

