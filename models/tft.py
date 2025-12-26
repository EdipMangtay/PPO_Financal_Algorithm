"""
Temporal Fusion Transformer (TFT) Oracle Model
Predicts price probability for next 12 candles with context-aware features.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TFT_PREDICTION_HORIZON,
    TFT_MAX_ENCODER_LENGTH,
    TFT_MAX_DECODER_LENGTH,
    TFT_HIDDEN_SIZE,
    TFT_ATTENTION_HEAD_SIZE,
    TFT_DROPOUT,
    CONTEXT_FEATURES,
    TECHNICAL_INDICATORS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force float32 matmul precision for RTX 5070
torch.set_float32_matmul_precision('medium')


class TFTModel:
    """Temporal Fusion Transformer for price prediction with context awareness."""
    
    def __init__(
        self,
        prediction_horizon: int = TFT_PREDICTION_HORIZON,
        max_encoder_length: int = TFT_MAX_ENCODER_LENGTH,
        max_decoder_length: int = TFT_MAX_DECODER_LENGTH,
        hidden_size: int = TFT_HIDDEN_SIZE,
        attention_head_size: int = TFT_ATTENTION_HEAD_SIZE,
        dropout: float = TFT_DROPOUT,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """Initialize TFT model."""
        self.prediction_horizon = prediction_horizon
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device
        
        self.model: Optional[TemporalFusionTransformer] = None
        self.training_data: Optional[TimeSeriesDataSet] = None
        self.static_categoricals: List[str] = ['coin']
        self.time_varying_known_reals: List[str] = []
        self.time_varying_unknown_reals: List[str] = []
        
        logger.info(f"TFT Model initialized on {device}")
    
    def _prepare_dataframe(
        self, 
        data_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Prepare multi-coin dataframe for TFT.
        Each coin becomes a group with static categorical 'coin'.
        """
        dfs = []
        for coin, df in data_dict.items():
            df_copy = df.copy()
            df_copy['coin'] = coin
            df_copy['time_idx'] = range(len(df_copy))
            dfs.append(df_copy)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Reset time_idx per coin
        combined_df['time_idx'] = combined_df.groupby('coin').cumcount()
        
        return combined_df
    
    def _identify_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify time-varying known and unknown real features."""
        # Known features (available in decoder): time-based features
        known_reals = ['time_idx']
        
        # Unknown features (only in encoder): price, volume, indicators, context
        unknown_reals = [
            'open', 'high', 'low', 'close', 'volume',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'ATR', 'BB_upper', 'BB_middle', 'BB_lower',
            'Volume_MA'
        ]
        
        # Add context features (BTC, USDT dominance)
        unknown_reals.extend(CONTEXT_FEATURES)
        
        # Filter to only include columns that exist
        known_reals = [f for f in known_reals if f in df.columns]
        unknown_reals = [f for f in unknown_reals if f in df.columns]
        
        return known_reals, unknown_reals
    
    def create_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        target: str = 'close'
    ) -> TimeSeriesDataSet:
        """Create TimeSeriesDataSet from coin data dictionary."""
        logger.info("Preparing TFT dataset...")
        
        # Combine all coins into single dataframe
        df = self._prepare_dataframe(data_dict)
        
        # Identify features
        known_reals, unknown_reals = self._identify_features(df)
        
        self.time_varying_known_reals = known_reals
        self.time_varying_unknown_reals = unknown_reals
        
        logger.info(f"Known reals: {known_reals}")
        logger.info(f"Unknown reals: {unknown_reals[:5]}... (showing first 5)")
        
        # Create TimeSeriesDataSet
        training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=target,
            group_ids=["coin"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            # FIX: Renamed max_decoder_length to max_prediction_length for pytorch-forecasting compatibility
            max_prediction_length=self.max_decoder_length,
            static_categoricals=self.static_categoricals,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["coin"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        self.training_data = training
        logger.info(f"Dataset created with {len(training)} samples")
        
        return training
    
    def build_model(self, training_data: Optional[TimeSeriesDataSet] = None) -> TemporalFusionTransformer:
        """Build TFT model architecture."""
        if training_data is None:
            training_data = self.training_data
        
        if training_data is None:
            raise ValueError("Must provide training_data or call create_dataset first")
        
        logger.info("Building TFT model...")
        
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size,
            output_size=7,  # 7 quantiles for probabilistic prediction
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        self.model.to(self.device)
        logger.info(f"TFT model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        return self.model
    
    def train(
        self,
        training_data: Optional[TimeSeriesDataSet] = None,
        validation_data: Optional[TimeSeriesDataSet] = None,
        epochs: int = 10,
        batch_size: int = 64,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Train the TFT model."""
        if self.model is None:
            self.build_model(training_data)
        
        if training_data is None:
            training_data = self.training_data
        
        if training_data is None:
            raise ValueError("Must provide training_data")
        
        # Create dataloaders
        train_dataloader = training_data.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        
        if validation_data is not None:
            val_dataloader = validation_data.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
        else:
            val_dataloader = None
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in train_dataloader:
                # CRITICAL FIX: Handle tuple unpacking correctly
                x, y = batch
                # x is a dictionary of tensors, move to device
                x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                
                # Handle target y (it might be a tuple, list, or tensor)
                if isinstance(y, tuple) or isinstance(y, list):
                    y = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in y)
                else:
                    y = y.to(self.device)
                
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.model.loss(y_hat, y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0.0
            history['train_loss'].append(avg_loss)
            
            # Validation
            if val_dataloader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        # CRITICAL FIX: Handle tuple unpacking correctly
                        x, y = batch
                        # x is a dictionary of tensors, move to device
                        x = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                        
                        # Handle target y (it might be a tuple, list, or tensor)
                        if isinstance(y, tuple) or isinstance(y, list):
                            y = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in y)
                        else:
                            y = y.to(self.device)
                        
                        y_hat = self.model(x)
                        loss = self.model.loss(y_hat, y)
                        val_loss += loss.item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
                history['val_loss'].append(avg_val_loss)
                self.model.train()
                
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                if verbose:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
        return history
    
    def predict(
        self,
        data: pd.DataFrame,
        coin: str,
        return_index: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Predict next 12 candles for a coin.
        
        Returns:
            predictions: Array of predicted prices (12 candles)
            confidence: Confidence score (0-1) based on prediction variance
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.model.eval()
        
        # Prepare data
        df = data.copy()
        df['coin'] = coin
        df['time_idx'] = range(len(df))
        
        # Create dataset for prediction
        known_reals, unknown_reals = self._identify_features(df)
        
        pred_dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="close",
            group_ids=["coin"],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            # FIX: Renamed max_decoder_length to max_prediction_length for pytorch-forecasting compatibility
            max_prediction_length=self.max_decoder_length,
            static_categoricals=self.static_categoricals,
            time_varying_known_reals=known_reals,
            time_varying_unknown_reals=unknown_reals,
            target_normalizer=GroupNormalizer(groups=["coin"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Get last sequence
        x, _ = pred_dataset[0]
        x = {k: v.unsqueeze(0).to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        
        # Predict
        with torch.no_grad():
            predictions = self.model(x)
        
        # Extract median prediction (quantile 0.5)
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Get predictions for next 12 candles
        pred_values = predictions[0, :, 3].cpu().numpy()  # Index 3 is median (0.5 quantile)
        
        # Calculate confidence based on prediction variance
        if predictions.shape[-1] > 1:
            # Use inter-quantile range as uncertainty measure
            q25 = predictions[0, :, 1].cpu().numpy()  # 0.25 quantile
            q75 = predictions[0, :, 5].cpu().numpy()  # 0.75 quantile
            uncertainty = np.mean(np.abs(q75 - q25))
            confidence = 1.0 / (1.0 + uncertainty / np.mean(pred_values))
            confidence = np.clip(confidence, 0.0, 1.0)
        else:
            confidence = 0.5
        
        if return_index:
            return pred_values, confidence, df.index[-self.max_decoder_length:]
        
        return pred_values, confidence
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'prediction_horizon': self.prediction_horizon,
                'max_encoder_length': self.max_encoder_length,
                'max_decoder_length': self.max_decoder_length,
                'hidden_size': self.hidden_size,
                'attention_head_size': self.attention_head_size,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
            }
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str, training_data: Optional[TimeSeriesDataSet] = None):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint['config']
        
        # Update config
        self.prediction_horizon = config['prediction_horizon']
        self.max_encoder_length = config['max_encoder_length']
        self.max_decoder_length = config['max_decoder_length']
        self.hidden_size = config['hidden_size']
        self.attention_head_size = config['attention_head_size']
        self.dropout = config['dropout']
        self.learning_rate = config['learning_rate']
        
        # Rebuild model
        if training_data is not None:
            self.training_data = training_data
            self.build_model(training_data)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning("Cannot load model without training_data. Call create_dataset first.")


if __name__ == "__main__":
    # Test TFT model
    logger.info("Testing TFT model...")
    model = TFTModel()
    logger.info("TFT model created successfully")

