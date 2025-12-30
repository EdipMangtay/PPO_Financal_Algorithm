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
    TECHNICAL_INDICATORS,
    OPTIMAL_NUM_WORKERS
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
        device: Optional[str] = None
    ):
        """
        Initialize TFT model.
        
        CRITICAL: Properly detects and uses GPU if available.
        """
        self.prediction_horizon = prediction_horizon
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # ====================================================================
        # CRITICAL FIX: GPU Detection and Selection
        # ====================================================================
        if device is None:
            # Auto-detect GPU
            if torch.cuda.is_available():
                self.device = 'cuda'
                # Print GPU diagnostics
                print(f"GPU DIAGNOSTICS:")
                print(f"  - PyTorch version: {torch.__version__}")
                print(f"  - CUDA available: {torch.cuda.is_available()}")
                print(f"  - CUDA device name: {torch.cuda.get_device_name(0)}")
                print(f"  - CUDA device count: {torch.cuda.device_count()}")
                print(f"  - Current device: {torch.cuda.current_device()}")
                if hasattr(torch.cuda, 'get_device_capability'):
                    capability = torch.cuda.get_device_capability(0)
                    print(f"  - Compute capability: {capability[0]}.{capability[1]}")
                print(f"  - Selected accelerator: GPU (cuda:0)")
            else:
                self.device = 'cpu'
                print(f"GPU DIAGNOSTICS:")
                print(f"  - PyTorch version: {torch.__version__}")
                print(f"  - CUDA available: False")
                print(f"  - Selected accelerator: CPU")
        else:
            self.device = device
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning(f"CUDA requested but not available. Falling back to CPU.")
                self.device = 'cpu'
        
        self.model: Optional[TemporalFusionTransformer] = None
        self.training_data: Optional[TimeSeriesDataSet] = None
        self.static_categoricals: List[str] = ['coin']
        self.time_varying_known_reals: List[str] = []
        self.time_varying_unknown_reals: List[str] = []
        
        logger.info(f"TFT Model initialized on {self.device}")
    
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
        """
        Create TimeSeriesDataSet from coin data dictionary.
        
        CRITICAL: Includes comprehensive validation to prevent None/empty dataset errors.
        """
        logger.info("Preparing TFT dataset...")
        
        # ====================================================================
        # VALIDATION 1: Check input data is not empty
        # ====================================================================
        if not data_dict:
            raise ValueError("data_dict is empty. Cannot create dataset.")
        
        # Combine all coins into single dataframe
        df = self._prepare_dataframe(data_dict)
        
        # ====================================================================
        # VALIDATION 2: Check dataframe is not empty
        # ====================================================================
        if df.empty:
            raise ValueError("Combined dataframe is empty after preparation.")
        
        logger.info(f"Combined dataframe shape: {df.shape}")
        
        # ====================================================================
        # VALIDATION 3: Check required columns exist
        # ====================================================================
        required_cols = ['coin', 'time_idx', target]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
        
        # ====================================================================
        # VALIDATION 4: Check for infinite values
        # ====================================================================
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        if inf_cols:
            logger.warning(f"Found infinite values in columns: {inf_cols}. Replacing with NaN.")
            df[inf_cols] = df[inf_cols].replace([np.inf, -np.inf], np.nan)
        
        # ====================================================================
        # VALIDATION 5: Check NaN percentage per column
        # ====================================================================
        nan_pct = df.isnull().sum() / len(df) * 100
        high_nan_cols = nan_pct[nan_pct > 50].index.tolist()
        if high_nan_cols:
            logger.warning(f"Columns with >50% NaN: {high_nan_cols}. Consider dropping or imputing.")
        
        # Drop rows with NaN in critical columns
        rows_before = len(df)
        df = df.dropna(subset=[target, 'coin', 'time_idx'])
        rows_dropped = rows_before - len(df)
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with NaN in critical columns ({rows_dropped/rows_before*100:.1f}%)")
        
        # ====================================================================
        # VALIDATION 6: Check minimum length for encoder/decoder windows
        # ====================================================================
        min_required_length = self.max_encoder_length + self.max_decoder_length
        group_lengths = df.groupby('coin').size()
        short_groups = group_lengths[group_lengths < min_required_length].index.tolist()
        if short_groups:
            logger.warning(f"Groups with insufficient length (<{min_required_length}): {short_groups}")
            # Filter out short groups
            df = df[~df['coin'].isin(short_groups)]
            logger.info(f"After filtering short groups: {df.shape}")
        
        if df.empty:
            raise ValueError(f"Dataframe is empty after filtering. Need at least {min_required_length} rows per coin.")
        
        # ====================================================================
        # VALIDATION 7: Check group_ids consistency
        # ====================================================================
        unique_coins = df['coin'].unique()
        logger.info(f"Unique coins in dataset: {len(unique_coins)}")
        if len(unique_coins) == 0:
            raise ValueError("No valid coin groups found in dataframe.")
        
        # Identify features
        known_reals, unknown_reals = self._identify_features(df)
        
        self.time_varying_known_reals = known_reals
        self.time_varying_unknown_reals = unknown_reals
        
        logger.info(f"Known reals: {known_reals}")
        logger.info(f"Unknown reals: {len(unknown_reals)} features (showing first 5: {unknown_reals[:5]})")
        
        # ====================================================================
        # VALIDATION 8: Ensure we have at least some features
        # ====================================================================
        if not unknown_reals:
            logger.warning("No unknown real features found. Model may not train properly.")
        
        # Create TimeSeriesDataSet
        try:
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
                target_normalizer=GroupNormalizer(groups=["coin"], transformation=None),  # No transformation for log returns
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True
            )
        except Exception as e:
            logger.error(f"Failed to create TimeSeriesDataSet: {e}")
            logger.error(f"Dataframe shape: {df.shape}")
            logger.error(f"Dataframe columns: {df.columns.tolist()}")
            logger.error(f"Target column '{target}' dtype: {df[target].dtype}")
            logger.error(f"Target column NaN count: {df[target].isnull().sum()}")
            raise
        
        # ====================================================================
        # VALIDATION 9: Check dataset is not None and has samples
        # ====================================================================
        if training is None:
            raise ValueError("TimeSeriesDataSet creation returned None.")
        
        dataset_length = len(training)
        if dataset_length == 0:
            raise ValueError("TimeSeriesDataSet has 0 samples. Check data filtering and window sizes.")
        
        self.training_data = training
        logger.info(f"✓ Dataset created successfully with {dataset_length} samples")
        logger.info(f"  - First timestamp: {df['time_idx'].min() if 'time_idx' in df.columns else 'N/A'}")
        logger.info(f"  - Last timestamp: {df['time_idx'].max() if 'time_idx' in df.columns else 'N/A'}")
        logger.info(f"  - Number of features: {len(unknown_reals)}")
        
        return training
    
    def build_model(
        self, 
        training_data: Optional[TimeSeriesDataSet] = None,
        output_size: Optional[int] = None,
        loss: Optional[nn.Module] = None,
        config: Optional[Dict] = None
    ) -> TemporalFusionTransformer:
        """
        Build TFT model architecture.
        
        Args:
            training_data: TimeSeriesDataSet
            output_size: Output size (1 for regression, 7 for quantiles). If None, uses config or default.
            loss: Loss function. If None, uses config or default.
            config: Config dict for task mode inference
        """
        if training_data is None:
            training_data = self.training_data
        
        if training_data is None:
            raise ValueError("Must provide training_data or call create_dataset first")
        
        logger.info("Building TFT model...")
        
        # Determine task mode and output_size/loss
        from utils.model_contracts import infer_task_mode, get_loss_for_mode, validate_tft_contract
        
        # Get config if available
        if config is None:
            config = {}
        
        # Determine output_size
        if output_size is None:
            task_config = config.get('task', {})
            if isinstance(task_config, dict) and 'mode' in task_config:
                mode = task_config['mode']
                if mode == 'regression':
                    output_size = 1
                elif mode == 'quantile':
                    output_size = 7  # Default 7 quantiles
                else:
                    output_size = 1  # Default to regression
            else:
                # Default: quantile mode (backward compatible)
                output_size = 7
        
        # Determine loss
        if loss is None:
            task_config = config.get('task', {})
            if isinstance(task_config, dict) and 'mode' in task_config:
                mode = task_config['mode']
                quantiles = task_config.get('quantiles', None)
                loss = get_loss_for_mode(mode, quantiles)
            else:
                # Default: QuantileLoss (backward compatible)
                loss = QuantileLoss()
        
        # Validate contract
        mode = infer_task_mode(config, output_size, loss.__class__.__name__)
        is_valid, error_msg = validate_tft_contract(output_size, loss, mode, config)
        if not is_valid:
            raise ValueError(f"TFT Contract Validation Failed: {error_msg}")
        
        logger.info(f"Building TFT with mode={mode}, output_size={output_size}, loss={loss.__class__.__name__}")
        
        self.model = TemporalFusionTransformer.from_dataset(
            training_data,
            learning_rate=self.learning_rate,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            hidden_continuous_size=self.hidden_size,
            output_size=output_size,
            loss=loss,
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        self.model.to(self.device)
        logger.info(f"TFT model built with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Store mode for later use
        self.task_mode = mode
        self.output_size = output_size
        
        return self.model
    
    def _extract_prediction_tensor(self, output):
        """
        Extract prediction tensor from pytorch-forecasting model output.
        
        CRITICAL FIX: Handles Output objects, dicts, and tensors robustly.
        """
        # If output has .prediction attribute (pytorch-forecasting Output object)
        if hasattr(output, 'prediction'):
            return output.prediction
        
        # If output is a dict and contains "prediction"
        if isinstance(output, dict):
            if "prediction" in output:
                return output["prediction"]
            else:
                raise ValueError(
                    f"Output is dict but missing 'prediction' key. "
                    f"Available keys: {list(output.keys())}"
                )
        
        # If output is a Tensor, use it directly
        if isinstance(output, torch.Tensor):
            return output
        
        # Otherwise raise clear error
        raise ValueError(
            f"Cannot extract prediction tensor from output type: {type(output)}. "
            f"Output attributes: {dir(output) if hasattr(output, '__dict__') else 'N/A'}"
        )
    
    def extract_median_quantile(self, pred_tensor: torch.Tensor) -> torch.Tensor:
        """
        PROFIT-FIRST FIX: Extract 0.5 quantile (median) from TFT quantile predictions.
        
        For metrics and backtest, we need a single point prediction, not 7 quantiles.
        Use the median (0.5 quantile = index 3) as the best point estimate.
        
        Args:
            pred_tensor: Shape (N, decoder_len, 7) for quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        
        Returns:
            Median predictions, shape (N, decoder_len)
        """
        if pred_tensor.ndim == 3 and pred_tensor.shape[-1] == 7:
            # Extract median quantile (index 3 = 0.5)
            return pred_tensor[:, :, 3]
        elif pred_tensor.ndim == 2:
            # Already single output or already extracted
            return pred_tensor
        else:
            raise ValueError(f"Unexpected pred_tensor shape: {pred_tensor.shape}")
    
    def extract_quantile_confidence(self, pred_tensor: torch.Tensor) -> torch.Tensor:
        """
        CONFIDENCE METRIC: Measure prediction uncertainty using quantile spread.
        
        High Confidence = Narrow spread between 0.9 and 0.1 quantiles
        Low Confidence = Wide spread (high uncertainty, avoid trading)
        
        Args:
            pred_tensor: Shape (N, decoder_len, 7)
        
        Returns:
            Confidence spread, shape (N, decoder_len). Lower = More confident.
        """
        if pred_tensor.ndim == 3 and pred_tensor.shape[-1] == 7:
            # Quantile indices: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            q_90 = pred_tensor[:, :, 5]  # 0.9 quantile
            q_10 = pred_tensor[:, :, 1]  # 0.1 quantile
            spread = torch.abs(q_90 - q_10)
            return spread
        else:
            # Cannot compute spread, return zeros
            return torch.zeros_like(pred_tensor)
    
    def _extract_target_tensor(self, y):
        """
        Extract target tensor from y (handles tuple/list like (target, weight)).
        
        CRITICAL FIX: Returns only the target tensor, not the full tuple.
        """
        if isinstance(y, (list, tuple)):
            # y is (target, weight) or similar - use first element (target)
            y_true = y[0]
            if not isinstance(y_true, torch.Tensor):
                raise ValueError(
                    f"y[0] is not a Tensor. Type: {type(y_true)}, "
                    f"y structure: {[type(item) for item in y]}"
                )
            return y_true
        elif isinstance(y, torch.Tensor):
            return y
        else:
            raise ValueError(
                f"y is not a tuple/list or Tensor. Type: {type(y)}"
            )
    
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
        
        # Create dataloaders - i5 Ultra 245KF (14 çekirdek) Optimized
        train_dataloader = training_data.to_dataloader(
            train=True, 
            batch_size=batch_size, 
            num_workers=OPTIMAL_NUM_WORKERS,  # 14 çekirdek → 8 worker (tam güç)
            pin_memory=True  # GPU'ya hızlı transfer
        )
        
        if validation_data is not None:
            val_dataloader = validation_data.to_dataloader(
                train=False, 
                batch_size=batch_size * 10, 
                num_workers=OPTIMAL_NUM_WORKERS,  # 14 çekirdek → 8 worker
                pin_memory=True
            )
        else:
            val_dataloader = None
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        debug_logged = False  # Flag for one-time debug logging
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # ====================================================================
                # CRITICAL FIX: FIX TFT TRAINING LOOP (Tuple Crash)
                # ====================================================================
                # PyTorch Forecasting's TimeSeriesDataSet returns batch as tuple (x, y)
                # x is a dictionary of tensors, y is a tuple or tensor
                # ====================================================================
                try:
                    # Explicit unpacking
                    x, y = batch
                except (ValueError, TypeError) as e:
                    logger.error(f"Error unpacking batch at index {batch_idx}: {e}")
                    logger.error(f"Batch type: {type(batch)}")
                    if hasattr(batch, '__len__'):
                        logger.error(f"Batch length: {len(batch)}")
                    raise
                
                # ====================================================================
                # CRITICAL FIX: Move entire batch to device recursively
                # ====================================================================
                from utils.device import move_to_device
                device = next(self.model.parameters()).device
                batch = move_to_device((x, y), device)
                x, y = batch
                
                # ====================================================================
                # CRITICAL FIX: Extract target tensor from y (handles tuple/list)
                # ====================================================================
                y_true = self._extract_target_tensor(y)
                
                # Forward pass
                optimizer.zero_grad()
                output = self.model(x)
                
                # ====================================================================
                # CRITICAL FIX: Use canonical loss computation
                # model.loss is an nn.Module (QuantileLoss), expects tensors (pred, target)
                # ====================================================================
                from utils.model_contracts import compute_tft_loss
                loss = compute_tft_loss(self.model, output, y)
                
                # ====================================================================
                # ONE-TIME DEBUG LOGGING (first batch only)
                # ====================================================================
                if not debug_logged and batch_idx == 0:
                    print("=" * 60)
                    print("TFT TRAINING DEBUG (First Batch):")
                    print(f"  output type: {type(output)}")
                    print(f"  output has .prediction: {hasattr(output, 'prediction')}")
                    if hasattr(output, 'prediction'):
                        print(f"  output.prediction shape: {output.prediction.shape}")
                    if hasattr(output, '__dict__'):
                        print(f"  output attributes: {list(output.__dict__.keys())}")
                    print(f"  y type: {type(y)}")
                    if isinstance(y, (list, tuple)):
                        print(f"  y length: {len(y)}")
                        print(f"  y element types: {[type(item) for item in y]}")
                        if len(y) > 0 and torch.is_tensor(y[0]):
                            print(f"  y[0] shape: {y[0].shape}")
                    print(f"  loss type: {type(loss)}, loss value: {loss.item() if torch.is_tensor(loss) else loss}")
                    print(f"  Using compute_tft_loss (canonical path)")
                    print("=" * 60)
                    debug_logged = True
                
                # Backward pass
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
                    for batch_idx, batch in enumerate(val_dataloader):
                        # ====================================================================
                        # CRITICAL FIX: FIX TFT VALIDATION LOOP (Tuple Crash)
                        # ====================================================================
                        try:
                            # Explicit unpacking
                            x, y = batch
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error unpacking validation batch at index {batch_idx}: {e}")
                            logger.error(f"Batch type: {type(batch)}")
                            raise
                        
                        # ====================================================================
                        # CRITICAL FIX: Move entire batch to device recursively
                        # ====================================================================
                        from utils.device import move_to_device
                        device = next(self.model.parameters()).device
                        batch = move_to_device((x, y), device)
                        x, y = batch
                        
                        # ====================================================================
                        # CRITICAL FIX: Extract target tensor from y (handles tuple/list)
                        # ====================================================================
                        y_true = self._extract_target_tensor(y)
                        
                        # Forward pass
                        output = self.model(x)
                        
                        # ====================================================================
                        # CRITICAL FIX: Use canonical loss computation (handles quantiles)
                        # ====================================================================
                        from utils.model_contracts import compute_tft_loss
                        loss = compute_tft_loss(self.model, output, y)
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
            target_normalizer=GroupNormalizer(groups=["coin"], transformation=None),  # No transformation for log returns
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Get last sequence
        x, _ = pred_dataset[0]
        
        # CRITICAL FIX: Move entire batch to device recursively
        from utils.device import move_to_device
        device = next(self.model.parameters()).device
        x = move_to_device(x, device)
        # Add batch dimension if needed
        if isinstance(x, dict):
            x = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() < 2 else v for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(0) if x.dim() < 2 else x
        
        # Predict
        with torch.no_grad():
            output = self.model(x)
        
        # CRITICAL FIX: Use extract_prediction_tensor to handle Output objects
        from utils.model_contracts import extract_prediction_tensor
        predictions = extract_prediction_tensor(output)  # [B, T, Q] for quantile, [B, T] for regression
        
        # Validate predictions shape
        if predictions.ndim < 2:
            raise ValueError(f"Predictions must be at least 2D [B, T, ...], got shape: {predictions.shape}")
        
        # For quantile mode, extract median (quantile 0.5)
        # For regression mode, use predictions directly
        if predictions.ndim == 3 and predictions.shape[-1] > 1:
            # Quantile mode: [B, T, Q] -> extract median index
            median_idx = predictions.shape[-1] // 2  # Middle quantile (0.5)
            pred_values = predictions[0, :, median_idx].cpu().numpy()  # [T]
            
            # Calculate confidence based on inter-quantile range
            if predictions.shape[-1] >= 7:
                # Use 0.25 and 0.75 quantiles if available
                q25_idx = 1  # Assuming quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
                q75_idx = 5
                if q25_idx < predictions.shape[-1] and q75_idx < predictions.shape[-1]:
                    q25 = predictions[0, :, q25_idx].cpu().numpy()
                    q75 = predictions[0, :, q75_idx].cpu().numpy()
                    uncertainty = np.mean(np.abs(q75 - q25))
                    confidence = 1.0 / (1.0 + uncertainty / (np.mean(np.abs(pred_values)) + 1e-8))
                    confidence = np.clip(confidence, 0.0, 1.0)
                else:
                    # Fallback: use variance across quantiles
                    pred_std = np.std(predictions[0, :, :].cpu().numpy(), axis=1).mean()
                    confidence = 1.0 / (1.0 + pred_std / (np.mean(np.abs(pred_values)) + 1e-8))
                    confidence = np.clip(confidence, 0.0, 1.0)
            else:
                # Fewer quantiles - use variance
                pred_std = np.std(predictions[0, :, :].cpu().numpy(), axis=1).mean()
                confidence = 1.0 / (1.0 + pred_std / (np.mean(np.abs(pred_values)) + 1e-8))
                confidence = np.clip(confidence, 0.0, 1.0)
        elif predictions.ndim == 2:
            # Regression mode: [B, T] -> use directly
            pred_values = predictions[0, :].cpu().numpy()  # [T]
            # No quantile information for confidence - use fixed value
            confidence = 0.5
        else:
            raise ValueError(
                f"Unexpected predictions shape: {predictions.shape}. "
                f"Expected [B, T, Q] for quantile or [B, T] for regression."
            )
        
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
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

