"""
Preflight Checks - Validates environment and data before training
Auto-fixes common issues when possible
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader_new import load_or_resample
from data.validators import validate_dataframe
from features.build_features import build_features
from utils.device import get_device, get_device_info, move_to_device, find_device_mismatches, model_device_sanity_check
from utils.io import load_yaml

logger = logging.getLogger(__name__)

class PreflightChecker:
    """Preflight check system with auto-fix capabilities."""
    
    def __init__(self, config: Dict, run_id: str):
        """Initialize preflight checker."""
        self.config = config
        self.run_id = run_id
        self.artifacts_dir = Path(config.get('paths', {}).get('artifacts_dir', 'artifacts')) / run_id
        self.fixes_applied = []
        self.errors = []
    
    def check_environment(self) -> Tuple[bool, List[str]]:
        """Check Python, packages, CUDA."""
        errors = []
        
        # Python version
        if sys.version_info < (3, 8):
            errors.append(f"Python {sys.version_info.major}.{sys.version_info.minor} < 3.8")
        
        # Required packages
        required = ['torch', 'numpy', 'pandas', 'optuna', 'pytorch_forecasting', 'yaml']
        for pkg in required:
            try:
                __import__(pkg if pkg != 'yaml' else 'yaml')
            except ImportError:
                errors.append(f"Missing package: {pkg}")
        
        # CUDA if requested (warn but don't block - will fallback to CPU)
        if self.config.get('device', 'cuda') == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available - will use CPU")
                # Don't add to errors - allow fallback to CPU
        
        return len(errors) == 0, errors
    
    def check_disk_space(self, min_gb: float = 1.0) -> Tuple[bool, str]:
        """Check available disk space."""
        try:
            import shutil
            stat = shutil.disk_usage('.')
            free_gb = stat.free / (1024 ** 3)
            if free_gb < min_gb:
                return False, f"Only {free_gb:.2f} GB free (need {min_gb} GB)"
            return True, f"{free_gb:.2f} GB free"
        except Exception as e:
            return False, f"Error checking disk: {e}"
    
    def check_data(
        self,
        timeframe: str,
        coin: str
    ) -> Tuple[bool, List[str], Optional[pd.DataFrame]]:
        """
        Check data for a timeframe.
        
        Returns:
            (is_valid, errors, fixed_df)
        """
        errors = []
        df = None
        
        try:
            # Load data
            paths_config = load_yaml(Path("config/paths.yaml"))
            date_range = self.config['date_range']
            
            df = load_or_resample(
                coin=coin,
                target_timeframe=timeframe,
                date_range=date_range,
                data_dir=paths_config['data_dir']
            )
            
            if df is None or len(df) == 0:
                errors.append(f"No data loaded for {timeframe}")
                return False, errors, None
            
            # Auto-fix: Sort and remove duplicates
            initial_len = len(df)
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
                df = df.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)
                if len(df) < initial_len:
                    self.fixes_applied.append(f"{timeframe}: Removed {initial_len - len(df)} duplicate timestamps")
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = set(required_cols) - set(df.columns)
            if missing:
                errors.append(f"Missing columns: {missing}")
            
            # Check for NaNs in critical columns
            for col in required_cols:
                if col in df.columns:
                    nan_count = df[col].isna().sum()
                    nan_pct = nan_count / len(df) * 100
                    
                    if nan_pct > 50:
                        errors.append(f"{col}: {nan_pct:.1f}% NaN (too many)")
                    elif nan_pct > 0:
                        # Auto-fix: forward fill then back fill
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                        self.fixes_applied.append(f"{timeframe}: Fixed {nan_count} NaN in {col}")
            
            # Check for Inf
            for col in required_cols:
                if col in df.columns:
                    inf_count = np.isinf(df[col]).sum()
                    if inf_count > 0:
                        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                        self.fixes_applied.append(f"{timeframe}: Fixed {inf_count} Inf in {col}")
            
            # Check minimum samples
            lookback = self.config.get('model', {}).get('max_encoder_length', 60)
            horizon = self.config.get('model', {}).get('max_decoder_length', 12)
            min_samples = lookback + horizon + 100  # Extra buffer
            
            if len(df) < min_samples:
                errors.append(f"Only {len(df)} samples (need {min_samples} for lookback={lookback} + horizon={horizon})")
            
            # Validate with validators
            feature_cols = []  # Will be populated after feature building
            is_valid, val_errors = validate_dataframe(
                df,
                timeframe=timeframe,
                feature_cols=feature_cols,
                timestamp_col='timestamp'
            )
            if not is_valid:
                errors.extend(val_errors)
            
        except Exception as e:
            errors.append(f"Error loading data: {e}")
        
        return len(errors) == 0, errors, df
    
    def model_device_sanity_check_with_loss(
        self,
        model,
        dataloader
    ) -> Tuple[bool, List[str]]:
        """
        Sanity check model forward pass with proper loss computation.
        
        This ensures:
        - Batch is moved to model device correctly
        - Forward pass works
        - Loss computation uses model.loss(output, y) (canonical path)
        - Shapes are validated (mode-aware)
        
        Returns:
            (success, error_messages)
        """
        errors = []
        
        try:
            device = next(model.model.parameters()).device
            
            # Get one batch
            batch = next(iter(dataloader))
            
            # Move entire batch to model device recursively
            batch = move_to_device(batch, device)
            
            # Check for device mismatches
            mismatches = find_device_mismatches(batch, device)
            if mismatches:
                errors.append(f"Device mismatches in batch: {', '.join(mismatches)}")
                return False, errors
            
            x, y = batch
            
            # Forward pass
            model.model.eval()
            with torch.no_grad():
                output = model.model(x)
                
                # CRITICAL: Use canonical loss computation (extracts tensors, handles weighting)
                from utils.model_contracts import compute_tft_loss
                
                try:
                    loss = compute_tft_loss(model.model, output, y)
                except Exception as e:
                    # Extract diagnostics for better error message
                    from utils.model_contracts import (
                        extract_prediction_tensor,
                        extract_target_tensor,
                        infer_task_mode
                    )
                    
                    try:
                        pred = extract_prediction_tensor(output)
                        target = extract_target_tensor(y)
                        
                        output_size = getattr(model.model, 'output_size', 7)
                        loss_fn_used = getattr(model.model, 'loss', None)
                        loss_class_name = loss_fn_used.__class__.__name__ if loss_fn_used else "QuantileLoss"
                        mode = infer_task_mode(self.config, output_size, loss_class_name)
                        
                        errors.append(
                            f"Loss computation failed: {e}\n"
                            f"  Mode: {mode}\n"
                            f"  pred shape: {pred.shape}\n"
                            f"  target shape: {target.shape}\n"
                            f"  device: {device}"
                        )
                    except:
                        errors.append(f"Loss computation failed: {e}")
                    
                    return False, errors
                
                # Validate loss is finite and on correct device
                if not torch.isfinite(loss):
                    errors.append(f"Loss is not finite: {loss.item()}")
                    return False, errors
                
                if loss.device != device:
                    errors.append(f"Loss on wrong device: {loss.device}, expected: {device}")
                    return False, errors
                
                # Validate shapes (mode-aware, non-blocking if loss succeeded)
                from utils.model_contracts import (
                    extract_prediction_tensor,
                    extract_target_tensor,
                    validate_pred_target_shapes,
                    infer_task_mode
                )
                
                try:
                    pred = extract_prediction_tensor(output)
                    target = extract_target_tensor(y)
                    
                    output_size = getattr(model.model, 'output_size', 7)
                    loss_fn_used = getattr(model.model, 'loss', None)
                    loss_class_name = loss_fn_used.__class__.__name__ if loss_fn_used else "QuantileLoss"
                    mode = infer_task_mode(self.config, output_size, loss_class_name)
                    
                    is_valid, error_msg, diagnostics = validate_pred_target_shapes(mode, pred, target)
                    
                    if not is_valid:
                        errors.append(f"Shape validation failed: {error_msg}")
                        errors.append(f"  Diagnostics: {diagnostics}")
                        return False, errors
                except Exception as e:
                    errors.append(f"Shape validation error: {e}")
                    return False, errors
            
            return True, []
            
        except Exception as e:
            errors.append(f"Model device sanity check failed: {e}")
            import traceback
            errors.append(f"Traceback: {traceback.format_exc()}")
            return False, errors
    
    def check_model_forward(
        self,
        timeframe: str,
        train_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """Check model forward pass on tiny batch."""
        errors = []
        
        try:
            from models.tft import TFTModel
            from pytorch_forecasting import TimeSeriesDataSet
            
            # Create tiny dataset
            coin = self.config['coin']
            tiny_df = train_df.head(100).copy()
            tiny_df['coin'] = coin
            tiny_df['time_idx'] = range(len(tiny_df))
            
            # Create TFT model
            model_config = self.config.get('model', {})
            model = TFTModel(
                prediction_horizon=model_config.get('max_decoder_length', 12),
                max_encoder_length=model_config.get('max_encoder_length', 60),
                max_decoder_length=model_config.get('max_decoder_length', 12),
                hidden_size=model_config.get('hidden_size', 128),
                device=get_device(self.config.get('device', 'cuda'))
            )
            
            # Create dataset
            data_dict = {coin: tiny_df}
            dataset = model.create_dataset(data_dict, target='close')
            # Pass config to build_model for task mode inference
            model.build_model(dataset, config=self.config)
            
            # Forward pass check using canonical helper
            dataloader = dataset.to_dataloader(train=True, batch_size=2, num_workers=0)
            
            # Use the helper function that does everything correctly
            success, check_errors = self.model_device_sanity_check_with_loss(model, dataloader)
            if not success:
                errors.extend(check_errors)
                return False, errors
            
        except Exception as e:
            errors.append(f"Model forward pass failed: {e}")
        
        return len(errors) == 0, errors
    
    def run_preflight(self, timeframe: str) -> str:
        """
        Run all preflight checks for a timeframe.
        
        Returns:
            "OK_TO_TRAIN", "FIXED_AND_OK", or "BLOCKED"
        """
        coin = self.config['coin']
        
        logger.info(f"Running preflight checks for {timeframe}...")
        
        # Environment check
        env_ok, env_errors = self.check_environment()
        if not env_ok:
            self.errors.extend([f"ENV: {e}" for e in env_errors])
            return "BLOCKED"
        
        # Disk space
        disk_ok, disk_msg = self.check_disk_space()
        if not disk_ok:
            self.errors.append(f"DISK: {disk_msg}")
            return "BLOCKED"
        logger.info(f"Disk space: {disk_msg}")
        
        # Data check
        data_ok, data_errors, fixed_df = self.check_data(timeframe, coin)
        if not data_ok:
            self.errors.extend([f"DATA: {e}" for e in data_errors])
            return "BLOCKED"
        
        # Model forward check (skip if data too small)
        if fixed_df is not None and len(fixed_df) > 200:
            model_ok, model_errors = self.check_model_forward(timeframe, fixed_df)
            if not model_ok:
                self.errors.extend([f"MODEL: {e}" for e in model_errors])
                return "BLOCKED"
        
        # Report
        if self.fixes_applied:
            logger.info(f"Applied fixes: {self.fixes_applied}")
            return "FIXED_AND_OK"
        else:
            return "OK_TO_TRAIN"

def run_preflight(config_path: str, run_id: str, timeframe: str) -> str:
    """Run preflight checks."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load paths config
    paths_config = load_yaml(Path("config/paths.yaml"))
    config['paths'] = paths_config
    
    checker = PreflightChecker(config, run_id)
    result = checker.run_preflight(timeframe)
    
    if result == "BLOCKED":
        logger.error(f"Preflight BLOCKED for {timeframe}:")
        for error in checker.errors:
            logger.error(f"  - {error}")
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--timeframe", type=str, required=True)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    result = run_preflight(args.config, args.run_id, args.timeframe)
    print(f"Result: {result}")
    sys.exit(0 if result != "BLOCKED" else 1)

