"""
Optuna Hyperparameter Optimization - Per Timeframe
"""

import sys
import os
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import json
import time
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tft import TFTModel
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, MAE
from training.trainer import train_with_early_stopping
from utils.seed import set_seed
from utils.device import get_device, recommend_num_workers, log_hardware_summary
from utils.io import save_json

# ============================================================================
# SYSTEM GUARD: Prevent CPU saturation on Windows (Intel Ultra + USB freeze)
# ============================================================================
# PyTorch's internal thread pools can saturate all CPU cores, preventing
# OS interrupts (USB, Bluetooth, etc.) from being serviced. This causes
# system freezing on Windows 11 with high-core-count CPUs.
# 
# Solution: Limit intra-op (BLAS operations) and inter-op (parallel ops) threads
# to leave headroom for OS scheduler and device drivers.
# 
# Note: These can only be set once per process. If already set (e.g., in
# verification scripts), we skip to avoid RuntimeError.
# ============================================================================
try:
    torch.set_num_threads(4)           # Intra-op parallelism (matrix ops)
    torch.set_num_interop_threads(2)   # Inter-op parallelism (parallel operations)
except RuntimeError as e:
    # Already set - this is OK (likely called from verification or parent script)
    if "cannot set number" not in str(e):
        raise

logger = logging.getLogger(__name__)

def _cleanup_trial_resources(model=None, optimizer=None, train_loader=None, val_loader=None):
    """
    Aggressive memory cleanup to prevent leaks and CPU/GPU saturation.
    
    CRITICAL for Windows stability: Ensures OS can reclaim resources between trials.
    Without this, repeated trials cause memory bloat and system freezing.
    
    Args:
        model: TFTModel instance (or None)
        optimizer: PyTorch optimizer (or None)
        train_loader: DataLoader (or None)
        val_loader: DataLoader (or None)
    """
    import gc
    
    # Delete loaders first (they hold references to datasets)
    if train_loader is not None:
        del train_loader
    if val_loader is not None:
        del val_loader
    
    # Delete optimizer (holds model parameter references)
    if optimizer is not None:
        del optimizer
    
    # Delete model last
    if model is not None:
        # If it's a TFTModel wrapper, delete the inner model too
        if hasattr(model, 'model'):
            del model.model
        del model
    
    # Force CUDA cache flush (if available)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for GPU operations to complete
    
    # Force Python garbage collection (reclaim cyclic references)
    gc.collect()

def _build_sqlite_url(db_path: Path) -> str:
    """
    Build a SQLite storage URL that is safe for multithreaded Optuna access.
    
    On Windows, SQLite enforces same-thread access by default. Appending
    `check_same_thread=false` avoids `ProgrammingError` when Optuna spawns
    background threads for parallel jobs or progress reporting.
    """
    resolved = db_path.resolve()
    return f"sqlite:///{resolved.as_posix()}?check_same_thread=false"

def _filter_features(df: pd.DataFrame, feature_flags: Dict) -> pd.DataFrame:
    """
    Filter features based on Optuna-selected feature group flags.
    
    Feature Groups:
    - Oscillators: RSI, STOCH, CCI, WilliamsR, MFI
    - Moving Averages: EMA, SMA, linreg
    - Volatility: ATR, BB (Bollinger Bands), KC (Keltner Channels)
    - Volume: OBV, VWAP, Volume_MA, Volume_Ratio
    - Patterns: MACD, SuperTrend, ADX, slope, momentum
    
    Args:
        df: DataFrame with all features
        feature_flags: Dict with boolean flags for each feature group
    
    Returns:
        Filtered DataFrame
    """
    # Base columns that are ALWAYS kept (price data, target, identifiers)
    base_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target', 'time_idx', 'coin']
    base_cols = [c for c in base_cols if c in df.columns]
    
    # Get all feature columns
    all_features = [c for c in df.columns if c not in base_cols]
    
    # If no features to filter, return as-is
    if not all_features:
        return df
    
    # Columns to KEEP (start with base columns)
    keep_cols = set(base_cols)
    
    # Feature group patterns (case-insensitive matching)
    oscillator_patterns = ['rsi', 'stoch', 'cci', 'williamsr', 'mfi']
    ma_patterns = ['ema', 'sma', 'linreg', 'intercept', 'rsquared']
    volatility_patterns = ['atr', 'bb_', 'kc_', 'volatility']
    volume_patterns = ['obv', 'vwap', 'volume_ma', 'volume_ratio']
    pattern_patterns = ['macd', 'supertrend', 'adx', 'slope', 'momentum', 'cross', 'dist_', 'strong_momentum']
    
    # Apply filters
    for col in all_features:
        col_lower = col.lower()
        
        # Check each feature group
        if feature_flags.get('use_oscillators', True):
            if any(pattern in col_lower for pattern in oscillator_patterns):
                keep_cols.add(col)
                continue
        
        if feature_flags.get('use_moving_averages', True):
            if any(pattern in col_lower for pattern in ma_patterns):
                keep_cols.add(col)
                continue
        
        if feature_flags.get('use_volatility', True):
            if any(pattern in col_lower for pattern in volatility_patterns):
                keep_cols.add(col)
                continue
        
        if feature_flags.get('use_volume', True):
            if any(pattern in col_lower for pattern in volume_patterns):
                keep_cols.add(col)
                continue
        
        if feature_flags.get('use_patterns', True):
            if any(pattern in col_lower for pattern in pattern_patterns):
                keep_cols.add(col)
                continue
        
        # Price-based features (price_change, price_high, price_low) - always keep
        if 'price_' in col_lower:
            keep_cols.add(col)
    
    # Return filtered DataFrame
    return df[list(keep_cols)]

def objective(
    trial: optuna.Trial,
    timeframe: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Dict,
    coin: str
) -> float:
    """
    Optuna objective function.
    
    Returns:
        Score to maximize (negative MAE for regression)
    """
    try:
        # ====================================================================
        # FEATURE SELECTION FLAGS (NEW: Reduce noise by selecting feature groups)
        # ====================================================================
        use_oscillators = trial.suggest_categorical('use_oscillators', [True, False])
        use_moving_averages = trial.suggest_categorical('use_moving_averages', [True, False])
        use_volatility = trial.suggest_categorical('use_volatility', [True, False])
        use_volume = trial.suggest_categorical('use_volume', [True, False])
        use_patterns = trial.suggest_categorical('use_patterns', [True, False])
        
        # Store feature flags for later use
        feature_flags = {
            'use_oscillators': use_oscillators,
            'use_moving_averages': use_moving_averages,
            'use_volatility': use_volatility,
            'use_volume': use_volume,
            'use_patterns': use_patterns
        }
        
        # ====================================================================
        # MODEL HYPERPARAMETERS (ANTI-CRASH LIMITS)
        # ====================================================================
        lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
        
        # Physical Batch Size (SAFETY LIMIT: Max 64 to prevent OOM)
        batch_size = trial.suggest_categorical('batch_size', [32, 64])
        
        # Gradient Accumulation (Virtual Batch Size Multiplier)
        # Simulates larger batches without memory cost
        # Example: 64 × 4 = 256 effective batch size
        accumulate_grad_batches = trial.suggest_categorical('accumulate_grad_batches', [1, 2, 4, 8])
        
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        
        # Hidden Size (SAFETY LIMIT: Max 256 to prevent OOM)
        # 512 REMOVED - Caused Trial 20 crash (168 encoder × 512 hidden = 50M+ params)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 160, 256])
        
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
        
        # ====================================================================
        # CONTEXT WINDOW OPTIMIZATION (For Momentum/Returns)
        # ====================================================================
        # Returns are noisier than prices, model needs more context to find patterns
        # Allow model to look back 24-168 steps (1 day to 1 week in hourly data)
        # Keep prediction short (1-2 steps) for active trading
        max_encoder_length = trial.suggest_int('max_encoder_length', 24, 168, step=12)
        max_decoder_length = trial.suggest_int('max_decoder_length', 1, 3)  # Short-term momentum
        
        # Set seed for reproducibility
        seed = config.get('seed', 42) + trial.number
        set_seed(seed)
        
        # Create model
        device = get_device(config.get('device', 'cuda'))
        model = TFTModel(
            prediction_horizon=max_decoder_length,
            max_encoder_length=max_encoder_length,
            max_decoder_length=max_decoder_length,
            hidden_size=hidden_size,
            attention_head_size=4,  # Fixed for now
            dropout=dropout,
            learning_rate=lr,
            device=device
        )
        
        # ====================================================================
        # APPLY FEATURE SELECTION (Filter features based on Optuna flags)
        # ====================================================================
        train_data_filtered = _filter_features(train_data.copy(), feature_flags)
        val_data_filtered = _filter_features(val_data.copy(), feature_flags)
        
        # Log feature selection
        original_features = len([c for c in train_data.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target', 'time_idx', 'coin']])
        filtered_features = len([c for c in train_data_filtered.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'target', 'time_idx', 'coin']])
        logger.info(f"Trial {trial.number} Feature Selection: {original_features} -> {filtered_features} features")
        logger.info(f"  Flags: {feature_flags}")
        
        # Store feature count for analysis
        trial.set_user_attr('n_features_original', original_features)
        trial.set_user_attr('n_features_selected', filtered_features)
        
        # Create datasets
        train_dict = {coin: train_data_filtered}
        val_dict = {coin: val_data_filtered}
        
        train_dataset = model.create_dataset(train_dict, target='target')
        val_dataset = model.create_dataset(val_dict, target='target')
        
        # Pass config to build_model for task mode inference
        model.build_model(train_dataset, config=config)
        
        # Create dataloaders with OOM guard
        worker_cap = config.get('num_workers_cap') or config.get('dataloader_max_workers')
        num_workers = recommend_num_workers(config.get('num_workers'), hard_cap=worker_cap)
        train_loader = None
        val_loader = None
        
        # Try batch size, reduce if OOM
        current_batch = batch_size
        max_retries = 3
        for retry in range(max_retries):
            try:
                train_loader = train_dataset.to_dataloader(
                    train=True,
                    batch_size=current_batch,
                    num_workers=num_workers,
                    pin_memory=(device == 'cuda')
                )
                val_loader = val_dataset.to_dataloader(
                    train=False,
                    batch_size=current_batch * 10,
                    num_workers=num_workers,
                    pin_memory=(device == 'cuda')
                )
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and retry < max_retries - 1:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_batch = max(32, current_batch // 2)
                    trial.set_user_attr('batch_size_reduced', current_batch)
                    logger.warning(f"OOM, reducing batch to {current_batch}")
                else:
                    raise
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        loss_fn = QuantileLoss()
        
        # Train with early stopping
        early_stopping = {
            'enabled': True,
            'patience': 3,  # Shorter for HPO
            'min_delta': 1e-4
        }
        
        history = train_with_early_stopping(
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=5,  # Fewer epochs for HPO
            device=device,
            mixed_precision=config.get('mixed_precision', 'bf16'),
            grad_clip=config.get('grad_clip', 1.0),
            accumulate_grad_batches=accumulate_grad_batches,  # Anti-OOM: Virtual batch size
            early_stopping=early_stopping,
            checkpoint_dir=None  # No checkpointing during HPO
        )
        
        # ====================================================================
        # STRATEGIC CHANGE: From "Academic Loss" to "Financial Profitability"
        # ====================================================================
        # We now evaluate models based on:
        # 1. Directional Accuracy (DA): Does the model predict price direction correctly?
        # 2. Proxy PnL: Simulated profit/loss based on directional predictions
        # 3. QuantileLoss: Still used as a constraint (not primary objective)
        # ====================================================================
        
        if not history['val_loss'] or len(history['val_loss']) == 0:
            raise ValueError("No validation loss computed")
        
        # Get validation loss (for composite score)
        val_losses_clean = [v for v in history['val_loss'] if np.isfinite(v)]
        if not val_losses_clean:
            trial.set_user_attr('nan_loss', True)
            raise optuna.TrialPruned("All validation losses are NaN")
        
        val_loss = val_losses_clean[-1]  # Use last valid loss
        
        # ====================================================================
        # COMPUTE FINANCIAL METRICS on Validation Set
        # ====================================================================
        model.model.eval()  # Set PyTorch model to eval mode
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            from utils.device import move_to_device
            model_device = next(model.model.parameters()).device
            
            for batch in val_loader:
                batch = move_to_device(batch, model_device)
                x, y = batch
                
                # Get prediction
                output = model.model(x)
                
                # Extract prediction and target tensors
                from utils.model_contracts import extract_prediction_tensor, extract_target_tensor
                pred = extract_prediction_tensor(output)
                target = extract_target_tensor(y)
                
                # For quantile models, use median (index 3 of 7 quantiles, or middle)
                if pred.dim() == 3 and pred.shape[-1] > 1:
                    # Shape: [batch, horizon, quantiles] -> use median quantile
                    median_idx = pred.shape[-1] // 2
                    pred_median = pred[:, :, median_idx]  # [batch, horizon]
                else:
                    pred_median = pred.squeeze(-1) if pred.dim() > 2 else pred
                
                # Use first timestep of prediction horizon for direction
                if pred_median.dim() > 1:
                    pred_t1 = pred_median[:, 0]  # First horizon step
                else:
                    pred_t1 = pred_median
                
                # Target: use first timestep
                if target.dim() > 1:
                    target_t1 = target[:, 0]
                else:
                    target_t1 = target
                
                predictions_list.append(pred_t1.cpu().numpy())
                targets_list.append(target_t1.cpu().numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(predictions_list)
        targets = np.concatenate(targets_list)
        
        # ====================================================================
        # METRIC 1: Directional Accuracy (DA)
        # ====================================================================
        # Predicted direction: sign(prediction)
        # Actual direction: sign(target return)
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        # ====================================================================
        # HUNTER REWARD SYSTEM: 5x Leverage + Trailing Stop Simulation
        # ====================================================================
        # Strategy: Active momentum trading with leverage
        # Goal: Maximize profit on big moves, avoid noise
        # 
        # Components:
        # 1. Trading Threshold (0.02% noise filter)
        # 2. 5x Leverage on correct directions
        # 3. Trailing Stop Bonus (1.5x multiplier on big moves)
        # 4. Inactivity Penalty (force model to trade)
        # ====================================================================
        
        # 1. Trading Threshold (avoid noise)
        THRESHOLD = 0.0002  # 0.02% minimum move to trade
        trade_active = np.abs(predictions) > THRESHOLD
        
        # 2. Direction-based PnL (sign matching)
        pred_direction = np.sign(predictions)
        raw_pnl = pred_direction * targets  # Correct direction = positive PnL
        
        # 3. Apply 5x Leverage
        leveraged_pnl = raw_pnl * 5.0
        
        # 4. Trailing Stop Reward (Big Move Bonus)
        # If actual move > 0.5% AND we were correct, apply 1.5x multiplier
        # This simulates "letting winners run" with a trailing stop
        big_move_mask = (np.abs(targets) > 0.005) & (raw_pnl > 0)
        
        final_scores = leveraged_pnl.copy()
        final_scores[big_move_mask] *= 1.5  # Rocket bonus!
        
        # 5. Calculate metrics only on active trades
        num_trades = int(np.sum(trade_active))
        
        if num_trades > 0:
            active_scores = final_scores[trade_active]
            proxy_pnl_total = float(np.sum(active_scores))
            proxy_pnl_mean = float(np.mean(active_scores))
            win_rate = float(np.mean(active_scores > 0))
        else:
            proxy_pnl_total = 0.0
            proxy_pnl_mean = 0.0
            win_rate = 0.0
        
        # 6. Inactivity Penalty
        # Model must make at least 15 trades (FIXED THRESHOLD)
        # Old: 5% of validation set (unfair for large datasets)
        # New: Absolute minimum (fair for all dataset sizes)
        min_trades = 15
        
        if num_trades < min_trades:
            # Severely penalize inactive models
            composite_score = -999.0
            logger.warning(f"Trial {trial.number} INACTIVE: {num_trades} trades < {min_trades} minimum")
        else:
            # Composite Score = Total Leveraged PnL - Loss Penalty
            # We want models that make money, not just low loss
            composite_score = proxy_pnl_total - (val_loss * 10.0)
        
        # ====================================================================
        # LOGGING & REPORTING (Hunter Metrics)
        # ====================================================================
        logger.info(f"Trial {trial.number} Hunter Metrics:")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.4f} ({directional_accuracy*100:.2f}%)")
        logger.info(f"  Active Trades: {num_trades} / {len(predictions)} ({num_trades/len(predictions)*100:.1f}%)")
        logger.info(f"  Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
        logger.info(f"  Proxy PnL (5x leveraged, mean): {proxy_pnl_mean:.6f}")
        logger.info(f"  Proxy PnL (5x leveraged, total): {proxy_pnl_total:.6f}")
        logger.info(f"  Validation Loss: {val_loss:.6f}")
        logger.info(f"  Composite Score: {composite_score:.6f}")
        
        # Store metrics as trial attributes for Optuna Dashboard
        trial.set_user_attr('directional_accuracy', float(directional_accuracy))
        trial.set_user_attr('num_trades', num_trades)
        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('proxy_pnl_mean', float(proxy_pnl_mean))
        trial.set_user_attr('proxy_pnl_total', float(proxy_pnl_total))
        trial.set_user_attr('val_loss', float(val_loss))
        trial.set_user_attr('composite_score', float(composite_score))
        
        # Report composite score for pruning (higher is better)
        trial.report(composite_score, step=len(val_losses_clean))
        
        # Check if should prune
        if trial.should_prune():
            # ====================================================================
            # AGGRESSIVE CLEANUP: Free memory before pruning
            # ====================================================================
            _cleanup_trial_resources(model, optimizer, train_loader, val_loader)
            raise optuna.TrialPruned()
        
        # ====================================================================
        # AGGRESSIVE CLEANUP: Free memory before returning (success case)
        # ====================================================================
        _cleanup_trial_resources(model, optimizer, train_loader, val_loader)
        
        # Return composite score (Optuna will maximize this)
        return composite_score
    
    except optuna.TrialPruned:
        # Cleanup before re-raising
        _cleanup_trial_resources(locals().get('model'), locals().get('optimizer'), 
                                 locals().get('train_loader'), locals().get('val_loader'))
        raise
    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
            trial.set_user_attr('oom', True)
            logger.warning(f"Trial {trial.number} OOM, pruning...")
            # AGGRESSIVE CLEANUP: Critical for OOM recovery
            _cleanup_trial_resources(locals().get('model'), locals().get('optimizer'), 
                                     locals().get('train_loader'), locals().get('val_loader'))
            raise optuna.TrialPruned("OOM")
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Trial {trial.number} failed: {error_msg}")
        logger.error(traceback.format_exc())
        trial.set_user_attr('error', error_msg)
        # AGGRESSIVE CLEANUP: Prevent memory leaks on failure
        _cleanup_trial_resources(locals().get('model'), locals().get('optimizer'), 
                                 locals().get('train_loader'), locals().get('val_loader'))
        # Mark as pruned instead of failing to avoid crashing the study
        raise optuna.TrialPruned(f"Trial failed: {error_msg}")

def run_optuna_hpo(
    timeframe: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    config: Dict,
    run_id: str,
    coin: str,
    n_trials: int = 50,
    timeout_minutes: Optional[int] = None,
    resume: bool = False
) -> Dict:
    """
    Run Optuna HPO for a timeframe.
    
    Returns:
        Dict with best_params, best_value, study info
    """
    log_hardware_summary(logger)
    # Setup storage
    artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id / timeframe
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    storage_path = artifacts_dir / 'optuna.db'
    storage_url = _build_sqlite_url(storage_path)
    
    # Create or load study
    study_name = f"tft_{timeframe}_{run_id}"
    
    if resume and Path(artifacts_dir / 'optuna.db').exists():
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=config.get('seed', 42)),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        )
        logger.info(f"Resuming Optuna study for {timeframe}")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',  # Maximize negative MAE
            sampler=TPESampler(seed=config.get('seed', 42)),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            load_if_exists=resume
        )
        logger.info(f"Created new Optuna study for {timeframe}")
    
    # Run optimization
    timeout_seconds = timeout_minutes * 60 if timeout_minutes else None
    start_time = time.time()
    
    # Get n_jobs from config (default to 1 for sequential to avoid CPU overload)
    optuna_n_jobs = config.get('hpo', {}).get('n_jobs', 1)
    
    try:
        study.optimize(
            lambda trial: objective(trial, timeframe, train_data, val_data, config, coin),
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=True,
            n_jobs=optuna_n_jobs  # Sequential by default for CPU safety
        )
    except KeyboardInterrupt:
        logger.info("Optuna optimization interrupted")
    
    duration_seconds = time.time() - start_time
    
    # Get best trial
    if len(study.trials) == 0:
        raise ValueError("No trials completed")
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Save best params
    best_info = {
        'best_params': best_params,
        'best_value': float(best_trial.value) if best_trial.value is not None else None,
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_failed': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'duration_seconds': duration_seconds,
        'seed': config.get('seed', 42),
        'metric_name': 'negative_mae',
        'timeframe': timeframe,
    }
    
    save_json(best_info, artifacts_dir / 'optuna_best.json')
    
    logger.info(f"Optuna HPO completed for {timeframe}")
    logger.info(f"  Best value: {best_info['best_value']:.6f}")
    logger.info(f"  Best params: {best_params}")
    logger.info(f"  Trials: {best_info['n_complete']} complete, {best_info['n_pruned']} pruned, {best_info['n_failed']} failed")
    
    return best_info
