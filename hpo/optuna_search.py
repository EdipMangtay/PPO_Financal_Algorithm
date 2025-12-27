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
from utils.device import get_device
from utils.io import save_json

logger = logging.getLogger(__name__)

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
        # Suggest hyperparameters
        lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
        
        # Model config
        model_config = config.get('model', {})
        max_encoder_length = model_config.get('max_encoder_length', 60)
        max_decoder_length = model_config.get('max_decoder_length', 12)
        
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
        
        # Create datasets
        train_dict = {coin: train_data}
        val_dict = {coin: val_data}
        
        train_dataset = model.create_dataset(train_dict, target='target')
        val_dataset = model.create_dataset(val_dict, target='target')
        
        # Pass config to build_model for task mode inference
        model.build_model(train_dataset, config=config)
        
        # Create dataloaders with OOM guard
        num_workers = config.get('num_workers', 0)
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
            early_stopping=early_stopping,
            checkpoint_dir=None  # No checkpointing during HPO
        )
        
        # Get validation loss
        if history['val_loss'] and len(history['val_loss']) > 0:
            # Get best validation loss (lowest, most recent valid)
            val_losses_clean = [v for v in history['val_loss'] if np.isfinite(v)]
            if not val_losses_clean:
                trial.set_user_attr('nan_loss', True)
                raise optuna.TrialPruned("All validation losses are NaN")
            
            val_loss = val_losses_clean[-1]  # Use last valid loss
            
            # Report intermediate value for pruning
            trial.report(val_loss, step=len(val_losses_clean))
            
            # Check if should prune
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Return negative loss (to maximize, since Optuna maximizes)
            return -val_loss
        else:
            raise ValueError("No validation loss computed")
    
    except optuna.TrialPruned:
        raise
    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
            trial.set_user_attr('oom', True)
            logger.warning(f"Trial {trial.number} OOM, pruning...")
            raise optuna.TrialPruned("OOM")
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Trial {trial.number} failed: {error_msg}")
        logger.error(traceback.format_exc())
        trial.set_user_attr('error', error_msg)
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
    # Setup storage
    artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id / timeframe
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{artifacts_dir / 'optuna.db'}"
    
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
    
    try:
        study.optimize(
            lambda trial: objective(trial, timeframe, train_data, val_data, config, coin),
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=True
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

