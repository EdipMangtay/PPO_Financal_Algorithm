"""
Final Training - Train model with best HPO params
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tft import TFTModel
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from training.trainer import train_with_early_stopping
from utils.seed import set_seed
from utils.device import get_device
from utils.io import load_json, save_json

logger = logging.getLogger(__name__)

def train_final_model(
    timeframe: str,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    best_params: Dict,
    config: Dict,
    run_id: str,
    coin: str
) -> Dict:
    """
    Train final model with best HPO parameters.
    
    CRITICAL FIXES:
    1. Keep val_data separate for early stopping (prevent overfitting)
    2. Safety clamp dropout to minimum 0.15 for generalization
    3. Enable early stopping to prevent test set degradation
    
    Returns:
        Dict with model_path, metrics, history
    """
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Setup paths
    artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id / timeframe
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(config.get('device', 'cuda'))
    
    # Extract params
    lr = best_params.get('lr', config['learning_rate'][timeframe])
    batch_size = best_params.get('batch_size', config['batch_size'][timeframe])
    dropout_raw = best_params.get('dropout', config['model']['dropout'])
    hidden_size = best_params.get('hidden_size', config['model']['hidden_size'])
    weight_decay = best_params.get('weight_decay', 1e-5)
    accumulate_grad_batches = best_params.get('accumulate_grad_batches', 1)
    
    # SAFETY CLAMP: Minimum dropout for generalization on test set
    dropout = max(dropout_raw, 0.15)
    if dropout != dropout_raw:
        logger.warning(f"Dropout clamped from {dropout_raw:.2f} to {dropout:.2f} for better generalization")
    
    model_config = config.get('model', {})
    max_encoder_length = best_params.get('max_encoder_length', model_config.get('max_encoder_length', 60))
    max_decoder_length = best_params.get('max_decoder_length', model_config.get('max_decoder_length', 3))
    
    logger.info(f"Training final model for {timeframe} with best params:")
    logger.info(f"  lr={lr}, batch_size={batch_size}, hidden_size={hidden_size}, dropout={dropout}")
    logger.info(f"  encoder_length={max_encoder_length}, decoder_length={max_decoder_length}")
    logger.info(f"  accumulate_grad_batches={accumulate_grad_batches}")
    
    # Create model
    model = TFTModel(
        prediction_horizon=max_decoder_length,
        max_encoder_length=max_encoder_length,
        max_decoder_length=max_decoder_length,
        hidden_size=hidden_size,
        attention_head_size=4,
        dropout=dropout,
        learning_rate=lr,
        device=device
    )
    
    # CRITICAL FIX: Keep train and val SEPARATE for early stopping
    # This prevents overfitting on train and ensures good test performance
    train_dict = {coin: train_data}
    val_dict = {coin: val_data}
    
    # Create datasets
    train_dataset = model.create_dataset(train_dict, target='target')
    val_dataset = model.create_dataset(val_dict, target='target')
    
    # Pass config to build_model for task mode inference
    model.build_model(train_dataset, config=config)
    
    # Create dataloaders
    num_workers = config.get('num_workers', 0)
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 2,
        num_workers=num_workers,
        pin_memory=(device == 'cuda')
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    loss_fn = QuantileLoss()
    
    # ENABLE EARLY STOPPING: Stop when no improvement, save best model
    # min_delta = 0.00001 (10x more sensitive than 0.0001)
    # This allows small but real improvements to be recognized
    early_stopping_config = {
        'enabled': True,
        'patience': 5,
        'min_delta': 0.00001,  # 10x daha hassas (küçük iyileşmeleri de yakalar)
        'mode': 'min'
    }
    logger.info("Early stopping ENABLED with patience=5, min_delta=0.00001")
    
    epochs = config['epochs'][timeframe]
    
    ckpt_dir = artifacts_dir / 'checkpoints'
    logger.info(f"Training up to {epochs} epochs (may stop early if no improvement)")
    
    history = train_with_early_stopping(
        model=model.model,
        train_loader=train_loader,
        val_loader=val_loader,  # CRITICAL: Use val_loader for early stopping
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        mixed_precision=config.get('mixed_precision', 'bf16'),
        grad_clip=config.get('grad_clip', 1.0),
        early_stopping=early_stopping_config,
        checkpoint_dir=ckpt_dir,
        accumulate_grad_batches=accumulate_grad_batches
    )
    
    # Save model
    model_path = artifacts_dir / 'model.pt'
    model.save(str(model_path))
    
    logger.info(f"Final model saved to {model_path}")
    
    return {
        'model_path': str(model_path),
        'history': history,
        'params': best_params,
    }

