"""
Train One Timeframe - Complete training pipeline for a single timeframe
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging
import sys
import os

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader_new import load_or_resample
from data.validators import validate_dataframe
from data.cache import compute_feature_hash, compute_dataset_hash, load_from_cache, save_to_cache
from features.build_features import build_features, save_feature_report
from training.trainer import train_with_early_stopping
from models.tft import TFTModel
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import torch

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame by time (no shuffling to avoid lookahead).
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        'train': df.iloc[:train_end].copy(),
        'val': df.iloc[train_end:val_end].copy(),
        'test': df.iloc[val_end:].copy(),
    }

def train_one_timeframe(
    config: Dict,
    timeframe: str,
    run_id: str,
    coin: str = None
) -> Dict:
    """
    Train model for one timeframe.
    
    Returns:
        Dict with model_path, metrics, calibration info
    """
    # Get config values
    if coin is None:
        coin = config['coin']
    
    date_range = config['date_range']
    train_config = config
    feature_config_path = "config/features.yaml"
    paths_config_path = "config/paths.yaml"
    
    # Load feature and path configs
    feature_config = load_config(feature_config_path)
    paths_config = load_config(paths_config_path)
    
    # Setup paths
    artifacts_dir = Path(paths_config['artifacts_dir']) / run_id / timeframe
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = paths_config['cache_dir']
    ckpt_dir = Path(paths_config['ckpt_dir']) / run_id / timeframe
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training {coin} {timeframe} model")
    
    # Load data
    logger.info("Loading raw data...")
    df_raw = load_or_resample(
        coin=coin,
        target_timeframe=timeframe,
        date_range=date_range,
        data_dir=paths_config['data_dir']
    )
    
    # Validate data
    logger.info("Validating data...")
    is_valid, errors = validate_dataframe(
        df_raw,
        timeframe=timeframe,
        feature_cols=[],
        timestamp_col='timestamp'
    )
    if not is_valid:
        raise ValueError(f"Data validation failed: {errors}")
    
    # Build features
    logger.info("Building features...")
    feature_hash = compute_feature_hash(feature_config)
    dataset_hash = compute_dataset_hash(coin, timeframe, date_range, feature_hash)
    
    # Try cache
    cached_data = None
    if config.get('caching', True):
        cached_data = load_from_cache(cache_dir, dataset_hash)
    
    if cached_data:
        features_df = cached_data['features_df']
        target = cached_data['target']
        metadata = cached_data['metadata']
        logger.info("Loaded from cache")
    else:
        features_df, target, metadata = build_features(
            df_raw,
            timeframe=timeframe,
            feature_config=feature_config,
            target_horizon_bars=feature_config['target']['horizon_bars'][timeframe]
        )
        
        # Save to cache
        if config.get('caching', True):
            from data.cache import save_to_cache
            save_to_cache(cache_dir, dataset_hash, {
                'features_df': features_df,
                'target': target,
                'metadata': metadata
            })
    
    # Save feature report
    save_feature_report(metadata, artifacts_dir / 'feature_report.json', timeframe)
    
    # Split data
    logger.info("Splitting data...")
    splits = split_time_series(
        features_df,
        train_ratio=config['split']['train'],
        val_ratio=config['split']['val'],
        test_ratio=config['split']['test']
    )
    
    # Add target to splits
    for split_name in splits:
        split_idx = splits[split_name].index
        splits[split_name]['target'] = target.loc[split_idx]
    
    # Create TFT model
    model_config = train_config.get('model', {})
    model = TFTModel(
        prediction_horizon=feature_config['target']['horizon_bars'][timeframe],
        max_encoder_length=model_config.get('max_encoder_length', 60),
        max_decoder_length=model_config.get('max_decoder_length', 12),
        hidden_size=model_config.get('hidden_size', 128),
        attention_head_size=model_config.get('attention_head_size', 4),
        dropout=model_config.get('dropout', 0.1),
        learning_rate=train_config['learning_rate'][timeframe],
        device=train_config.get('device', 'cuda')
    )
    
    # Create datasets
    logger.info("Creating TFT datasets...")
    train_data_dict = {coin: splits['train']}
    val_data_dict = {coin: splits['val']}
    
    train_dataset = model.create_dataset(train_data_dict, target='target')
    val_dataset = model.create_dataset(val_data_dict, target='target')
    
    model.build_model(train_dataset)
    
    # Create dataloaders
    batch_size = train_config['batch_size'][timeframe]
    num_workers = train_config.get('num_workers', 0)
    
    train_loader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(train_config.get('device', 'cuda') == 'cuda')
    )
    
    val_loader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size * 10,
        num_workers=num_workers,
        pin_memory=(train_config.get('device', 'cuda') == 'cuda')
    )
    
    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.model.parameters(), lr=train_config['learning_rate'][timeframe])
    loss_fn = QuantileLoss()
    
    # Train
    logger.info("Starting training...")
    early_stopping = train_config.get('early_stopping', {})
    
    history = train_with_early_stopping(
        model=model.model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=train_config['epochs'][timeframe],
        device=train_config.get('device', 'cuda'),
        mixed_precision=train_config.get('mixed_precision', 'bf16'),
        grad_clip=train_config.get('grad_clip', 1.0),
        early_stopping=early_stopping if early_stopping.get('enabled', False) else None,
        checkpoint_dir=ckpt_dir
    )
    
    # Save final model
    model_path = ckpt_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'metadata': metadata,
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Compute metrics
    metrics = {
        'train_loss_final': history['train_loss'][-1] if history['train_loss'] else float('nan'),
        'val_loss_final': history['val_loss'][-1] if history['val_loss'] else float('nan'),
        'best_val_loss': history.get('best_val_loss', float('nan')),
        'best_epoch': history.get('best_epoch', 0),
        'nan_incidents': history.get('nan_incidents', 0),
        'oom_incidents': history.get('oom_incidents', 0),
    }
    
    return {
        'model_path': str(model_path),
        'metrics': metrics,
        'history': history,
        'metadata': metadata,
    }


