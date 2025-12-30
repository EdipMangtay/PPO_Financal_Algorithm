"""
ANTI-CRASH STABILITY TEST
=========================

Tests system stability under MAXIMUM stress configuration:
- hidden_size=256 (max allowed)
- max_encoder_length=168 (max allowed)  
- batch_size=64 (max allowed)

If this passes without OOM, the system is stable for all Optuna trials.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from models.tft import TFTModel
from data.loader_new import load_or_resample
from data.validators import validate_dataframe
from features.build_features import build_features
from training.train_one import split_time_series
from training.trainer import train_with_early_stopping
from pytorch_forecasting.metrics import QuantileLoss
from utils.seed import set_seed
from utils.device import get_device, log_hardware_summary
from utils.io import load_yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_stability_test():
    """
    Run a stability test with maximum stress parameters.
    
    This simulates the worst-case scenario that caused Trial 20 to crash:
    - Large model (hidden_size=256)
    - Long context (max_encoder_length=168)
    - Maximum batch size (batch_size=64)
    
    If this completes without OOM, the system is stable.
    """
    
    print("=" * 80)
    print("ANTI-CRASH STABILITY TEST")
    print("=" * 80)
    print("\nTesting MAXIMUM stress configuration:")
    print("  hidden_size: 256 (max allowed)")
    print("  max_encoder_length: 168 (max allowed)")
    print("  batch_size: 64 (max physical limit)")
    print("  accumulate_grad_batches: 1 (no accumulation for stress test)")
    print("\nIf this passes, all Optuna trials will be safe from OOM crashes.")
    print("=" * 80)
    
    try:
        # 1. Load configuration
        logger.info("Step 1: Loading configuration...")
        config = load_yaml(Path("config/train.yaml"))
        paths_config = load_yaml(Path("config/paths.yaml"))
        config['paths'] = paths_config
        
        set_seed(42)
        device = get_device(config.get('device', 'cuda'))
        log_hardware_summary()
        
        # 2. Load data (small subset for speed)
        logger.info("Step 2: Loading data (15m timeframe, reduced for speed)...")
        coin = "BTC/USDT"
        timeframe = "15m"
        
        df = load_or_resample(
            coin=coin,
            target_timeframe=timeframe,
            date_range=config['date_range'],
            data_dir=config['paths']['data_dir']
        )
        
        # Skip validate_dataframe for raw data (no features yet)
        logger.info(f"Loaded {len(df)} bars")
        
        # 3. Build features
        logger.info("Step 3: Building features...")
        features_config = load_yaml(Path("config/features.yaml"))
        df_with_features, target_series, _ = build_features(df, timeframe=timeframe, feature_config=features_config)
        
        # Add target back to dataframe
        df_with_features['target'] = target_series
        
        logger.info(f"Built {len(df_with_features.columns)} features, {len(df_with_features)} rows")
        
        # 4. Split data (use 30% for speed, enough for model)
        logger.info("Step 4: Splitting data (30% for fast OOM test)...")
        subset_size = int(len(df_with_features) * 0.3)
        df_subset = df_with_features.iloc[:subset_size].copy()
        
        # Simple time-based split
        n = len(df_subset)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_data = df_subset.iloc[:train_end].copy()
        val_data = df_subset.iloc[train_end:val_end].copy()
        test_data = df_subset.iloc[val_end:].copy()
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # 5. Create MAXIMUM STRESS MODEL
        logger.info("Step 5: Creating MAXIMUM STRESS model...")
        
        # STRESS TEST PARAMETERS (Maximum values allowed in Optuna)
        MAX_HIDDEN_SIZE = 256
        MAX_ENCODER_LENGTH = 168
        MAX_BATCH_SIZE = 64
        MAX_DECODER_LENGTH = 3
        
        model = TFTModel(
            prediction_horizon=MAX_DECODER_LENGTH,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_decoder_length=MAX_DECODER_LENGTH,
            hidden_size=MAX_HIDDEN_SIZE,
            attention_head_size=4,
            dropout=0.1,
            learning_rate=1e-3,
            device=device
        )
        
        # 6. Create datasets
        logger.info("Step 6: Creating datasets...")
        train_dict = {coin: train_data}
        val_dict = {coin: val_data}
        
        train_dataset = model.create_dataset(train_dict, target='target')
        val_dataset = model.create_dataset(val_dict, target='target')
        
        model.build_model(train_dataset, config=config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # 7. Create dataloaders with MAXIMUM STRESS batch size
        logger.info(f"Step 7: Creating dataloaders (batch_size={MAX_BATCH_SIZE})...")
        
        train_loader = train_dataset.to_dataloader(
            train=True,
            batch_size=MAX_BATCH_SIZE,
            num_workers=0,  # Windows safe
            pin_memory=(device == 'cuda')
        )
        
        val_loader = val_dataset.to_dataloader(
            train=False,
            batch_size=MAX_BATCH_SIZE * 2,  # Larger for inference
            num_workers=0,
            pin_memory=(device == 'cuda')
        )
        
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # 8. Run DRY RUN training (1 epoch, 10 batches max)
        logger.info("Step 8: Running DRY RUN training (1 epoch or 10 batches, whichever is less)...")
        logger.info("This is the CRITICAL OOM TEST...")
        
        optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=1e-3,
            weight_decay=1e-6
        )
        
        loss_fn = QuantileLoss()
        
        # Dry run: 1 epoch or 10 batches
        history = train_with_early_stopping(
            model=model.model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=1,  # Just 1 epoch for dry run
            device=device,
            mixed_precision=config.get('mixed_precision', 'bf16'),
            grad_clip=1.0,
            accumulate_grad_batches=1,  # No accumulation for stress test
            early_stopping=None,  # No early stopping for dry run
            checkpoint_dir=None
        )
        
        # 9. Check VRAM usage (if CUDA)
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Peak VRAM usage: {vram_used:.2f} GB / {vram_total:.2f} GB ({vram_used/vram_total*100:.1f}%)")
        
        # 10. Validate training worked
        if not history['train_loss'] or not history['val_loss']:
            raise ValueError("Training produced no losses!")
        
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        logger.info(f"Final train loss: {final_train_loss:.6f}")
        logger.info(f"Final val loss: {final_val_loss:.6f}")
        
        # SUCCESS!
        print("\n" + "=" * 80)
        print("SUCCESS: SYSTEM IS STABLE")
        print("=" * 80)
        print("\nStress test PASSED with maximum parameters:")
        print(f"  [OK] hidden_size: {MAX_HIDDEN_SIZE}")
        print(f"  [OK] max_encoder_length: {MAX_ENCODER_LENGTH}")
        print(f"  [OK] batch_size: {MAX_BATCH_SIZE}")
        print(f"  [OK] Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        if torch.cuda.is_available():
            print(f"\n  GPU Memory:")
            print(f"    Peak usage: {vram_used:.2f} GB / {vram_total:.2f} GB ({vram_used/vram_total*100:.1f}%)")
            print(f"    Safety margin: {vram_total - vram_used:.2f} GB ({(1 - vram_used/vram_total)*100:.1f}%)")
        
        print("\n[OK] All Optuna trials will run safely without OOM crashes!")
        print("=" * 80)
        
        return True
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 80)
            print("[FAIL] OUT OF MEMORY (OOM)")
            print("=" * 80)
            print("\nThe system CANNOT handle maximum stress parameters.")
            print("\nRECOMMENDATIONS:")
            print("  1. Reduce max_encoder_length to 120 (from 168)")
            print("  2. Reduce hidden_size max to 128 (from 256)")
            print("  3. Reduce batch_size to 32 (from 64)")
            print("  4. Enable gradient accumulation (set accumulate_grad_batches > 1)")
            print("\nPlease adjust hpo/optuna_search.py accordingly.")
            print("=" * 80)
            return False
        else:
            raise
    
    except Exception as e:
        logger.error(f"Stability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_stability_test()
    sys.exit(0 if success else 1)

