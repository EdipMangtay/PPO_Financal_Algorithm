"""
Backtest Only - Use existing trained models
Run backtest on test set for all timeframes
"""

import sys
import os
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logging import setup_logging
from utils.io import load_yaml, save_json
from utils.device import get_device
from data.loader_new import load_or_resample
from features.build_features import build_features
from training.train_one import split_time_series
from models.tft import TFTModel
from evaluation.metrics import compute_metrics
from backtest.backtest import run_backtest_on_test

logger = logging.getLogger(__name__)

def run_backtest_for_timeframe(
    timeframe: str,
    artifacts_dir: Path,
    config: dict,
    coin: str
):
    """Run backtest for a single timeframe using existing model."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST: {timeframe}")
    logger.info(f"{'='*60}")
    
    # Check if model exists
    model_path = artifacts_dir / timeframe / 'model.pt'
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    # Load best params
    best_params_path = artifacts_dir / timeframe / 'optuna_best.json'
    if not best_params_path.exists():
        logger.error(f"Best params not found: {best_params_path}")
        return None
    
    import json
    with open(best_params_path, 'r') as f:
        best_params_data = json.load(f)
        best_params = best_params_data['best_params']
    
    logger.info(f"Loading data for {timeframe}...")
    
    # Load data
    date_range = config['date_range']
    df = load_or_resample(
        coin=coin,
        target_timeframe=timeframe,
        date_range=date_range,
        data_dir=Path(config['paths']['data_dir'])
    )
    
    logger.info(f"Loaded {len(df)} bars")
    
    # Build features
    logger.info("Building features...")
    df_with_features, target_series, feature_report = build_features(
        df, 
        timeframe=timeframe, 
        feature_config=config['features']
    )
    
    # Add target back
    df_with_features['target'] = target_series
    
    logger.info(f"Features built: {len(df_with_features.columns)} columns")
    
    # Split data
    splits = split_time_series(
        df_with_features,
        train_ratio=config['split']['train'],
        val_ratio=config['split']['val'],
        test_ratio=config['split']['test']
    )
    
    train_data = splits['train']
    val_data = splits['val']
    test_data = splits['test']
    logger.info(f"Test set: {len(test_data)} samples")
    
    # Get device
    device = get_device(config.get('device', 'cuda'))
    
    # Load model checkpoint to get config
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    model_config_loaded = checkpoint['config']
    
    # Create model instance with saved config
    model = TFTModel(
        prediction_horizon=model_config_loaded['prediction_horizon'],
        max_encoder_length=model_config_loaded['max_encoder_length'],
        max_decoder_length=model_config_loaded['max_decoder_length'],
        hidden_size=model_config_loaded['hidden_size'],
        attention_head_size=model_config_loaded['attention_head_size'],
        dropout=model_config_loaded['dropout'],
        learning_rate=model_config_loaded['learning_rate'],
        device=device
    )
    
    # Need to build model first with a dataset, then load weights
    # Create a small dataset for model structure
    temp_train_dict = {coin: train_data.head(1000)}  # Use small subset just for structure
    temp_dataset = model.create_dataset(temp_train_dict, target='target')
    model.build_model(temp_dataset, config=config)
    
    # Now load the actual trained weights
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.to(device)
    model.model.eval()
    
    logger.info(f"Model loaded: {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    # Create test dataset
    test_dict = {coin: test_data}
    test_dataset = model.create_dataset(test_dict, target='target')
    
    # Create test dataloader
    test_loader = test_dataset.to_dataloader(
        train=False,
        batch_size=best_params.get('batch_size', 64) * 2,
        num_workers=0
    )
    
    logger.info(f"Running inference on test set ({len(test_loader)} batches)...")
    
    # Run predictions
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x, y = batch
            
            # PyTorch Forecasting returns y as (targets, weights) tuple
            if isinstance(y, tuple):
                y = y[0]
            
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            
            pred = model.model(x)
            # Extract prediction tensor from Output object
            pred_tensor = model._extract_prediction_tensor(pred)
            test_predictions.append(pred_tensor.cpu().numpy())
            test_targets.append(y.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    logger.info(f"Predictions shape (raw): {test_predictions.shape}")
    logger.info(f"Targets shape (raw): {test_targets.shape}")
    
    # Extract median quantile (index 3, which is 0.5) for quantile predictions
    # Shape: (N, decoder_len, 7 quantiles) -> (N, decoder_len)
    if test_predictions.ndim == 3 and test_predictions.shape[-1] == 7:
        logger.info("Extracting median quantile (0.5) from predictions...")
        test_predictions = test_predictions[:, :, 3]  # Median quantile
        logger.info(f"Predictions shape (after median extraction): {test_predictions.shape}")
    
    # Compute metrics
    logger.info("Computing test metrics...")
    metrics = compute_metrics(test_targets, test_predictions)
    
    metrics_path = artifacts_dir / timeframe / 'metrics_test.json'
    save_json(metrics, metrics_path)
    
    logger.info(f"✓ Test metrics:")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  R2: {metrics['r2']:.4f}")
    
    # Save predictions
    preds_df = pd.DataFrame({
        'target': test_targets.flatten(),
        'prediction': test_predictions.flatten()
    })
    
    preds_path = artifacts_dir / timeframe / 'preds_test.parquet'
    preds_df.to_parquet(preds_path, index=False)
    logger.info(f"✓ Predictions saved to {preds_path}")
    
    # Run backtest
    logger.info("Running backtest...")
    backtest_config = config.get('backtest', {})
    
    # Get run_id from artifacts_dir
    run_id = artifacts_dir.name
    
    backtest_results = run_backtest_on_test(
        timeframe=timeframe,
        test_data=test_data,
        predictions=test_predictions.flatten(),
        config=backtest_config,
        run_id=run_id
    )
    
    backtest_path = artifacts_dir / timeframe / 'backtest_results.json'
    save_json(backtest_results, backtest_path)
    
    # Extract metrics from nested structure
    metrics = backtest_results.get('metrics', backtest_results)
    
    logger.info(f"✓ Backtest results:")
    logger.info(f"  Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return_pct', 0):.2f}%)")
    logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
    logger.info(f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
    
    logger.info(f"\n✓ {timeframe} backtest completed!")
    
    return {
        'metrics': metrics,
        'backtest': backtest_results
    }


def main():
    """Main function."""
    
    # Load config first (needed for log dir)
    config_path = 'config/train.yaml'
    config = load_yaml(config_path)
    
    # Load paths config
    paths_config = load_yaml('config/paths.yaml')
    config['paths'] = paths_config
    
    # Load features config
    features_config = load_yaml('config/features.yaml')
    config['features'] = features_config
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Get run_id from user
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='Run ID (e.g., 20251230_005546)')
    parser.add_argument('--timeframes', type=str, default='15m,1h,4h', help='Comma-separated timeframes')
    args = parser.parse_args()
    
    run_id = args.run_id
    timeframes = args.timeframes.split(',')
    
    artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id
    
    if not artifacts_dir.exists():
        print(f"ERROR: Run directory not found: {artifacts_dir}")
        return
    
    # Setup logging
    log_dir = artifacts_dir / 'backtest_logs'
    setup_logging(log_dir=log_dir)
    
    logger.info("="*60)
    logger.info("BACKTEST ONLY - Using Existing Models")
    logger.info("="*60)
    
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"Artifacts dir: {artifacts_dir}")
    
    coin = config['coin']
    
    # Run backtest for each timeframe
    results = {}
    for timeframe in timeframes:
        try:
            result = run_backtest_for_timeframe(
                timeframe=timeframe,
                artifacts_dir=artifacts_dir,
                config=config,
                coin=coin
            )
            results[timeframe] = result
        except Exception as e:
            logger.error(f"Failed to run backtest for {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[timeframe] = None
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("BACKTEST SUMMARY")
    logger.info("="*60)
    
    for timeframe in timeframes:
        result = results.get(timeframe)
        if result:
            metrics = result['backtest'].get('metrics', result['backtest'])
            logger.info(f"\n{timeframe}:")
            logger.info(f"  Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return_pct', 0):.2f}%)")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"  Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"  Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
        else:
            logger.info(f"\n{timeframe}: FAILED")
    
    logger.info("\n" + "="*60)
    logger.info("BACKTEST COMPLETED!")
    logger.info("="*60)


if __name__ == '__main__':
    main()

