"""
Retrain Models with Best Params from Optuna + Run Backtest
Uses existing Optuna results, retrains with PROFIT-FIRST pipeline, then backtests.
"""

import sys
import os
import yaml
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed
from utils.logging import setup_logging
from utils.io import load_yaml, save_json, load_json
from utils.device import get_device
from data.loader_new import load_or_resample
from features.build_features import build_features
from training.train_one import split_time_series
from training.train_final import train_final_model
from models.tft import TFTModel
from evaluation.metrics import compute_metrics
from backtest.backtest import run_backtest_on_test

logger = logging.getLogger(__name__)


def retrain_and_backtest_timeframe(
    timeframe: str,
    artifacts_dir: Path,
    config: dict,
    coin: str,
    run_id: str
):
    """Retrain model with best params and run backtest for a single timeframe."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RETRAIN + BACKTEST: {timeframe}")
    logger.info(f"{'='*60}")
    
    # Load best params
    best_params_path = artifacts_dir / timeframe / 'optuna_best.json'
    if not best_params_path.exists():
        logger.error(f"Best params not found: {best_params_path}")
        return None
    
    import json
    with open(best_params_path, 'r') as f:
        best_params_data = json.load(f)
        best_params = best_params_data['best_params']
    
    logger.info(f"Loaded best params from Optuna:")
    logger.info(f"  hidden_size={best_params.get('hidden_size')}")
    logger.info(f"  batch_size={best_params.get('batch_size')}")
    logger.info(f"  encoder_length={best_params.get('max_encoder_length')}")
    logger.info(f"  dropout={best_params.get('dropout'):.3f}")
    
    # Load data
    logger.info(f"Loading data for {timeframe}...")
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
    logger.info(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # RETRAIN with best params (PROFIT-FIRST pipeline)
    logger.info(f"\n{'='*60}")
    logger.info("RETRAINING MODEL WITH PROFIT-FIRST PIPELINE")
    logger.info("Changes: Val set, Early Stopping, Dropout>=0.15")
    logger.info(f"{'='*60}")
    
    train_results = train_final_model(
        timeframe=timeframe,
        train_data=train_data,
        val_data=val_data,
        best_params=best_params,
        config=config,
        run_id=run_id,
        coin=coin
    )
    
    logger.info(f"✓ Model retrained and saved to {train_results['model_path']}")
    
    # TEST EVALUATION with PROFIT-FIRST metrics
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATING ON TEST SET (PROFIT-FIRST METRICS)")
    logger.info(f"{'='*60}")
    
    device = get_device(config.get('device', 'cuda'))
    
    # Load model
    checkpoint = torch.load(train_results['model_path'], map_location=device, weights_only=False)
    model_config_loaded = checkpoint['config']
    
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
    
    # Build model structure
    temp_train_dict = {coin: train_data.head(1000)}
    temp_dataset = model.create_dataset(temp_train_dict, target='target')
    model.build_model(temp_dataset, config=config)
    
    # Load weights
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.to(device)
    model.model.eval()
    
    logger.info(f"Model loaded: {sum(p.numel() for p in model.model.parameters()):,} parameters")
    
    # Create test dataset
    test_dict = {coin: test_data}
    test_dataset = model.create_dataset(test_dict, target='target')
    
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
            
            if isinstance(y, tuple):
                y = y[0]
            
            x = {k: v.to(device) for k, v in x.items()}
            y = y.to(device)
            
            pred = model.model(x)
            pred_tensor = model._extract_prediction_tensor(pred)
            test_predictions.append(pred_tensor.cpu().numpy())
            test_targets.append(y.cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                logger.info(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    test_predictions = np.concatenate(test_predictions, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    
    logger.info(f"Predictions shape: {test_predictions.shape}")
    logger.info(f"Targets shape: {test_targets.shape}")
    
    # Compute PROFIT-FIRST metrics
    logger.info("Computing PROFIT-FIRST metrics...")
    leverage = config.get('backtest', {}).get('max_leverage', 5.0)
    metrics = compute_metrics(test_targets, test_predictions, leverage=leverage)
    
    metrics_path = artifacts_dir / timeframe / 'metrics_test.json'
    save_json(metrics, metrics_path)
    
    logger.info(f"\n✓ PROFIT-FIRST METRICS:")
    logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f} ({metrics.get('directional_accuracy', 0)*100:.2f}%)")
    logger.info(f"  Cumulative PnL (5x): {metrics.get('pnl_cumulative', 0):.4f}")
    logger.info(f"  Sharpe Ratio: {metrics.get('pnl_sharpe', 0):.4f}")
    logger.info(f"  Win Rate: {metrics.get('pnl_win_rate', 0):.4f} ({metrics.get('pnl_win_rate', 0)*100:.2f}%)")
    logger.info(f"  Total Trades: {metrics.get('pnl_total_trades', 0)}")
    logger.info(f"\n✓ Legacy metrics (for reference):")
    logger.info(f"  MAE: {metrics.get('mae', 0):.6f}")
    logger.info(f"  RMSE: {metrics.get('rmse', 0):.6f}")
    logger.info(f"  R2: {metrics.get('r2', 0):.4f}")
    
    # Save predictions
    from evaluation.metrics import extract_median_quantile
    test_predictions_median = extract_median_quantile(test_predictions)
    
    preds_df = pd.DataFrame({
        'target': test_targets.flatten(),
        'prediction': test_predictions_median.flatten()
    })
    
    preds_path = artifacts_dir / timeframe / 'preds_test.parquet'
    preds_df.to_parquet(preds_path, index=False)
    logger.info(f"✓ Predictions saved to {preds_path}")
    
    # Run backtest
    logger.info(f"\n{'='*60}")
    logger.info("RUNNING BACKTEST")
    logger.info(f"{'='*60}")
    
    backtest_config = config.get('backtest', {})
    
    backtest_results = run_backtest_on_test(
        timeframe=timeframe,
        test_data=test_data,
        predictions=test_predictions_median.flatten(),
        config=backtest_config,
        run_id=run_id
    )
    
    bt_metrics = backtest_results.get('metrics', backtest_results)
    
    logger.info(f"\n✓ Backtest results:")
    logger.info(f"  Total Return: {bt_metrics.get('total_return', 0):.4f} ({bt_metrics.get('total_return_pct', 0):.2f}%)")
    logger.info(f"  Sharpe Ratio: {bt_metrics.get('sharpe_ratio', 0):.4f}")
    logger.info(f"  Max Drawdown: {bt_metrics.get('max_drawdown_pct', 0):.2f}%")
    logger.info(f"  Total Trades: {bt_metrics.get('total_trades', 0)}")
    logger.info(f"  Win Rate: {bt_metrics.get('win_rate_pct', 0):.2f}%")
    
    logger.info(f"\n✓ {timeframe} COMPLETE!")
    
    return {
        'metrics': metrics,
        'backtest': backtest_results
    }


def main():
    """Main function."""
    
    # Load config
    config_path = 'config/train.yaml'
    config = load_yaml(config_path)
    
    paths_config = load_yaml('config/paths.yaml')
    config['paths'] = paths_config
    
    features_config = load_yaml('config/features.yaml')
    config['features'] = features_config
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Get run_id from user
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
    log_dir = artifacts_dir / 'retrain_logs'
    setup_logging(log_dir=log_dir)
    
    logger.info("="*60)
    logger.info("PROFIT-FIRST RETRAIN + BACKTEST")
    logger.info("="*60)
    logger.info(f"\nRun ID: {run_id}")
    logger.info(f"Timeframes: {timeframes}")
    logger.info(f"\nCritical Fixes Applied:")
    logger.info("  1. Quantile Extraction: Using 0.5 (median) for metrics")
    logger.info("  2. Metrics: Directional Accuracy + PnL (priority)")
    logger.info("  3. Early Stopping: Enabled with val set")
    logger.info("  4. Dropout Clamp: Minimum 0.15 for generalization")
    logger.info("  5. Confidence Threshold: Quantile spread for signal filtering")
    
    coin = config['coin']
    
    # Retrain and backtest for each timeframe
    results = {}
    for timeframe in timeframes:
        try:
            result = retrain_and_backtest_timeframe(
                timeframe=timeframe,
                artifacts_dir=artifacts_dir,
                config=config,
                coin=coin,
                run_id=run_id
            )
            results[timeframe] = result
        except Exception as e:
            logger.error(f"Failed for {timeframe}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results[timeframe] = None
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROFIT-FIRST RETRAIN SUMMARY")
    logger.info("="*60)
    
    for timeframe in timeframes:
        result = results.get(timeframe)
        if result:
            metrics = result['metrics']
            bt_metrics = result['backtest'].get('metrics', result['backtest'])
            
            logger.info(f"\n{timeframe} PROFIT METRICS:")
            logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f} ({metrics.get('directional_accuracy', 0)*100:.2f}%)")
            logger.info(f"  PnL Sharpe: {metrics.get('pnl_sharpe', 0):.4f}")
            logger.info(f"  Backtest Return: {bt_metrics.get('total_return_pct', 0):.2f}%")
            logger.info(f"  Backtest Trades: {bt_metrics.get('total_trades', 0)}")
        else:
            logger.info(f"\n{timeframe}: FAILED")
    
    logger.info("\n" + "="*60)
    logger.info("RETRAIN COMPLETED!")
    logger.info("="*60)


if __name__ == '__main__':
    main()

