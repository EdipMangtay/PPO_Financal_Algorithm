"""
BTC Pipeline - Optimized for Single Coin, 3 Timeframes
Sequential Optuna HPO, then parallel training (CPU optimized)
"""

import sys
import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed, get_seed_info
from utils.logging import setup_logging
from utils.io import load_yaml, save_json, load_json
from utils.device import get_device, get_device_info
from scripts.preflight import PreflightChecker
from data.loader_new import load_or_resample
from data.validators import validate_dataframe
from features.build_features import build_features, save_feature_report
from training.train_one import split_time_series
from hpo.optuna_search import run_optuna_hpo
from training.train_final import train_final_model
from models.tft import TFTModel
from pytorch_forecasting import TimeSeriesDataSet
from evaluation.metrics import compute_metrics
from backtest.backtest import run_backtest_on_test
import pandas as pd
import numpy as np
import torch

# CPU Optimization - Set before any imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['PYTORCH_NUM_THREADS'] = '1'

def generate_run_id() -> str:
    """Generate run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_all_configs(config_path: str) -> Dict:
    """Load all configuration files."""
    train_config = load_yaml(Path(config_path))
    paths_config = load_yaml(Path("config/paths.yaml"))
    feature_config = load_yaml(Path("config/features.yaml"))
    
    # Merge paths into main config
    train_config['paths'] = paths_config
    train_config['features'] = feature_config
    
    # Force BTC only and 3 timeframes
    if 'coin' not in train_config or train_config['coin'] != "BTC/USDT":
        train_config['coin'] = "BTC/USDT"
    if 'timeframes' not in train_config:
        train_config['timeframes'] = ["15m", "1h", "4h"]
    
    return train_config

def process_timeframe_hpo(
    timeframe: str,
    config: Dict,
    run_id: str,
    hpo_trials: int = 50,
    hpo_timeout: Optional[int] = None
) -> Optional[Dict]:
    """
    Process one timeframe: preflight -> HPO only.
    
    Returns:
        Dict with best_params or None if failed
    """
    coin = config['coin']
    logger = logging.getLogger(f"btc_hpo_{timeframe}")
    
    try:
        logger.info("=" * 60)
        logger.info(f"BTC {timeframe} - OPTUNA HPO")
        logger.info("=" * 60)
        
        # Setup paths
        artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id / timeframe
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(artifacts_dir, timeframe=timeframe)
        
        # Preflight
        logger.info("Step 1: Preflight checks...")
        checker = PreflightChecker(config, run_id)
        preflight_result = checker.run_preflight(timeframe)
        
        if preflight_result == "BLOCKED":
            logger.error(f"Preflight BLOCKED for {timeframe}")
            return None
        
        # Load data
        logger.info("Step 2: Loading data...")
        date_range = config.get('date_range', {'start': '2023-01-01', 'end': '2024-12-31'})
        
        try:
            df = load_or_resample(
                coin=coin,
                target_timeframe=timeframe,
                date_range=date_range,
                data_dir=config['paths']['data_dir']
            )
            # Validate data (feature_cols empty since features not built yet)
            is_valid, errors = validate_dataframe(
                df,
                timeframe=timeframe,
                feature_cols=[],  # Empty - features not built yet
                timestamp_col='timestamp'
            )
            if not is_valid:
                logger.warning(f"Data validation warnings: {errors}")
                # Continue anyway - preflight already checked
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None
        
        if df is None or len(df) == 0:
            logger.error(f"Invalid or empty dataframe for {timeframe}")
            return None
        
        # Build features
        logger.info("Step 3: Building features...")
        feature_config = config.get('features', {})
        horizon_bars = feature_config.get('target', {}).get('horizon_bars', {}).get(timeframe, 12)
        
        features_df, target, metadata = build_features(
            df,
            timeframe=timeframe,
            feature_config=feature_config,
            target_horizon_bars=horizon_bars
        )
        
        if features_df is None or len(features_df) == 0:
            logger.error(f"Feature building failed for {timeframe}")
            return None
        
        # Add target column
        features_df['target'] = target.values
        df = features_df  # Use features_df as df for rest of pipeline
        
        # Split data
        logger.info("Step 4: Splitting data...")
        split_config = config.get('split', {})
        train_ratio = split_config.get('train', 0.7)
        val_ratio = split_config.get('val', 0.15)
        test_ratio = split_config.get('test', 0.15)
        
        splits = split_time_series(df, train_ratio, val_ratio, test_ratio)
        train_data = splits['train']
        val_data = splits['val']
        test_data = splits['test']
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Run Optuna HPO
        logger.info("Step 5: Running Optuna HPO...")
        config['run_id'] = run_id  # Pass run_id to objective
        
        hpo_results = run_optuna_hpo(
            timeframe=timeframe,
            train_data=train_data,
            val_data=val_data,
            config=config,
            run_id=run_id,
            coin=coin,
            n_trials=hpo_trials,
            timeout_minutes=hpo_timeout,
            resume=False
        )
        
        best_params = hpo_results['best_params']
        logger.info(f"✓ HPO completed for {timeframe}")
        logger.info(f"  Best params: {best_params}")
        logger.info(f"  Best value: {hpo_results.get('best_value', 'N/A')}")
        
        # Save HPO results
        save_json({
            'timeframe': timeframe,
            'best_params': best_params,
            'hpo_results': hpo_results,
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data)
        }, artifacts_dir / 'hpo_summary.json')
        
        return {
            'timeframe': timeframe,
            'best_params': best_params,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'hpo_results': hpo_results
        }
        
    except Exception as e:
        logger.error(f"HPO failed for {timeframe}: {e}")
        logger.error(traceback.format_exc())
        return None

def process_timeframe_training(
    timeframe: str,
    best_params: Dict,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Dict,
    run_id: str
) -> Optional[Dict]:
    """
    Train final model for one timeframe.
    
    Returns:
        Dict with training results or None if failed
    """
    coin = config['coin']
    logger = logging.getLogger(f"btc_train_{timeframe}")
    
    try:
        logger.info("=" * 60)
        logger.info(f"BTC {timeframe} - FINAL TRAINING")
        logger.info("=" * 60)
        
        # Setup paths
        artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id / timeframe
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Train final model
        logger.info("Training final model...")
        train_results = train_final_model(
            timeframe=timeframe,
            train_data=train_data,
            val_data=val_data,
            best_params=best_params,
            config=config,
            run_id=run_id,
            coin=coin
        )
        
        logger.info(f"✓ Model trained: {train_results['model_path']}")
        
        # Evaluate on test
        logger.info("Evaluating on test set...")
        device = get_device(config.get('device', 'cuda'))
        
        # Create test dataset
        test_dict = {coin: test_data}
        temp_model = TFTModel(
            prediction_horizon=config['model']['max_decoder_length'],
            max_encoder_length=config['model']['max_encoder_length'],
            max_decoder_length=config['model']['max_decoder_length'],
            hidden_size=best_params['hidden_size'],
            device=device
        )
        test_dataset = temp_model.create_dataset(test_dict, target='target')
        
        # Load model
        model = TFTModel(
            prediction_horizon=config['model']['max_decoder_length'],
            max_encoder_length=config['model']['max_encoder_length'],
            max_decoder_length=config['model']['max_decoder_length'],
            hidden_size=best_params['hidden_size'],
            device=device
        )
        model.load(train_results['model_path'], test_dataset)
        
        # Get predictions
        test_predictions = []
        test_targets = []
        
        test_loader = test_dataset.to_dataloader(
            train=False,
            batch_size=best_params['batch_size'] * 10,
            num_workers=0  # Windows safe
        )
        
        model.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                # PyTorch Forecasting returns y as (targets, weights) tuple
                if isinstance(y, tuple):
                    y = y[0]  # Get targets only
                x = {k: v.to(device) for k, v in x.items()}
                y = y.to(device)
                
                pred = model.model(x)
                # CRITICAL FIX: Extract prediction tensor properly
                pred_tensor = model._extract_prediction_tensor(pred)
                test_predictions.append(pred_tensor.cpu().numpy())
                test_targets.append(y.cpu().numpy())
        
        test_predictions = np.concatenate(test_predictions, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        
        # PROFIT-FIRST METRICS: Compute with leverage from config
        leverage = config.get('backtest', {}).get('max_leverage', 5.0)
        metrics = compute_metrics(test_targets, test_predictions, leverage=leverage)
        save_json(metrics, artifacts_dir / 'metrics_test.json')
        
        # Log PROFIT metrics (priority)
        logger.info(f"✓ PROFIT METRICS:")
        logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.4f} ({metrics.get('directional_accuracy', 0)*100:.2f}%)")
        logger.info(f"  Cumulative PnL (5x): {metrics.get('pnl_cumulative', 0):.4f}")
        logger.info(f"  Sharpe Ratio: {metrics.get('pnl_sharpe', 0):.4f}")
        logger.info(f"  Win Rate: {metrics.get('pnl_win_rate', 0):.4f} ({metrics.get('pnl_win_rate', 0)*100:.2f}%)")
        logger.info(f"✓ Legacy metrics: MAE={metrics.get('mae', 0):.6f}, RMSE={metrics.get('rmse', 0):.6f}, R2={metrics.get('r2', 0):.4f}")
        
        # Save predictions
        preds_df = pd.DataFrame({
            'target': test_targets.flatten(),
            'prediction': test_predictions.flatten()
        })
        preds_df.to_parquet(artifacts_dir / 'preds_test.parquet', index=False)
        
        # Backtest
        logger.info("Running backtest...")
        backtest_results = run_backtest_on_test(
            test_data=test_data,
            predictions=test_predictions,
            config=config,
            timeframe=timeframe
        )
        
        save_json(backtest_results, artifacts_dir / 'backtest_metrics.json')
        logger.info(f"✓ Backtest: Return={backtest_results['total_return_pct']:.2f}%, Sharpe={backtest_results['sharpe_ratio']:.2f}")
        
        return {
            'timeframe': timeframe,
            'status': 'success',
            'test_metrics': metrics,
            'backtest_metrics': backtest_results,
            'model_path': str(train_results['model_path'])
        }
        
    except Exception as e:
        logger.error(f"Training failed for {timeframe}: {e}")
        logger.error(traceback.format_exc())
        return {
            'timeframe': timeframe,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="BTC Pipeline - 3 Timeframes")
    parser.add_argument('--config', type=str, default='config/train.yaml',
                       help='Path to config file')
    parser.add_argument('--hpo_trials', type=int, default=50,
                       help='Number of Optuna trials per timeframe')
    parser.add_argument('--hpo_timeout', type=int, default=None,
                       help='HPO timeout in minutes')
    parser.add_argument('--max_parallel_training', type=int, default=2,
                       help='Max parallel training jobs (default: 2 for CPU safety)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_all_configs(args.config)
    run_id = generate_run_id()
    
    # Setup global logging
    artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(artifacts_dir)
    
    global_logger = logging.getLogger("btc_pipeline")
    global_logger.info("=" * 60)
    global_logger.info("BTC PIPELINE - 3 TIMEFRAMES")
    global_logger.info("=" * 60)
    global_logger.info(f"Run ID: {run_id}")
    global_logger.info(f"Coin: {config['coin']}")
    global_logger.info(f"Timeframes: {config['timeframes']}")
    global_logger.info(f"HPO Trials: {args.hpo_trials}")
    global_logger.info(f"Max Parallel Training: {args.max_parallel_training}")
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    global_logger.info(f"Seed: {seed}")
    
    timeframes = config['timeframes']
    hpo_results = {}
    
    # ====================================================================
    # PHASE 1: SEQUENTIAL OPTUNA HPO (one at a time to avoid CPU overload)
    # ====================================================================
    global_logger.info("\n" + "=" * 60)
    global_logger.info("PHASE 1: OPTUNA HPO (Sequential)")
    global_logger.info("=" * 60)
    
    for timeframe in timeframes:
        global_logger.info(f"\n>>> Starting HPO for {timeframe}...")
        result = process_timeframe_hpo(
            timeframe=timeframe,
            config=config,
            run_id=run_id,
            hpo_trials=args.hpo_trials,
            hpo_timeout=args.hpo_timeout
        )
        
        if result:
            hpo_results[timeframe] = result
            global_logger.info(f"✓ {timeframe} HPO completed")
        else:
            global_logger.error(f"✗ {timeframe} HPO failed")
            hpo_results[timeframe] = None
    
    # ====================================================================
    # PHASE 2: PARALLEL TRAINING (limited parallelism for CPU safety)
    # ====================================================================
    global_logger.info("\n" + "=" * 60)
    global_logger.info("PHASE 2: FINAL TRAINING (Parallel)")
    global_logger.info("=" * 60)
    
    training_tasks = []
    for timeframe in timeframes:
        if timeframe in hpo_results and hpo_results[timeframe]:
            training_tasks.append({
                'timeframe': timeframe,
                'best_params': hpo_results[timeframe]['best_params'],
                'train_data': hpo_results[timeframe]['train_data'],
                'val_data': hpo_results[timeframe]['val_data'],
                'test_data': hpo_results[timeframe]['test_data']
            })
    
    all_summaries = {}
    
    # Parallel training with limited workers
    with ThreadPoolExecutor(max_workers=args.max_parallel_training) as executor:
        futures = {}
        for task in training_tasks:
            future = executor.submit(
                process_timeframe_training,
                task['timeframe'],
                task['best_params'],
                task['train_data'],
                task['val_data'],
                task['test_data'],
                config,
                run_id
            )
            futures[future] = task['timeframe']
        
        for future in as_completed(futures):
            timeframe = futures[future]
            try:
                result = future.result()
                all_summaries[timeframe] = result
                if result and result.get('status') == 'success':
                    global_logger.info(f"✓ {timeframe} training completed")
                else:
                    global_logger.error(f"✗ {timeframe} training failed")
            except Exception as e:
                timeframe = futures[future]
                global_logger.error(f"✗ {timeframe} training exception: {e}")
                all_summaries[timeframe] = {
                    'timeframe': timeframe,
                    'status': 'failed',
                    'error': str(e)
                }
    
    # ====================================================================
    # FINAL SUMMARY
    # ====================================================================
    global_summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'coin': config['coin'],
        'timeframes': timeframes,
        'hpo_trials': args.hpo_trials,
        'seed': seed,
        'results': all_summaries,
        'successful_timeframes': [tf for tf, s in all_summaries.items() if s and s.get('status') == 'success'],
        'failed_timeframes': [tf for tf, s in all_summaries.items() if not s or s.get('status') == 'failed']
    }
    
    save_json(global_summary, artifacts_dir / 'summary.json')
    
    # Print summary
    print("\n" + "=" * 60)
    print("BTC PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nRun ID: {run_id}")
    print(f"Summary: {artifacts_dir / 'summary.json'}\n")
    
    for tf in timeframes:
        if tf in all_summaries and all_summaries[tf]:
            s = all_summaries[tf]
            print(f"{tf}:")
            if s.get('status') == 'success':
                if 'test_metrics' in s:
                    print(f"  Test MAE: {s['test_metrics']['mae']:.6f}")
                    print(f"  Test R2: {s['test_metrics']['r2']:.4f}")
                if 'backtest_metrics' in s:
                    bt = s['backtest_metrics']
                    print(f"  Return: {bt['total_return_pct']:.2f}%")
                    print(f"  Sharpe: {bt['sharpe_ratio']:.2f}")
                    print(f"  Max DD: {bt['max_drawdown_pct']:.2f}%")
            else:
                print(f"  Status: {s.get('status', 'unknown')}")
                if 'error' in s:
                    print(f"  Error: {s['error']}")
        else:
            print(f"{tf}: FAILED")
        print()

if __name__ == "__main__":
    main()

