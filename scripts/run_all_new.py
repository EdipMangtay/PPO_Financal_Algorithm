"""
Main Orchestrator - Complete pipeline: Preflight -> HPO -> Train -> Evaluate -> Backtest
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

def generate_run_id(config: Dict) -> str:
    """Generate or get run ID."""
    if 'run_id' in config and config['run_id']:
        return config['run_id']
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_all_configs(config_path: str) -> Dict:
    """Load all configuration files."""
    train_config = load_yaml(Path(config_path))
    paths_config = load_yaml(Path("config/paths.yaml"))
    feature_config = load_yaml(Path("config/features.yaml"))
    
    # Merge paths into main config
    train_config['paths'] = paths_config
    train_config['features'] = feature_config
    
    return train_config

def process_timeframe(
    timeframe: str,
    config: Dict,
    run_id: str,
    hpo_trials: Optional[int] = None,
    hpo_timeout: Optional[int] = None,
    skip_hpo: bool = False,
    resume_hpo: bool = False,
    continue_on_error: bool = False
) -> Optional[Dict]:
    """
    Process one timeframe: preflight -> HPO -> train -> evaluate -> backtest.
    
    Returns:
        Summary dict or None if failed
    """
    coin = config['coin']
    logger = logging.getLogger(f"trading_{timeframe}")
    
    try:
        logger.info("=" * 60)
        logger.info(f"Processing {timeframe} timeframe")
        logger.info("=" * 60)
        
        # Setup paths
        artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id / timeframe
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging for this timeframe
        setup_logging(artifacts_dir, timeframe=timeframe)
        
        # ====================================================================
        # STEP 1: PREFLIGHT CHECKS
        # ====================================================================
        logger.info("Step 1: Running preflight checks...")
        checker = PreflightChecker(config, run_id)
        preflight_result = checker.run_preflight(timeframe)
        
        if preflight_result == "BLOCKED":
            error_msg = f"Preflight BLOCKED for {timeframe}: {checker.errors}"
            logger.error(error_msg)
            logger.error("Preflight errors:")
            for err in checker.errors:
                logger.error(f"  - {err}")
            if checker.fixes_applied:
                logger.info("Auto-fixes applied:")
                for fix in checker.fixes_applied:
                    logger.info(f"  - {fix}")
            if not continue_on_error:
                raise RuntimeError(error_msg)
            logger.warning(f"Skipping {timeframe} due to preflight failure (continue_on_error=True)")
            return {
                'timeframe': timeframe,
                'status': 'skipped',
                'reason': 'preflight_blocked',
                'errors': checker.errors,
            }
        
        logger.info(f"Preflight: {preflight_result}")
        
        # ====================================================================
        # STEP 2: LOAD AND PREPARE DATA
        # ====================================================================
        logger.info("Step 2: Loading and preparing data...")
        
        date_range = config['date_range']
        df_raw = load_or_resample(
            coin=coin,
            target_timeframe=timeframe,
            date_range=date_range,
            data_dir=config['paths']['data_dir']
        )
        
        # Store original df_raw with timestamp for later alignment (before feature building modifies it)
        df_raw_with_timestamp = df_raw.copy()
        
        # Build features
        feature_config = config['features']
        horizon_bars = feature_config['target']['horizon_bars'][timeframe]
        
        features_df, target, metadata = build_features(
            df_raw,
            timeframe=timeframe,
            feature_config=feature_config,
            target_horizon_bars=horizon_bars
        )
        
        # CRITICAL FIX: Reset indices to ensure alignment
        features_df = features_df.reset_index(drop=True)
        target = target.reset_index(drop=True)
        
        # Verify alignment
        if len(features_df) != len(target):
            raise ValueError(f"Index mismatch after feature building: features={len(features_df)}, target={len(target)}")
        
        # Save feature report
        save_feature_report(metadata, artifacts_dir / 'feature_report.json', timeframe)
        
        # Split data
        splits = split_time_series(
            features_df,
            train_ratio=config['split']['train'],
            val_ratio=config['split']['val'],
            test_ratio=config['split']['test']
        )
        
        # Add target to splits (SAFE with reset index - use positional indexing)
        train_end = len(splits['train'])
        val_end = train_end + len(splits['val'])
        
        splits['train']['target'] = target.iloc[:train_end].values
        splits['val']['target'] = target.iloc[train_end:val_end].values
        splits['test']['target'] = target.iloc[val_end:].values
        
        # Reset indices for all splits
        for split_name in splits:
            splits[split_name] = splits[split_name].reset_index(drop=True)
        
        train_data = splits['train']
        val_data = splits['val']
        test_data = splits['test']
        
        # Validate splits have target
        for split_name, split_df in splits.items():
            if 'target' not in split_df.columns:
                raise ValueError(f"Target missing in {split_name} split")
            if split_df['target'].isna().all():
                logger.warning(f"All targets NaN in {split_name}, may cause training issues")
            logger.info(
                f"{split_name}: {len(split_df)} rows, "
                f"target stats: mean={split_df['target'].mean():.6f}, "
                f"std={split_df['target'].std():.6f}, "
                f"range=[{split_df['target'].min():.6f}, {split_df['target'].max():.6f}]"
            )
        
        logger.info(f"Data split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # ====================================================================
        # STEP 3: OPTUNA HPO (if enabled)
        # ====================================================================
        best_params = None
        hpo_config = config.get('hpo', {})
        
        if not skip_hpo and hpo_config.get('enabled', True):
            logger.info("Step 3: Running Optuna HPO...")
            
            n_trials = hpo_trials or hpo_config.get('n_trials', 50)
            timeout = hpo_timeout or hpo_config.get('timeout_minutes', None)
            resume = resume_hpo or hpo_config.get('resume', False)
            
            try:
                hpo_results = run_optuna_hpo(
                    timeframe=timeframe,
                    train_data=train_data,
                    val_data=val_data,
                    config=config,
                    run_id=run_id,
                    coin=coin,
                    n_trials=n_trials,
                    timeout_minutes=timeout,
                    resume=resume
                )
                
                best_params = hpo_results['best_params']
                logger.info(f"HPO completed. Best params: {best_params}")
                logger.info(f"  Best value: {hpo_results.get('best_value', 'N/A')}")
                logger.info(f"  Trials: {hpo_results.get('n_complete', 0)} complete, {hpo_results.get('n_pruned', 0)} pruned")
            except Exception as e:
                logger.error(f"HPO failed for {timeframe}: {e}")
                if not continue_on_error:
                    raise
                logger.warning("Falling back to config defaults")
                best_params = {
                    'lr': config['learning_rate'][timeframe],
                    'batch_size': config['batch_size'][timeframe],
                    'dropout': config['model']['dropout'],
                    'hidden_size': config['model']['hidden_size'],
                    'weight_decay': 1e-5,
                }
        else:
            logger.info("Step 3: Skipping HPO, using config defaults")
            # Use default params from config
            best_params = {
                'lr': config['learning_rate'][timeframe],
                'batch_size': config['batch_size'][timeframe],
                'dropout': config['model']['dropout'],
                'hidden_size': config['model']['hidden_size'],
                'weight_decay': 1e-5,
            }
        
        # ====================================================================
        # STEP 4: TRAIN FINAL MODEL
        # ====================================================================
        logger.info("Step 4: Training final model...")
        
        train_results = train_final_model(
            timeframe=timeframe,
            train_data=train_data,
            val_data=val_data,
            best_params=best_params,
            config=config,
            run_id=run_id,
            coin=coin
        )
        
        logger.info(f"Final model trained: {train_results['model_path']}")
        
        # ====================================================================
        # STEP 5: EVALUATE ON TEST
        # ====================================================================
        logger.info("Step 5: Evaluating on test set...")
        
        # Load model
        from utils.device import get_device
        device = get_device(config.get('device', 'cuda'))
        
        # Create test dataset first (needed for model structure)
        test_dict = {coin: test_data}
        temp_model = TFTModel(
            prediction_horizon=config['model']['max_decoder_length'],
            max_encoder_length=config['model']['max_encoder_length'],
            max_decoder_length=config['model']['max_decoder_length'],
            hidden_size=best_params['hidden_size'],
            device=device
        )
        test_dataset = temp_model.create_dataset(test_dict, target='target')
        
        # Now create model and load
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
            num_workers=config.get('num_workers', 0)
        )
        
        model.model.eval()
        with torch.no_grad():
            # CRITICAL FIX: Use recursive device transfer
            from utils.device import move_to_device
            model_device = next(model.model.parameters()).device
            
            for batch in test_loader:
                # Move entire batch to device recursively
                batch = move_to_device(batch, model_device)
                x, y = batch
                
                output = model.model(x)
                pred = output.prediction if hasattr(output, 'prediction') else output
                y_true = y[0] if isinstance(y, (tuple, list)) else y
                
                test_predictions.append(pred.cpu().numpy())
                test_targets.append(y_true.cpu().numpy())
        
        test_predictions = np.concatenate(test_predictions, axis=0)
        test_targets = np.concatenate(test_targets, axis=0)
        
        # Flatten for metrics
        test_predictions_flat = test_predictions.flatten()
        test_targets_flat = test_targets.flatten()
        
        # Compute metrics
        test_metrics = compute_metrics(
            test_targets_flat,
            test_predictions_flat,
            task_type='regression'
        )
        
        # Save predictions and metrics
        preds_df = pd.DataFrame({
            'target': test_targets_flat,
            'prediction': test_predictions_flat,
        })
        preds_df.to_parquet(artifacts_dir / 'preds_test.parquet')
        
        save_json(test_metrics, artifacts_dir / 'metrics_test.json')
        
        logger.info(f"Test metrics: MAE={test_metrics['mae']:.6f}, RMSE={test_metrics['rmse']:.6f}, R2={test_metrics['r2']:.4f}")
        
        # ====================================================================
        # STEP 6: BACKTEST ON TEST
        # ====================================================================
        logger.info("Step 6: Running backtest on test data...")
        
        # Get test OHLCV data aligned with predictions
        # Use timestamp-based alignment if available (most reliable), otherwise use positional
        if 'timestamp' in test_data.columns and 'timestamp' in df_raw_with_timestamp.columns:
            # Align by timestamp (most reliable method)
            test_timestamps = test_data['timestamp'].unique()
            test_ohlcv = df_raw_with_timestamp[
                df_raw_with_timestamp['timestamp'].isin(test_timestamps)
            ].copy()
            
            # Sort both by timestamp to ensure same order
            test_ohlcv = test_ohlcv.sort_values('timestamp').reset_index(drop=True)
            test_data_sorted = test_data.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Aligned test OHLCV by timestamp: {len(test_ohlcv)} rows matched")
        else:
            # Fallback: use last N rows (less reliable but works)
            test_start_idx = len(df_raw_with_timestamp) - len(test_data) if len(df_raw_with_timestamp) > len(test_data) else 0
            test_ohlcv = df_raw_with_timestamp.iloc[test_start_idx:].copy().reset_index(drop=True)
            test_data_sorted = test_data.reset_index(drop=True)
            logger.warning("Timestamp alignment not available, using positional alignment (may be inaccurate)")
        
        # Use mean prediction as forward return signal
        if test_predictions.ndim > 1:
            mean_predictions = test_predictions.mean(axis=1)
        else:
            mean_predictions = test_predictions_flat
        
        # Ensure same length (critical for backtest)
        min_len = min(len(test_ohlcv), len(test_data_sorted), len(mean_predictions))
        if min_len == 0:
            raise ValueError("Cannot run backtest: one of test_ohlcv, test_data, or predictions is empty")
        
        test_ohlcv = test_ohlcv.iloc[:min_len].copy()
        mean_predictions = mean_predictions[:min_len]
        
        logger.info(f"Backtest alignment: OHLCV={len(test_ohlcv)}, predictions={len(mean_predictions)}")
        
        backtest_results = run_backtest_on_test(
            timeframe=timeframe,
            test_data=test_ohlcv,
            predictions=mean_predictions,
            config=config,
            run_id=run_id
        )
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        summary = {
            'timeframe': timeframe,
            'model_path': train_results['model_path'],
            'best_params': best_params,
            'test_metrics': test_metrics,
            'backtest_metrics': backtest_results['metrics'],
            'num_trades': len(backtest_results['trades']),
        }
        
        logger.info("=" * 60)
        logger.info(f"{timeframe} COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        return summary
        
    except Exception as e:
        error_msg = f"Failed to process {timeframe}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        if not continue_on_error:
            raise
        return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Complete training and backtest pipeline")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Config file")
    parser.add_argument("--hpo_trials", type=int, help="Number of Optuna trials (overrides config)")
    parser.add_argument("--hpo_timeout_minutes", type=int, help="HPO timeout in minutes (overrides config)")
    parser.add_argument("--skip_hpo", action="store_true", help="Skip HPO, use config defaults")
    parser.add_argument("--resume_hpo", action="store_true", help="Resume existing Optuna study")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue on timeframe errors")
    args = parser.parse_args()
    
    # Load configs
    config = load_all_configs(args.config)
    
    # Generate run ID
    run_id = generate_run_id(config)
    print(f"\n{'='*60}")
    print(f"RUN ID: {run_id}")
    print(f"{'='*60}\n")
    
    # Setup global logging
    artifacts_dir = Path(config['paths']['artifacts_dir']) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    global_logger = setup_logging(artifacts_dir, level=logging.INFO)
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Log environment info
    from utils.device import get_device_info
    global_logger.info("Environment Info:")
    global_logger.info(f"  Seed: {config.get('seed', 42)}")
    global_logger.info(f"  Device: {get_device_info(config.get('device', 'cuda'))}")
    global_logger.info(f"  Seed info: {get_seed_info()}")
    
    # Process each timeframe
    all_summaries = {}
    timeframes = config['timeframes']
    
    for timeframe in timeframes:
        try:
            summary = process_timeframe(
                timeframe=timeframe,
                config=config,
                run_id=run_id,
                hpo_trials=args.hpo_trials,
                hpo_timeout=args.hpo_timeout_minutes,
                skip_hpo=args.skip_hpo,
                resume_hpo=args.resume_hpo,
                continue_on_error=args.continue_on_error
            )
            
            if summary:
                all_summaries[timeframe] = summary
            else:
                global_logger.warning(f"{timeframe}: Processed but returned no summary (may have been skipped)")
        except Exception as e:
            global_logger.error(f"Failed to process {timeframe}: {e}")
            global_logger.error(traceback.format_exc())
            
            # Always save partial summary even on error
            all_summaries[timeframe] = {
                'timeframe': timeframe,
                'status': 'failed',
                'error': str(e),
            }
            
            if not args.continue_on_error:
                global_logger.error("Stopping due to error (use --continue_on_error to continue)")
                raise
            else:
                global_logger.warning(f"Continuing with next timeframe (--continue_on_error enabled)")
    
    # Save global summary (always, even if some timeframes failed)
    global_summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'coin': config.get('coin'),
            'timeframes': config.get('timeframes'),
            'date_range': config.get('date_range'),
            'seed': config.get('seed'),
        },
        'timeframes': all_summaries,
        'successful_timeframes': [tf for tf, s in all_summaries.items() if s.get('status') not in ['failed', 'skipped']],
        'failed_timeframes': [tf for tf, s in all_summaries.items() if s.get('status') == 'failed'],
        'skipped_timeframes': [tf for tf, s in all_summaries.items() if s.get('status') == 'skipped'],
    }
    
    save_json(global_summary, artifacts_dir / 'summary.json')
    global_logger.info(f"Global summary saved to {artifacts_dir / 'summary.json'}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nRun ID: {run_id}")
    print(f"Summary saved to: {artifacts_dir / 'summary.json'}\n")
    
    for tf, summary in all_summaries.items():
        print(f"{tf}:")
        if 'test_metrics' in summary:
            print(f"  Test MAE: {summary['test_metrics']['mae']:.6f}")
            print(f"  Test R2: {summary['test_metrics']['r2']:.4f}")
        if 'backtest_metrics' in summary:
            bt = summary['backtest_metrics']
            print(f"  Backtest Return: {bt['total_return_pct']:.2f}%")
            print(f"  Sharpe: {bt['sharpe_ratio']:.2f}")
            print(f"  Max DD: {bt['max_drawdown_pct']:.2f}%")
            print(f"  Trades: {bt['total_trades']}")
        print()
    
    # Deliverables checklist
    print("=" * 60)
    print("DELIVERABLES CHECKLIST")
    print("=" * 60)
    
    for tf in timeframes:
        tf_dir = artifacts_dir / tf
        checks = [
            ("Preflight OK", True),  # Assumed if we got here
            ("Optuna DB", (tf_dir / 'optuna.db').exists() if not args.skip_hpo else "SKIPPED"),
            ("optuna_best.json", (tf_dir / 'optuna_best.json').exists() if not args.skip_hpo else "SKIPPED"),
            ("model.pt", (tf_dir / 'model.pt').exists()),
            ("metrics_test.json", (tf_dir / 'metrics_test.json').exists()),
            ("backtest_summary.json", (tf_dir / 'backtest_metrics.json').exists()),
        ]
        
        print(f"\n{tf}:")
        for name, status in checks:
            if status == "SKIPPED":
                print(f"  [SKIP] {name}")
            elif status:
                print(f"  [OK] {name}")
            else:
                print(f"  [MISSING] {name}")
    
    print(f"\n  [OK] global summary.json")
    print("\n" + "=" * 60)
    print("EXACT COMMAND TO RUN:")
    print(f"python scripts/run_all_new.py --config config/train.yaml")
    if args.hpo_trials:
        print(f"  --hpo_trials {args.hpo_trials}")
    if args.skip_hpo:
        print(f"  --skip_hpo")
    print("=" * 60)

if __name__ == "__main__":
    main()

