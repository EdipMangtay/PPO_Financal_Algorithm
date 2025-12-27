"""
Orchestrator - Train and backtest all timeframes (LEGACY - use run_all_new.py)
"""

import sys
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime
import logging
import json

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.doctor import run_doctor
from training.train_one import train_one_timeframe
from signals.signal_15m import Signal15m
from signals.signal_1h import Signal1h
from signals.signal_4h import Signal4h
from backtest.engine import BacktestEngine, save_backtest_results
from backtest.plots import plot_equity_curve
from data.loader_new import load_or_resample
from features.build_features import build_features
from data.validators import validate_dataframe

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NOTE: This is the legacy orchestrator. For full HPO pipeline, use run_all_new.py

def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_run_id() -> str:
    """Generate unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_all(config_path: str, force: bool = False):
    """Run complete pipeline: train + backtest for all timeframes."""
    
    # Run doctor check
    logger.info("Running environment doctor...")
    doctor_results = run_doctor(verbose=True)
    
    if not doctor_results['all_checks_passed'] and not force:
        logger.error("Doctor checks failed. Use --force to continue anyway.")
        sys.exit(1)
    
    # Load configs
    logger.info(f"Loading config from {config_path}")
    train_config = load_config(config_path)
    feature_config = load_config("config/features.yaml")
    paths_config = load_config("config/paths.yaml")
    
    # Generate run ID
    run_id = generate_run_id()
    logger.info(f"Run ID: {run_id}")
    
    # Setup paths
    artifacts_dir = Path(paths_config['artifacts_dir']) / run_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Results summary
    summary = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': train_config,
        'timeframes': {},
    }
    
    # Process each timeframe
    for timeframe in train_config['timeframes']:
        logger.info("=" * 60)
        logger.info(f"Processing {timeframe} timeframe")
        logger.info("=" * 60)
        
        tf_summary = {}
        
        try:
            # Train model
            logger.info(f"Training {timeframe} model...")
            train_results = train_one_timeframe(
                config=train_config,
                timeframe=timeframe,
                run_id=run_id,
                coin=train_config['coin']
            )
            
            tf_summary['training'] = {
                'model_path': train_results['model_path'],
                'metrics': train_results['metrics'],
            }
            
            # Load signal generator
            logger.info(f"Loading {timeframe} signal generator...")
            if timeframe == '15m':
                signal_gen = Signal15m(train_results['model_path'])
            elif timeframe == '1h':
                signal_gen = Signal1h(train_results['model_path'])
            elif timeframe == '4h':
                signal_gen = Signal4h(train_results['model_path'])
            else:
                raise ValueError(f"Unknown timeframe: {timeframe}")
            
            # Load test data
            logger.info(f"Loading test data for {timeframe}...")
            df_raw = load_or_resample(
                coin=train_config['coin'],
                target_timeframe=timeframe,
                date_range=train_config['date_range'],
                data_dir=paths_config['data_dir']
            )
            
            # Build features
            logger.info(f"Building features for {timeframe}...")
            features_df, target, metadata = build_features(
                df_raw,
                timeframe=timeframe,
                feature_config=feature_config,
                target_horizon_bars=feature_config['target']['horizon_bars'][timeframe]
            )
            
            # Split to get test set
            from training.train_one import split_time_series
            splits = split_time_series(
                features_df,
                train_ratio=train_config['split']['train'],
                val_ratio=train_config['split']['val'],
                test_ratio=train_config['split']['test']
            )
            
            test_features = splits['test']
            
            # Generate signals
            logger.info(f"Generating signals for {timeframe}...")
            signals_df = signal_gen.generate_signals(test_features)
            
            # Run backtest
            logger.info(f"Running backtest for {timeframe}...")
            engine = BacktestEngine(
                initial_balance=10000.0,
                fee_rate=0.0004,
                slippage=0.0005,
            )
            
            # Align signals with test data
            test_df = df_raw.iloc[-len(test_features):].copy()
            test_signals = signals_df['signal'].values
            
            backtest_results = engine.run_backtest(
                df=test_df,
                signals=pd.Series(test_signals, index=test_df.index),
                price_col='close',
                timestamp_col='timestamp'
            )
            
            # Save backtest results
            tf_artifacts_dir = artifacts_dir / timeframe
            save_backtest_results(backtest_results, tf_artifacts_dir, timeframe)
            
            # Plot equity curve
            plot_equity_curve(
                backtest_results['equity_curve'],
                tf_artifacts_dir / 'equity_curve.png',
                title=f"{timeframe} Equity Curve"
            )
            
            tf_summary['backtest'] = backtest_results['metrics']
            tf_summary['num_trades'] = len(backtest_results['trades'])
            
            logger.info(f"✓ {timeframe} completed successfully")
            logger.info(f"  Total Return: {backtest_results['metrics']['total_return_pct']:.2f}%")
            logger.info(f"  Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
            logger.info(f"  Max Drawdown: {backtest_results['metrics']['max_drawdown_pct']:.2f}%")
            logger.info(f"  Win Rate: {backtest_results['metrics']['win_rate_pct']:.1f}%")
            logger.info(f"  Total Trades: {backtest_results['metrics']['total_trades']}")
            
        except Exception as e:
            logger.error(f"✗ {timeframe} failed: {e}", exc_info=True)
            tf_summary['error'] = str(e)
        
        summary['timeframes'][timeframe] = tf_summary
    
    # Save summary
    summary_path = artifacts_dir / 'run_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    for tf, tf_data in summary['timeframes'].items():
        if 'error' not in tf_data:
            print(f"\n{tf}:")
            if 'backtest' in tf_data:
                bt = tf_data['backtest']
                print(f"  Return: {bt['total_return_pct']:.2f}%")
                print(f"  Sharpe: {bt['sharpe_ratio']:.2f}")
                print(f"  Max DD: {bt['max_drawdown_pct']:.2f}%")
                print(f"  Win Rate: {bt['win_rate_pct']:.1f}%")
                print(f"  Trades: {bt['total_trades']}")
        else:
            print(f"\n{tf}: ERROR - {tf_data['error']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and backtest all timeframes")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Config file path")
    parser.add_argument("--force", action="store_true", help="Force run even if doctor fails")
    args = parser.parse_args()
    
    run_all(args.config, force=args.force)

