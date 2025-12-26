"""
GOD_LEVEL_TRADER_FINAL - Main Pipeline
Auto-Feature Engineering Edition with Dynamic Feature Selection
"""

# ====================================================================
# CRITICAL: Numpy compatibility patch for pandas-ta (Python 3.13)
# ====================================================================
import numpy as np
# Monkey patch for pandas-ta compatibility with numpy 2.x
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'bool_'):
    try:
        np.bool_ = np.bool
    except AttributeError:
        # numpy 2.0+ removed np.bool, use bool instead
        np.bool_ = bool

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TOP_20_COINS, SCALP_TIMEFRAME, SWING_TIMEFRAME
from data.loader import DataLoader
from data_engine.features import FeatureGenerator
from tuning.optimizer import TwoLayerOptimizer, load_feature_config
from models.tft import TFTModel
from models.ppo import PPOTradingAgent
from env.trading_env import TradingEnv
from trainer import Trainer
from run_bot import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Main pipeline for dynamic feature engineering and training.
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.data_loader = DataLoader()
        self.feature_generator = FeatureGenerator()
        self.trainer = Trainer()
        
    async def step1_download_data(
        self,
        days: int = 90,
        coins: List[str] = None,
        timeframe: str = SCALP_TIMEFRAME
    ) -> Dict[str, Dict]:
        """
        Step 1: Download extensive history for all coins.
        
        Returns:
            Dictionary: {coin: {timeframe: DataFrame}}
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DATA DOWNLOAD")
        logger.info("=" * 60)
        
        if coins is None:
            coins = TOP_20_COINS
        
        logger.info(f"Downloading {days} days of {timeframe} data for {len(coins)} coins...")
        
        # Download data
        data = await self.data_loader.fetch_recent(days=days, timeframe=timeframe, coins=coins)
        
        logger.info(f"Downloaded data for {len(data)} coins")
        
        return {coin: {timeframe: df} for coin, df in data.items()}
    
    def step2_feature_race(
        self,
        data_dict: Dict[str, Dict],
        n_trials_per_coin: int = 100,
        timeout_per_coin: int = None
    ) -> Dict[str, Dict]:
        """
        Step 2: The "Feature Race" - Run Optuna for each coin to find its DNA.
        
        Args:
            data_dict: {coin: {timeframe: DataFrame}}
            n_trials_per_coin: Number of Optuna trials per coin
            timeout_per_coin: Timeout per coin in seconds
        
        Returns:
            Dictionary of best feature configs: {coin: config}
        """
        logger.info("=" * 60)
        logger.info("STEP 2: THE FEATURE RACE")
        logger.info("=" * 60)
        logger.info(f"Running {n_trials_per_coin} trials per coin to find optimal features...")
        
        best_configs = {}
        
        for coin, timeframes in data_dict.items():
            for timeframe, df in timeframes.items():
                logger.info(f"\nOptimizing features for {coin} {timeframe}...")
                
                try:
                    # Create optimizer
                    optimizer = TwoLayerOptimizer(
                        coin=coin,
                        timeframe=timeframe,
                        data=df,
                        n_trials=n_trials_per_coin,
                        timeout=timeout_per_coin
                    )
                    
                    # Run optimization
                    study, best_config = optimizer.optimize()
                    
                    # Save config
                    optimizer.save_config(best_config)
                    
                    best_configs[f"{coin}_{timeframe}"] = best_config
                    
                    logger.info(f"✓ Completed {coin} {timeframe}")
                    logger.info(f"  Selected {len(best_config['selected_features'])} features")
                    logger.info(f"  Best Sharpe: {best_config['performance']['sharpe_ratio']:.4f}")
                    
                except Exception as e:
                    logger.error(f"✗ Failed {coin} {timeframe}: {e}")
                    continue
        
        logger.info(f"\nFeature race completed for {len(best_configs)} coin/timeframe combinations")
        return best_configs
    
    def step3_deep_training(
        self,
        data_dict: Dict[str, Dict],
        feature_configs: Dict[str, Dict],
        timeframe: str = SCALP_TIMEFRAME
    ):
        """
        Step 3: Deep Training - Train TFT + PPO using only optimized features.
        
        Args:
            data_dict: {coin: {timeframe: DataFrame}}
            feature_configs: Best feature configs from Step 2
            timeframe: Timeframe to train on
        """
        logger.info("=" * 60)
        logger.info("STEP 3: DEEP TRAINING")
        logger.info("=" * 60)
        
        # Prepare data with optimized features
        optimized_data = {}
        
        for coin, timeframes in data_dict.items():
            if timeframe not in timeframes:
                continue
            
            config_key = f"{coin}_{timeframe}"
            if config_key not in feature_configs:
                logger.warning(f"No feature config for {coin} {timeframe}. Skipping.")
                continue
            
            df = timeframes[timeframe].copy()
            feature_config = feature_configs[config_key]
            
            # Generate all features
            df_with_features = self.feature_generator.generate_candidate_features(df)
            
            # Select only optimized features
            selected_features = feature_config['selected_features']
            base_cols = ['open', 'high', 'low', 'close']
            if 'volume' in df_with_features.columns:
                base_cols.append('volume')
            
            # Filter to selected features
            available_features = [f for f in selected_features if f in df_with_features.columns]
            feature_cols = base_cols + available_features
            
            optimized_df = df_with_features[feature_cols].copy()
            optimized_data[coin] = optimized_df
            
            logger.info(f"Prepared {coin}: {len(available_features)}/{len(selected_features)} features available")
        
        if not optimized_data:
            logger.error("No optimized data prepared. Cannot train.")
            return
        
        logger.info(f"\nTraining on {len(optimized_data)} coins with optimized features...")
        
        # Train TFT
        logger.info("Training TFT model...")
        self.trainer.tft_model = TFTModel()
        self.trainer.pretrain_tft(optimized_data, epochs=30, batch_size=64)
        
        # Create environment
        logger.info("Creating trading environment...")
        env = TradingEnv(data=optimized_data)
        
        # Train PPO (5M steps for high-frequency training)
        logger.info("Training PPO agent (5M steps for high-frequency)...")
        self.trainer.ppo_agent = PPOTradingAgent(
            observation_space=env.observation_space,
            action_space=env.action_space
        )
        self.trainer.train_ppo(env, total_timesteps=5000000)  # 5M steps
        
        logger.info("✓ Deep training completed")
    
    async def run_full_pipeline(
        self,
        days: int = 90,
        n_trials: int = 100,
        coins: List[str] = None,
        timeframe: str = SCALP_TIMEFRAME
    ):
        """Run the complete pipeline."""
        logger.info("=" * 60)
        logger.info("GOD_LEVEL_TRADER_FINAL - AUTO-FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Step 1: Download Data
        data_dict = await self.step1_download_data(days=days, coins=coins, timeframe=timeframe)
        
        # Step 2: Feature Race
        feature_configs = self.step2_feature_race(data_dict, n_trials_per_coin=n_trials)
        
        # Step 3: Deep Training
        self.step3_deep_training(data_dict, feature_configs, timeframe=timeframe)
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 60)
        logger.info("Feature configs saved to: feature_configs/")
        logger.info("Trained models saved to: models/checkpoints/")


async def run_live_trading(
    coins: List[str] = None,
    timeframe: str = SCALP_TIMEFRAME
):
    """
    Run live trading with optimized features.
    Loads feature configs and uses only those features.
    """
    logger.info("=" * 60)
    logger.info("LIVE TRADING MODE - Using Optimized Features")
    logger.info("=" * 60)
    
    if coins is None:
        coins = TOP_20_COINS
    
    # Load data
    loader = DataLoader()
    raw_data = await loader.fetch_recent(days=7, timeframe=timeframe, coins=coins)
    
    # Load feature configs and prepare optimized data
    feature_generator = FeatureGenerator()
    optimized_data = {}
    
    for coin in coins:
        config = load_feature_config(coin, timeframe)
        
        if config is None:
            logger.warning(f"No feature config for {coin}. Using all features.")
            # Fallback: use all features
            if coin in raw_data:
                optimized_data[coin] = feature_generator.generate_candidate_features(raw_data[coin])
        else:
            # Use optimized features
            if coin in raw_data:
                df = raw_data[coin].copy()
                df_with_features = feature_generator.generate_candidate_features(df)
                
                # Select only optimized features
                selected_features = config['selected_features']
                base_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df_with_features.columns:
                    base_cols.append('volume')
                
                available_features = [f for f in selected_features if f in df_with_features.columns]
                feature_cols = base_cols + available_features
                
                optimized_data[coin] = df_with_features[feature_cols].copy()
                logger.info(f"{coin}: Using {len(available_features)} optimized features")
    
    if not optimized_data:
        logger.error("No optimized data prepared. Cannot start live trading.")
        return
    
    # Start trading bot
    bot = TradingBot()
    bot.env = TradingEnv(data=optimized_data)
    bot.load_models()
    
    if bot.tft_model is None or bot.ppo_agent is None:
        logger.error("Models not loaded. Run training pipeline first.")
        return
    
    bot.live_mode()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GOD_LEVEL_TRADER_FINAL - Auto-Feature Engineering Edition"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['pipeline', 'live', 'optimize'],
        required=True,
        help='Mode: pipeline (full training), live (trading), optimize (feature race only)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Days of historical data to fetch'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of Optuna trials per coin'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default=SCALP_TIMEFRAME,
        help='Trading timeframe'
    )
    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        default=None,
        help='Specific coins to process (default: all top 20)'
    )
    
    args = parser.parse_args()
    
    pipeline = FeatureEngineeringPipeline()
    
    if args.mode == 'pipeline':
        # Run full pipeline
        asyncio.run(pipeline.run_full_pipeline(
            days=args.days,
            n_trials=args.trials,
            coins=args.coins,
            timeframe=args.timeframe
        ))
    
    elif args.mode == 'optimize':
        # Run only feature optimization
        async def optimize_only():
            data_dict = await pipeline.step1_download_data(
                days=args.days,
                coins=args.coins,
                timeframe=args.timeframe
            )
            feature_configs = pipeline.step2_feature_race(
                data_dict,
                n_trials_per_coin=args.trials
            )
            logger.info(f"Feature optimization completed. Configs saved to feature_configs/")
        
        asyncio.run(optimize_only())
    
    elif args.mode == 'live':
        # Run live trading
        asyncio.run(run_live_trading(coins=args.coins, timeframe=args.timeframe))


if __name__ == "__main__":
    main()

