"""
Two-Layer Optuna Optimizer
Layer 1: Feature Selection (which features to use)
Layer 2: Parameter Tuning (indicator parameters)
"""

import optuna
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_engine.features import FeatureGenerator
from env.trading_env import TradingEnv
from models.tft import TFTModel
from models.ppo import PPOTradingAgent
from config import INITIAL_BALANCE, SCALP_TIMEFRAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TwoLayerOptimizer:
    """
    Two-layer optimization:
    1. Feature Selection: Which features to use
    2. Parameter Tuning: Indicator parameters
    """
    
    def __init__(
        self,
        coin: str,
        timeframe: str,
        data: pd.DataFrame,
        n_trials: int = 100,
        timeout: Optional[int] = None
    ):
        """Initialize optimizer for a specific coin and timeframe."""
        self.coin = coin
        self.timeframe = timeframe
        self.data = data
        self.n_trials = n_trials
        self.timeout = timeout
        self.feature_generator = FeatureGenerator()
        
        # Generate full feature pool
        logger.info(f"Generating feature pool for {coin} {timeframe}...")
        self.feature_pool_df = self.feature_generator.generate_candidate_features(data)
        self.all_features = self.feature_generator.get_feature_list(self.feature_pool_df)
        
        logger.info(f"Generated {len(self.all_features)} candidate features")
        
        # Feature categories for organized selection
        self.feature_categories = self._categorize_features()
        
    def _categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features for organized selection."""
        categories = {
            'linreg': [f for f in self.all_features if 'linreg' in f.lower() or 'slope' in f.lower()],
            'oscillators': [f for f in self.all_features if any(x in f.lower() for x in ['rsi', 'stoch', 'cci', 'williams'])],
            'trend': [f for f in self.all_features if any(x in f.lower() for x in ['ema', 'supertrend', 'adx', 'macd'])],
            'volatility': [f for f in self.all_features if any(x in f.lower() for x in ['atr', 'bb', 'kc', 'bollinger'])],
            'volume': [f for f in self.all_features if any(x in f.lower() for x in ['obv', 'mfi', 'vwap', 'volume'])],
            'price': [f for f in self.all_features if 'price_' in f.lower()]
        }
        return categories
    
    def _suggest_feature_selection(self, trial: optuna.Trial) -> List[str]:
        """
        Layer 1: Feature Selection
        Optuna chooses which features to use.
        """
        selected_features = []
        
        # Select from each category
        for category, features in self.feature_categories.items():
            if not features:
                continue
            
            # Toggle category on/off
            use_category = trial.suggest_categorical(f'use_{category}', [True, False])
            
            if use_category and len(features) > 0:
                # Select subset of features from category
                if len(features) <= 5:
                    # Use all features in small categories
                    selected_features.extend(features)
                else:
                    # Select top N features from category
                    n_features = trial.suggest_int(f'n_{category}_features', 1, min(10, len(features)))
                    # Randomly select (in real implementation, could use importance)
                    import random
                    selected = random.sample(features, min(n_features, len(features)))
                    selected_features.extend(selected)
        
        # Ensure we have at least some features
        if len(selected_features) == 0:
            # Fallback: select some core features
            selected_features = [
                f for f in self.all_features 
                if any(x in f for x in ['RSI_14', 'linreg_50', 'ATR_14', 'VWAP'])
            ][:10]
        
        return list(set(selected_features))  # Remove duplicates
    
    def _suggest_indicator_parameters(self, trial: optuna.Trial, selected_features: List[str]) -> Dict:
        """
        Layer 2: Parameter Tuning
        Optuna selects parameter values for chosen indicators.
        """
        params = {}
        
        # RSI periods
        if any('RSI' in f for f in selected_features):
            params['rsi_period'] = trial.suggest_int('rsi_period', 5, 30)
        
        # LinReg lengths (include shorter periods for high-frequency)
        if any('linreg' in f.lower() for f in selected_features):
            params['linreg_length'] = trial.suggest_int('linreg_length', 10, 200)  # Start from 10 for frequent signals
        
        # ATR periods
        if any('ATR' in f for f in selected_features):
            params['atr_period'] = trial.suggest_int('atr_period', 7, 21)
        
        # EMA lengths
        if any('EMA' in f for f in selected_features):
            params['ema_fast'] = trial.suggest_int('ema_fast', 8, 21)
            params['ema_slow'] = trial.suggest_int('ema_slow', 21, 55)
        
        # MACD parameters
        if any('MACD' in f for f in selected_features):
            params['macd_fast'] = trial.suggest_int('macd_fast', 8, 15)
            params['macd_slow'] = trial.suggest_int('macd_slow', 21, 30)
            params['macd_signal'] = trial.suggest_int('macd_signal', 7, 12)
        
        # Bollinger Bands
        if any('BB' in f for f in selected_features):
            params['bb_period'] = trial.suggest_int('bb_period', 15, 25)
            params['bb_std'] = trial.suggest_float('bb_std', 1.5, 3.0)
        
        # SuperTrend
        if any('SuperTrend' in f for f in selected_features):
            params['supertrend_period'] = trial.suggest_int('supertrend_period', 10, 21)
            params['supertrend_mult'] = trial.suggest_float('supertrend_mult', 2.0, 4.0)
        
        # ADX
        if any('ADX' in f for f in selected_features):
            params['adx_period'] = trial.suggest_int('adx_period', 10, 25)
        
        # Volume indicators
        if any('MFI' in f for f in selected_features):
            params['mfi_period'] = trial.suggest_int('mfi_period', 10, 25)
        
        return params
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino Ratio (downside deviation only).
        More appropriate for trading than Sharpe (doesn't penalize upside volatility).
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        
        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            # No downside, return high score
            return 10.0 if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)  # Annualized
        return sortino
    
    def _steps_per_month(self, timeframe: str) -> int:
        """Calculate approximate steps per month based on timeframe."""
        timeframe_map = {
            '1m': 30 * 24 * 30,  # 30 days * 24 hours * 30 candles/hour
            '5m': 30 * 24 * 12,
            '15m': 30 * 24 * 4,
            '1h': 30 * 24,
            '4h': 30 * 6,
            '1d': 30
        }
        return timeframe_map.get(timeframe.lower(), 30 * 24 * 4)  # Default to 15m
    
    def _backtest_feature_config(
        self,
        selected_features: List[str],
        indicator_params: Dict,
        steps: int = 500,
        trial: Optional[optuna.Trial] = None
    ) -> Dict[str, float]:
        """
        Backtest a feature configuration.
        Returns performance metrics including Sortino Ratio and Total Trades.
        
        Args:
            trial: Optuna trial for pruning
        """
        try:
            # Select features from pool
            base_cols = ['open', 'high', 'low', 'close']
            if 'volume' in self.feature_pool_df.columns:
                base_cols.append('volume')
            
            # Filter to selected features
            available_features = [f for f in selected_features if f in self.feature_pool_df.columns]
            
            if len(available_features) == 0:
                if trial:
                    trial.report(-999.0, step=0)
                    raise optuna.TrialPruned()
                return {
                    'sortino_ratio': -999.0,
                    'total_return': -1.0,
                    'max_drawdown': 1.0,
                    'num_trades': 0
                }
            
            # Create feature DataFrame
            feature_cols = base_cols + available_features
            feature_df = self.feature_pool_df[feature_cols].copy()
            
            # Prepare data dict (required format)
            data_dict = {self.coin: feature_df}
            
            # Create environment
            env = TradingEnv(data=data_dict, initial_balance=INITIAL_BALANCE)
            
            # Run backtest
            obs, info = env.reset()
            portfolio_values = [info['portfolio_value']]
            returns = []
            
            # Calculate steps per month for pruning
            steps_per_month = self._steps_per_month(self.timeframe)
            min_trades_per_month = 15
            
            # Simple random policy for feature evaluation
            # (Full TFT+PPO training would be too slow for 100 trials)
            for step in range(steps):
                # Random action (simplified for speed)
                action = np.random.uniform(-0.3, 0.3, size=(1,))
                confidence = 0.6  # Mock confidence
                
                # Get ATR (try different periods)
                atr = 0.0
                for atr_period in [14, 21, 7]:
                    atr_col = f'ATR_{atr_period}'
                    if atr_col in feature_df.columns:
                        atr = feature_df[atr_col].iloc[min(step, len(feature_df)-1)]
                        break
                
                obs, reward, terminated, truncated, info = env.step(
                    action,
                    tft_confidence_15m=confidence,
                    tft_confidence_1h=confidence,
                    tft_confidence_4h=confidence,
                    atr=atr
                )
                
                portfolio_values.append(info['portfolio_value'])
                if len(portfolio_values) > 1:
                    ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    returns.append(ret)
                
                # Early pruning: Check if we're on track for minimum trades
                # Check every 50 steps (more frequent for faster pruning)
                if trial and step > 0 and step % 50 == 0:
                    months_simulated = step / steps_per_month if steps_per_month > 0 else 1
                    trades_per_month = info.get('total_trades', 0) / months_simulated if months_simulated > 0 else 0
                    
                    # Report intermediate score for pruning
                    current_score = 0.0  # Will be calculated at end
                    trial.report(current_score, step=step)
                    
                    # Prune if clearly not meeting minimum trades requirement
                    if months_simulated >= 0.5 and trades_per_month < min_trades_per_month * 0.5:
                        # After half a month, if we're at less than half the required rate, prune
                        logger.debug(f"Pruning trial at step {step}: Only {trades_per_month:.1f} trades/month (need {min_trades_per_month})")
                        raise optuna.TrialPruned()
                
                if terminated or truncated:
                    break
            
            # Calculate final metrics
            returns_array = np.array(returns) if returns else np.array([0.0])
            portfolio_array = np.array(portfolio_values)
            
            total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0] if len(portfolio_array) > 1 else 0.0
            
            # Sortino Ratio (downside deviation only)
            sortino = self._calculate_sortino_ratio(returns_array)
            
            # Sharpe ratio (for reference)
            if len(returns_array) > 0 and np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            # Max drawdown
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - peak) / peak
            max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
            
            total_trades = info.get('total_trades', 0)
            
            # Final check: Minimum trade constraint
            months_simulated = steps / steps_per_month if steps_per_month > 0 else 1
            trades_per_month = total_trades / months_simulated if months_simulated > 0 else 0
            
            if trades_per_month < min_trades_per_month:
                # Too passive, return very bad score
                logger.debug(f"Trial failed minimum trade constraint: {trades_per_month:.1f} trades/month")
                if trial:
                    trial.report(-999.0, step=steps)
                    raise optuna.TrialPruned()
                return {
                    'sortino_ratio': -999.0,
                    'total_return': -1.0,
                    'max_drawdown': 1.0,
                    'num_trades': total_trades,
                    'trades_per_month': trades_per_month
                }
            
            return {
                'sortino_ratio': sortino,
                'sharpe_ratio': sharpe,  # Keep for reference
                'total_return': total_return,
                'max_drawdown': max_dd,
                'num_trades': total_trades,
                'trades_per_month': trades_per_month
            }
            
        except optuna.TrialPruned:
            raise  # Re-raise pruning
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            if trial:
                trial.report(-999.0, step=0)
                raise optuna.TrialPruned()
            return {
                'sortino_ratio': -999.0,
                'total_return': -1.0,
                'max_drawdown': 1.0,
                'num_trades': 0
            }
    
    def optimize(self) -> Tuple[optuna.Study, Dict]:
        """
        Run two-layer optimization.
        
        Returns:
            (study, best_config) where best_config contains selected features and parameters
        """
        logger.info(f"Starting two-layer optimization for {self.coin} {self.timeframe}...")
        
        def objective(trial: optuna.Trial):
            # Layer 1: Feature Selection
            selected_features = self._suggest_feature_selection(trial)
            
            # Layer 2: Parameter Tuning
            indicator_params = self._suggest_indicator_parameters(trial, selected_features)
            
            # Backtest configuration (with trial for pruning)
            metrics = self._backtest_feature_config(
                selected_features=selected_features,
                indicator_params=indicator_params,
                steps=500,  # Reduced for speed during optimization
                trial=trial
            )
            
            # Store trial info
            trial.set_user_attr('selected_features', selected_features)
            trial.set_user_attr('indicator_params', indicator_params)
            trial.set_user_attr('num_features', len(selected_features))
            trial.set_user_attr('total_return', metrics['total_return'])
            trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
            trial.set_user_attr('num_trades', metrics['num_trades'])
            trial.set_user_attr('trades_per_month', metrics.get('trades_per_month', 0))
            trial.set_user_attr('sortino_ratio', metrics['sortino_ratio'])
            trial.set_user_attr('sharpe_ratio', metrics.get('sharpe_ratio', 0.0))
            
            # ====================================================================
            # AGGRESSIVE OBJECTIVE: Sortino_Ratio * log1p(Total_Trades)
            # ====================================================================
            # This forces high-frequency trading:
            # - Low trades (5) → log1p(5) ≈ 1.79 → heavily penalized
            # - High trades (100) → log1p(100) ≈ 4.61 → boosted
            # - Optuna will find indicators that generate frequent signals
            # ====================================================================
            total_trades = metrics['num_trades']
            sortino = metrics['sortino_ratio']
            
            # Ensure positive Sortino (negative means losing strategy)
            if sortino < 0:
                sortino = 0.0
            
            # Calculate aggressive score
            score = sortino * np.log1p(total_trades)
            
            logger.debug(
                f"Trial {trial.number}: Sortino={sortino:.2f}, "
                f"Trades={total_trades}, Score={score:.2f}"
            )
            
            return score
        
        # Create study with MedianPruner for early stopping
        # This will prune trials that are clearly underperforming
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,  # Don't prune first 5 trials
            n_warmup_steps=10,   # Wait 10 steps before pruning
            interval_steps=1     # Check every step
        )
        
        study = optuna.create_study(
            direction='maximize',
            study_name=f'feature_optimization_{self.coin}_{self.timeframe}',
            pruner=pruner
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Extract best configuration
        best_trial = study.best_trial
        best_config = {
            'coin': self.coin,
            'timeframe': self.timeframe,
            'selected_features': best_trial.user_attrs['selected_features'],
            'indicator_params': best_trial.user_attrs['indicator_params'],
            'num_features': best_trial.user_attrs['num_features'],
            'performance': {
                'aggressive_score': study.best_value,  # Sortino * log1p(Trades)
                'sortino_ratio': best_trial.user_attrs.get('sortino_ratio', 0.0),
                'sharpe_ratio': best_trial.user_attrs.get('sharpe_ratio', 0.0),
                'total_return': best_trial.user_attrs['total_return'],
                'max_drawdown': best_trial.user_attrs['max_drawdown'],
                'num_trades': best_trial.user_attrs.get('num_trades', 0),
                'trades_per_month': best_trial.user_attrs.get('trades_per_month', 0)
            }
        }
        
        logger.info("=" * 60)
        logger.info(f"OPTIMIZATION COMPLETED for {self.coin} {self.timeframe}")
        logger.info("=" * 60)
        logger.info(f"Best Aggressive Score: {study.best_value:.4f} (Sortino * log1p(Trades))")
        logger.info(f"Sortino Ratio: {best_config['performance']['sortino_ratio']:.4f}")
        logger.info(f"Total Trades: {best_config['performance']['num_trades']}")
        logger.info(f"Trades/Month: {best_config['performance']['trades_per_month']:.1f}")
        logger.info(f"Selected Features: {len(best_config['selected_features'])}")
        logger.info(f"Top 10 Features: {best_config['selected_features'][:10]}")
        logger.info("=" * 60)
        
        return study, best_config
    
    def save_config(self, config: Dict, output_dir: str = "feature_configs"):
        """Save feature configuration to JSON file."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filename = f"feature_config_{self.coin.replace('/', '_')}_{self.timeframe}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Feature config saved to {filepath}")
        return filepath


def load_feature_config(coin: str, timeframe: str, config_dir: str = "feature_configs") -> Optional[Dict]:
    """Load feature configuration from JSON file."""
    config_path = Path(config_dir) / f"feature_config_{coin.replace('/', '_')}_{timeframe}.json"
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Loaded feature config from {config_path}")
    return config


if __name__ == "__main__":
    # Test optimizer
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 101,
        'low': np.random.randn(1000).cumsum() + 99,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    optimizer = TwoLayerOptimizer(
        coin='SOL/USDT',
        timeframe='15m',
        data=sample_data,
        n_trials=10  # Small number for testing
    )
    
    study, config = optimizer.optimize()
    optimizer.save_config(config)

