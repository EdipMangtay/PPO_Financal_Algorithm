"""
Optuna Hyperparameter Tuner for Dynamic Trailing Stop Parameters
Optimizes ATR period and trailing multiplier to find the "sweet spot".
"""

import optuna
import numpy as np
from typing import Dict, Optional
import logging
import asyncio
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    INITIAL_BALANCE,
    TOP_20_COINS,
    SCALP_TIMEFRAME
)
from data.loader import DataLoader
from env.trading_env import TradingEnv
from models.tft import TFTModel
from models.ppo import PPOTradingAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExitParameterTuner:
    """
    Optimizes exit parameters (ATR period and trailing multiplier)
    to find the sweet spot between allowing volatility and exiting on trend breaks.
    """
    
    def __init__(
        self,
        data: Optional[Dict] = None,
        tft_model: Optional[TFTModel] = None,
        ppo_agent: Optional[PPOTradingAgent] = None,
        n_trials: int = 50,
        timeout: Optional[int] = None
    ):
        """Initialize tuner."""
        self.data = data
        self.tft_model = tft_model
        self.ppo_agent = ppo_agent
        self.n_trials = n_trials
        self.timeout = timeout
        
        logger.info("Exit Parameter Tuner initialized")
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
        return sharpe
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        return abs(max_drawdown)
    
    def _backtest_exit_params(
        self,
        atr_period: int,
        trailing_mult: float,
        steps: int = 1000
    ) -> Dict[str, float]:
        """
        Backtest exit parameters on the environment.
        
        Args:
            atr_period: ATR calculation period (e.g., 7, 14)
            trailing_mult: Trailing stop multiplier (e.g., 1.5 to 3.5)
            steps: Number of steps to backtest
        
        Returns:
            Dictionary with performance metrics
        """
        if self.data is None:
            raise ValueError("Data not provided. Load data first.")
        
        # Create environment with optimized parameters
        env = TradingEnv(
            data=self.data,
            initial_balance=INITIAL_BALANCE,
            atr_period=atr_period,
            trailing_stop_atr_multiplier=trailing_mult
        )
        
        # Reset environment
        obs, info = env.reset()
        portfolio_values = [info['portfolio_value']]
        returns = []
        
        lstm_states = None
        
        # Run backtest
        for step in range(steps):
            # Get TFT prediction
            if self.tft_model and self.ppo_agent:
                current_coin = env.current_coin
                coin_data = env.data.get(current_coin)
                
                if coin_data is not None and len(coin_data) > 0:
                    try:
                        predictions, confidence = self.tft_model.predict(coin_data, current_coin)
                        atr = coin_data['ATR'].iloc[-1] if 'ATR' in coin_data.columns else 0.0
                    except Exception as e:
                        logger.debug(f"TFT prediction failed: {e}")
                        confidence = 0.5
                        atr = 0.0
                else:
                    confidence = 0.5
                    atr = 0.0
                
                # Get PPO action
                action, lstm_states = self.ppo_agent.predict(obs, lstm_states=lstm_states)
            else:
                # Random action for baseline (if models not available)
                action = np.random.uniform(-0.3, 0.3, size=(1,))
                confidence = 0.5
                atr = 0.0
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(
                action,
                tft_confidence=confidence,
                atr=atr
            )
            
            portfolio_values.append(info['portfolio_value'])
            
            if len(portfolio_values) > 1:
                ret = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                returns.append(ret)
            
            if terminated or truncated:
                break
        
        # Calculate metrics
        returns_array = np.array(returns) if returns else np.array([0.0])
        portfolio_array = np.array(portfolio_values)
        
        total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0] if len(portfolio_array) > 1 else 0.0
        sharpe = self._calculate_sharpe_ratio(returns_array)
        max_dd = self._calculate_max_drawdown(portfolio_array)
        num_trades = info.get('total_trades', 0)
        win_rate = 0.0
        
        if len(env.trades) > 0:
            winning_trades = sum(1 for t in env.trades if t.pnl > 0)
            win_rate = winning_trades / len(env.trades)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'final_portfolio_value': portfolio_array[-1] if len(portfolio_array) > 0 else INITIAL_BALANCE
        }
        
        return metrics
    
    def optimize(
        self,
        steps_per_trial: int = 1000,
        objective: str = 'sharpe_ratio'
    ) -> optuna.Study:
        """
        Optimize exit parameters using Optuna.
        
        Args:
            steps_per_trial: Number of steps to backtest per trial
            objective: Objective metric ('sharpe_ratio', 'total_return', or 'composite')
        
        Returns:
            Optuna study with optimization results
        """
        logger.info(f"Starting optimization with {self.n_trials} trials...")
        
        def objective_func(trial):
            # Suggest hyperparameters
            atr_period = trial.suggest_int('atr_period', 7, 21, step=7)  # 7, 14, 21
            trailing_mult = trial.suggest_float('trailing_mult', 1.5, 3.5, step=0.25)  # 1.5 to 3.5
            
            logger.info(f"Trial {trial.number}: ATR_period={atr_period}, trailing_mult={trailing_mult:.2f}")
            
            try:
                # Backtest with these parameters
                metrics = self._backtest_exit_params(
                    atr_period=atr_period,
                    trailing_mult=trailing_mult,
                    steps=steps_per_trial
                )
                
                # Objective: Maximize Sharpe ratio (allows volatility but exits on trend breaks)
                if objective == 'sharpe_ratio':
                    score = metrics['sharpe_ratio']
                elif objective == 'total_return':
                    score = metrics['total_return']
                elif objective == 'composite':
                    # Composite: Sharpe * (1 - max_dd) * win_rate
                    score = metrics['sharpe_ratio'] * (1 - metrics['max_drawdown']) * metrics['win_rate']
                else:
                    score = metrics['sharpe_ratio']
                
                # Report metrics
                trial.set_user_attr('total_return', metrics['total_return'])
                trial.set_user_attr('max_drawdown', metrics['max_drawdown'])
                trial.set_user_attr('num_trades', metrics['num_trades'])
                trial.set_user_attr('win_rate', metrics['win_rate'])
                
                logger.info(
                    f"Trial {trial.number} results: "
                    f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                    f"Return={metrics['total_return']:.2%}, "
                    f"MaxDD={metrics['max_drawdown']:.2%}, "
                    f"Trades={metrics['num_trades']}, "
                    f"WinRate={metrics['win_rate']:.2%}"
                )
                
                return score
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                return -999.0  # Return very bad score on error
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name='exit_parameters_optimization'
        )
        
        # Run optimization
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Print best results
        logger.info("=" * 60)
        logger.info("OPTIMIZATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Best Trial: {study.best_trial.number}")
        logger.info(f"Best Parameters:")
        logger.info(f"  ATR Period: {study.best_params['atr_period']}")
        logger.info(f"  Trailing Multiplier: {study.best_params['trailing_mult']:.2f}")
        logger.info(f"Best Score (Sharpe): {study.best_value:.4f}")
        logger.info(f"Best Trial Metrics:")
        logger.info(f"  Total Return: {study.best_trial.user_attrs['total_return']:.2%}")
        logger.info(f"  Max Drawdown: {study.best_trial.user_attrs['max_drawdown']:.2%}")
        logger.info(f"  Win Rate: {study.best_trial.user_attrs['win_rate']:.2%}")
        logger.info(f"  Number of Trades: {study.best_trial.user_attrs['num_trades']}")
        logger.info("=" * 60)
        
        return study
    
    def save_results(self, study: optuna.Study, path: str = "tuning_results.txt"):
        """Save optimization results to file."""
        with open(path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EXIT PARAMETER OPTIMIZATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best Parameters:\n")
            f.write(f"  ATR Period: {study.best_params['atr_period']}\n")
            f.write(f"  Trailing Multiplier: {study.best_params['trailing_mult']:.2f}\n")
            f.write(f"Best Score (Sharpe): {study.best_value:.4f}\n\n")
            f.write("Best Trial Metrics:\n")
            f.write(f"  Total Return: {study.best_trial.user_attrs['total_return']:.2%}\n")
            f.write(f"  Max Drawdown: {study.best_trial.user_attrs['max_drawdown']:.2%}\n")
            f.write(f"  Win Rate: {study.best_trial.user_attrs['win_rate']:.2%}\n")
            f.write(f"  Number of Trades: {study.best_trial.user_attrs['num_trades']}\n")
        
        logger.info(f"Results saved to {path}")


def main():
    """Main entry point for tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize exit parameters for dynamic trailing stops")
    parser.add_argument('--trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--steps', type=int, default=1000, help='Steps per trial')
    parser.add_argument('--objective', type=str, default='sharpe_ratio', 
                       choices=['sharpe_ratio', 'total_return', 'composite'],
                       help='Objective metric to optimize')
    parser.add_argument('--data-days', type=int, default=7, help='Days of data to fetch')
    
    args = parser.parse_args()
    
    logger.info("Loading data...")
    loader = DataLoader()
    data = asyncio.run(loader.fetch_recent(days=args.data_days, timeframe=SCALP_TIMEFRAME))
    
    if not data:
        logger.error("Failed to load data. Exiting.")
        return
    
    logger.info(f"Loaded data for {len(data)} coins")
    
    # Try to load models (optional)
    tft_model = None
    ppo_agent = None
    
    tft_path = Path("models/checkpoints/tft_pretrained.pt")
    if tft_path.exists():
        logger.info("Loading TFT model...")
        # Note: Would need training_data to fully load, simplified for now
        tft_model = TFTModel()
    
    # Create tuner
    tuner = ExitParameterTuner(
        data=data,
        tft_model=tft_model,
        ppo_agent=ppo_agent,
        n_trials=args.trials,
        timeout=args.timeout
    )
    
    # Run optimization
    study = tuner.optimize(steps_per_trial=args.steps, objective=args.objective)
    
    # Save results
    tuner.save_results(study)


if __name__ == "__main__":
    main()

