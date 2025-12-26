"""
Model Validator
Compares model performance and determines if hot-swap is needed.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.tft import TFTModel
from models.ppo import PPOTradingAgent
from env.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Validator:
    """Validates and compares model performance."""
    
    def __init__(
        self,
        old_tft_model: Optional[TFTModel] = None,
        new_tft_model: Optional[TFTModel] = None,
        old_ppo_agent: Optional[PPOTradingAgent] = None,
        new_ppo_agent: Optional[PPOTradingAgent] = None
    ):
        """Initialize validator."""
        self.old_tft_model = old_tft_model
        self.new_tft_model = new_tft_model
        self.old_ppo_agent = old_ppo_agent
        self.new_ppo_agent = new_ppo_agent
        
        logger.info("Validator initialized")
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
        
        return sharpe
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        return abs(max_drawdown)
    
    def backtest_model(
        self,
        env: TradingEnv,
        tft_model: Optional[TFTModel],
        ppo_agent: Optional[PPOTradingAgent],
        steps: int = 1000
    ) -> Dict[str, float]:
        """
        Backtest a model combination on the environment.
        
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Backtesting model for {steps} steps...")
        
        obs, info = env.reset()
        portfolio_values = [info['portfolio_value']]
        returns = []
        
        lstm_states = None
        
        for step in range(steps):
            # Get TFT prediction
            if tft_model and ppo_agent:
                current_coin = env.current_coin
                coin_data = env.data.get(current_coin)
                
                if coin_data is not None and len(coin_data) > 0:
                    try:
                        predictions, confidence = tft_model.predict(coin_data, current_coin)
                        atr = coin_data['ATR'].iloc[-1] if 'ATR' in coin_data.columns else 0.0
                    except Exception as e:
                        logger.warning(f"TFT prediction failed: {e}")
                        confidence = 0.5
                        atr = 0.0
                else:
                    confidence = 0.5
                    atr = 0.0
                
                # Get PPO action
                action, lstm_states = ppo_agent.predict(obs, lstm_states=lstm_states)
            else:
                # Random action for baseline
                action = np.random.uniform(-0.5, 0.5, size=(1,))
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
        returns_array = np.array(returns)
        portfolio_array = np.array(portfolio_values)
        
        total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0]
        sharpe = self.calculate_sharpe_ratio(returns_array)
        max_dd = self.calculate_max_drawdown(portfolio_array)
        num_trades = info.get('total_trades', 0)
        
        metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'num_trades': num_trades,
            'final_portfolio_value': portfolio_array[-1]
        }
        
        logger.info(f"Backtest completed: Return={total_return:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")
        
        return metrics
    
    def compare_models(
        self,
        env: TradingEnv,
        steps: int = 1000
    ) -> Optional[Dict]:
        """
        Compare old and new models.
        
        Returns:
            Dictionary with comparison results and swap recommendation
        """
        logger.info("Comparing old and new models...")
        
        if self.new_tft_model is None or self.new_ppo_agent is None:
            logger.warning("New models not available for comparison.")
            return None
        
        # Backtest new model
        new_metrics = self.backtest_model(env, self.new_tft_model, self.new_ppo_agent, steps)
        
        # Backtest old model if available
        if self.old_tft_model is not None and self.old_ppo_agent is not None:
            old_metrics = self.backtest_model(env, self.old_tft_model, self.old_ppo_agent, steps)
        else:
            logger.warning("Old models not available. Using new model metrics only.")
            old_metrics = None
        
        # Compare
        if old_metrics is not None:
            new_sharpe = new_metrics['sharpe_ratio']
            old_sharpe = old_metrics['sharpe_ratio']
            
            should_swap = new_sharpe > old_sharpe
            
            comparison = {
                'old_metrics': old_metrics,
                'new_metrics': new_metrics,
                'old_sharpe': old_sharpe,
                'new_sharpe': new_sharpe,
                'should_swap': should_swap,
                'improvement': new_sharpe - old_sharpe
            }
            
            logger.info(f"Comparison: Old Sharpe={old_sharpe:.2f}, New Sharpe={new_sharpe:.2f}")
            logger.info(f"Recommendation: {'SWAP' if should_swap else 'KEEP OLD'}")
        else:
            comparison = {
                'new_metrics': new_metrics,
                'should_swap': True,  # If no old model, use new one
                'improvement': 0.0
            }
            logger.info("No old model to compare. Using new model.")
        
        return comparison


if __name__ == "__main__":
    # Test validator
    validator = Validator()
    logger.info("Validator created successfully")

