"""
Model Trainer
Handles training, fine-tuning, and optimization of TFT and PPO models.
"""

import optuna
import torch
import numpy as np
from typing import Dict, Optional, List
import pandas as pd
import logging
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT_SECONDS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    PPO_TOTAL_TIMESTEPS
)
from models.tft import TFTModel
from models.ppo import PPOTradingAgent
from env.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force float32 matmul precision
torch.set_float32_matmul_precision('medium')


class Trainer:
    """Handles model training and fine-tuning."""
    
    def __init__(
        self,
        tft_model: Optional[TFTModel] = None,
        ppo_agent: Optional[PPOTradingAgent] = None,
        models_dir: str = "models/checkpoints"
    ):
        """Initialize trainer."""
        self.tft_model = tft_model
        self.ppo_agent = ppo_agent
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Trainer initialized")
    
    def optimize_tft_hyperparameters(
        self,
        data: Dict[str, pd.DataFrame],
        n_trials: int = OPTUNA_N_TRIALS,
        timeout: int = OPTUNA_TIMEOUT_SECONDS
    ) -> Dict:
        """
        Optimize TFT hyperparameters using Optuna.
        
        Returns:
            Best hyperparameters dictionary
        """
        logger.info(f"Optimizing TFT hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            hidden_size = trial.suggest_int('hidden_size', 32, 128, step=16)
            dropout = trial.suggest_float('dropout', 0.0, 0.3)
            attention_head_size = trial.suggest_int('attention_head_size', 2, 8)
            
            # Create model with suggested hyperparameters
            model = TFTModel(
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                dropout=dropout,
                attention_head_size=attention_head_size
            )
            
            # Create dataset
            training_data = model.create_dataset(data)
            
            # Note: TimeSeriesDataSet doesn't support iloc splitting
            # Use all data for training, validation will be handled internally
            train_data = training_data
            val_data = None
            
            # Build and train model
            model.build_model(training_data)
            
            # Train for a few epochs
            history = model.train(
                training_data=training_data,
                validation_data=val_data,
                epochs=5,
                batch_size=64,
                verbose=False
            )
            
            # Return validation loss
            if history.get('val_loss'):
                return min(history['val_loss'])
            else:
                return min(history['train_loss'])
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        logger.info(f"Best TFT hyperparameters: {best_params}")
        
        return best_params
    
    def pretrain_tft(
        self,
        data: Dict[str, pd.DataFrame],
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: Optional[float] = None
    ) -> TFTModel:
        """Pretrain TFT model on historical data."""
        logger.info("Pretraining TFT model...")
        
        if self.tft_model is None:
            self.tft_model = TFTModel(learning_rate=learning_rate or 1e-3)
        
        # Create dataset
        training_data = self.tft_model.create_dataset(data)
        
        # Split data
        # Note: TimeSeriesDataSet doesn't support iloc, so we'll use all data
        # In production, use proper time-based splitting
        
        # Build model
        self.tft_model.build_model(training_data)
        
        # Train
        history = self.tft_model.train(
            training_data=training_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        # Save model
        save_path = self.models_dir / "tft_pretrained.pt"
        self.tft_model.save(str(save_path))
        logger.info(f"TFT model pretrained and saved to {save_path}")
        
        return self.tft_model
    
    def fine_tune_tft(
        self,
        data: Dict[str, pd.DataFrame],
        epochs: int = FINE_TUNE_EPOCHS,
        learning_rate: float = FINE_TUNE_LEARNING_RATE
    ) -> TFTModel:
        """Fine-tune TFT model with gentle learning rate (prevent catastrophic forgetting)."""
        if self.tft_model is None:
            logger.warning("No TFT model to fine-tune. Creating new model.")
            self.tft_model = TFTModel(learning_rate=learning_rate)
        
        logger.info(f"Fine-tuning TFT model with lr={learning_rate} for {epochs} epochs...")
        
        # Update learning rate
        self.tft_model.learning_rate = learning_rate
        
        # Create dataset with new data
        training_data = self.tft_model.create_dataset(data)
        
        # Fine-tune (continue training)
        history = self.tft_model.train(
            training_data=training_data,
            epochs=epochs,
            batch_size=64,
            verbose=True
        )
        
        # Save fine-tuned model
        save_path = self.models_dir / "tft_finetuned.pt"
        self.tft_model.save(str(save_path))
        logger.info(f"TFT model fine-tuned and saved to {save_path}")
        
        return self.tft_model
    
    def train_ppo(
        self,
        env: TradingEnv,
        total_timesteps: int = PPO_TOTAL_TIMESTEPS,
        learning_rate: Optional[float] = None
    ) -> PPOTradingAgent:
        """Train PPO agent."""
        logger.info(f"Training PPO agent for {total_timesteps} timesteps...")
        
        if self.ppo_agent is None:
            self.ppo_agent = PPOTradingAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                learning_rate=learning_rate
            )
        
        # Build model
        self.ppo_agent.build_model(env)
        
        # Train
        self.ppo_agent.train(total_timesteps=total_timesteps)
        
        # Save model
        save_path = self.models_dir / "ppo_trained"
        self.ppo_agent.save(str(save_path))
        logger.info(f"PPO agent trained and saved to {save_path}")
        
        return self.ppo_agent
    
    def fine_tune_ppo(
        self,
        env: TradingEnv,
        epochs: int = FINE_TUNE_EPOCHS,
        learning_rate: float = FINE_TUNE_LEARNING_RATE,
        timesteps_per_epoch: int = 1000
    ) -> PPOTradingAgent:
        """Fine-tune PPO agent with gentle learning rate."""
        if self.ppo_agent is None:
            logger.warning("No PPO agent to fine-tune.")
            return None
        
        logger.info(f"Fine-tuning PPO agent with lr={learning_rate} for {epochs} epochs...")
        
        self.ppo_agent.fine_tune(
            env=env,
            epochs=epochs,
            learning_rate=learning_rate,
            timesteps_per_epoch=timesteps_per_epoch
        )
        
        # Save fine-tuned model
        save_path = self.models_dir / "ppo_finetuned"
        self.ppo_agent.save(str(save_path))
        logger.info(f"PPO agent fine-tuned and saved to {save_path}")
        
        return self.ppo_agent


if __name__ == "__main__":
    # Test trainer
    trainer = Trainer()
    logger.info("Trainer created successfully")

