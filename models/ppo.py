"""
Recurrent PPO Commander Agent
Uses sb3-contrib for LSTM-based PPO with continuous action space.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
import gymnasium as gym
from gymnasium import spaces
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PPO_HIDDEN_SIZE,
    PPO_LEARNING_RATE,
    PPO_N_STEPS,
    PPO_BATCH_SIZE,
    PPO_N_EPOCHS,
    PPO_GAMMA,
    PPO_GAE_LAMBDA,
    PPO_CLIP_RANGE,
    PPO_ENT_COEF,
    PPO_VF_COEF
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force float32 matmul precision for RTX 5070
torch.set_float32_matmul_precision('medium')


class TradingFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for trading observations."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        hidden_size: int = PPO_HIDDEN_SIZE
    ):
        """Initialize feature extractor."""
        super().__init__(observation_space, features_dim)
        
        # Input: TFT Confidence + Volatility (ATR) + PnL State + Position Info
        n_input = observation_space.shape[0]
        
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations."""
        return self.net(observations)


class PPOTradingAgent:
    """Recurrent PPO agent for trading decisions."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = PPO_HIDDEN_SIZE,
        learning_rate: float = PPO_LEARNING_RATE,
        n_steps: int = PPO_N_STEPS,
        batch_size: int = PPO_BATCH_SIZE,
        n_epochs: int = PPO_N_EPOCHS,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_range: float = PPO_CLIP_RANGE,
        ent_coef: float = PPO_ENT_COEF,
        vf_coef: float = PPO_VF_COEF,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs: Optional[Dict] = None
    ):
        """Initialize PPO agent."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        
        # Default policy kwargs
        if policy_kwargs is None:
            policy_kwargs = {
                'features_extractor_class': TradingFeatureExtractor,
                'features_extractor_kwargs': {
                    'features_dim': 128,
                    'hidden_size': hidden_size
                },
                'net_arch': [dict(pi=[hidden_size, hidden_size], vf=[hidden_size, hidden_size])],
                'activation_fn': nn.ReLU,
            }
        
        # Create environment wrapper for training
        self.env = None
        self.model: Optional[RecurrentPPO] = None
        
        # Store hyperparameters
        self.hyperparams = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'vf_coef': vf_coef,
            'policy_kwargs': policy_kwargs,
            'device': device,
            'verbose': 1
        }
        
        logger.info(f"PPO Agent initialized on {device}")
    
    def build_model(self, env: gym.Env):
        """Build the PPO model with environment."""
        self.env = DummyVecEnv([lambda: env])
        
        self.model = RecurrentPPO(
            "MlpLstmPolicy",
            self.env,
            lstm_hidden_size=PPO_HIDDEN_SIZE,
            **self.hyperparams
        )
        
        logger.info("PPO model built successfully")
        return self.model
    
    def train(
        self,
        total_timesteps: int = 100000,
        callback=None,
        log_interval: int = 10
    ) -> "PPOTradingAgent":
        """Train the PPO agent."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        logger.info(f"Training PPO agent for {total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval
        )
        
        logger.info("Training completed")
        return self
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        lstm_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[np.ndarray, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Predict action from observation.
        
        Args:
            observation: Current observation (TFT Confidence + ATR + PnL + Position)
            deterministic: Whether to use deterministic policy
            lstm_states: LSTM hidden states for recurrent policy
        
        Returns:
            action: Continuous action in [-1, 1] (Long/Short/Neutral)
            lstm_states: Updated LSTM states
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Reshape for single observation
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        action, lstm_states = self.model.predict(
            observation,
            deterministic=deterministic,
            lstm_states=lstm_states
        )
        
        return action[0], lstm_states
    
    def get_action_probability(
        self,
        observation: np.ndarray,
        action: np.ndarray
    ) -> float:
        """Get probability of taking an action (for analysis)."""
        if self.model is None:
            raise ValueError("Model not built.")
        
        # This requires accessing the policy directly
        # For now, return a placeholder
        return 0.5
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        logger.info(f"PPO model saved to {path}")
    
    def load(self, path: str, env: Optional[gym.Env] = None):
        """Load model from disk."""
        if env is not None:
            self.build_model(env)
        
        if self.model is None:
            raise ValueError("Cannot load model without environment. Provide env parameter.")
        
        self.model = RecurrentPPO.load(path, env=self.env)
        logger.info(f"PPO model loaded from {path}")
    
    def fine_tune(
        self,
        env: gym.Env,
        epochs: int = 5,
        learning_rate: float = 1e-5,
        timesteps_per_epoch: int = 1000
    ):
        """Fine-tune the model with lower learning rate (prevent catastrophic forgetting)."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Update learning rate
        self.model.learning_rate = learning_rate
        
        logger.info(f"Fine-tuning PPO agent for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.model.learn(
                total_timesteps=timesteps_per_epoch,
                reset_num_timesteps=False,
                log_interval=5
            )
            logger.info(f"Fine-tuning epoch {epoch+1}/{epochs} completed")
        
        logger.info("Fine-tuning completed")


def create_ppo_agent(
    observation_space: gym.Space,
    action_space: gym.Space,
    **kwargs
) -> PPOTradingAgent:
    """Factory function to create PPO agent."""
    return PPOTradingAgent(
        observation_space=observation_space,
        action_space=action_space,
        **kwargs
    )


if __name__ == "__main__":
    # Test PPO agent
    logger.info("Testing PPO agent...")
    
    # Create dummy spaces
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    agent = PPOTradingAgent(obs_space, action_space)
    logger.info("PPO agent created successfully")

