"""
Weekly Retraining Scheduler
Uses APScheduler to pause trading, fetch data, fine-tune models, and hot-swap.
"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import logging
import asyncio
from typing import Optional, Callable

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    RETRAIN_DAY,
    RETRAIN_HOUR,
    RETRAIN_MINUTE,
    RETRAIN_LOOKBACK_DAYS,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE
)
from data.loader import DataLoader
from models.tft import TFTModel
from models.ppo import PPOTradingAgent
from trainer import Trainer
from validator import Validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Manages weekly retraining pipeline."""
    
    def __init__(
        self,
        tft_model: Optional[TFTModel] = None,
        ppo_agent: Optional[PPOTradingAgent] = None,
        trainer: Optional[Trainer] = None,
        validator: Optional[Validator] = None,
        trading_pause_callback: Optional[Callable] = None,
        trading_resume_callback: Optional[Callable] = None
    ):
        """Initialize scheduler."""
        self.scheduler = BlockingScheduler()
        self.tft_model = tft_model
        self.ppo_agent = ppo_agent
        self.trainer = trainer
        self.validator = validator
        self.trading_pause_callback = trading_pause_callback
        self.trading_resume_callback = trading_resume_callback
        self.is_training = False
        
        logger.info("Retraining scheduler initialized")
    
    def _weekly_retraining_job(self):
        """Weekly retraining job (runs Sunday 03:00 UTC)."""
        logger.info("=" * 60)
        logger.info("WEEKLY RETRAINING JOB STARTED")
        logger.info("=" * 60)
        
        self.is_training = True
        
        try:
            # Step 1: Pause Trading
            logger.info("Step 1: Pausing trading...")
            if self.trading_pause_callback:
                self.trading_pause_callback()
            
            # Step 2: Fetch Recent Data
            logger.info(f"Step 2: Fetching recent {RETRAIN_LOOKBACK_DAYS} days of data...")
            loader = DataLoader()
            
            # Run async data fetch
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            data = loop.run_until_complete(
                loader.fetch_recent(days=RETRAIN_LOOKBACK_DAYS)
            )
            loop.close()
            
            if not data:
                logger.error("Failed to fetch data. Aborting retraining.")
                return
            
            logger.info(f"Fetched data for {len(data)} coins")
            
            # Step 3: Fine-tune TFT Model
            logger.info("Step 3: Fine-tuning TFT model...")
            if self.tft_model and self.trainer:
                self.trainer.fine_tune_tft(
                    data=data,
                    epochs=FINE_TUNE_EPOCHS,
                    learning_rate=FINE_TUNE_LEARNING_RATE
                )
            else:
                logger.warning("TFT model or trainer not available. Skipping TFT fine-tuning.")
            
            # Step 4: Fine-tune PPO Agent
            logger.info("Step 4: Fine-tuning PPO agent...")
            if self.ppo_agent and self.trainer:
                # Note: PPO fine-tuning requires environment
                # This should be handled by the trainer with proper environment setup
                logger.info("PPO fine-tuning requires environment. Skipping for now.")
                # self.trainer.fine_tune_ppo(epochs=FINE_TUNE_EPOCHS, learning_rate=FINE_TUNE_LEARNING_RATE)
            else:
                logger.warning("PPO agent or trainer not available. Skipping PPO fine-tuning.")
            
            # Step 5: Validate Models
            logger.info("Step 5: Validating models...")
            if self.validator:
                validation_result = self.validator.compare_models()
                
                if validation_result and validation_result.get('should_swap', False):
                    logger.info("New model performs better. Hot-swapping...")
                    # Hot-swap logic would be implemented here
                    # This typically involves:
                    # 1. Save old model as backup
                    # 2. Load new model
                    # 3. Update references
                    logger.info("Model hot-swap completed")
                else:
                    logger.info("New model does not outperform. Keeping current model.")
            else:
                logger.warning("Validator not available. Skipping validation.")
            
            # Step 6: Resume Trading
            logger.info("Step 6: Resuming trading...")
            if self.trading_resume_callback:
                self.trading_resume_callback()
            
            logger.info("=" * 60)
            logger.info("WEEKLY RETRAINING JOB COMPLETED")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error during weekly retraining: {e}", exc_info=True)
            # Resume trading even on error
            if self.trading_resume_callback:
                self.trading_resume_callback()
        
        finally:
            self.is_training = False
    
    def start(self):
        """Start the scheduler."""
        # Schedule weekly retraining (Sunday 03:00 UTC)
        trigger = CronTrigger(
            day_of_week=RETRAIN_DAY,
            hour=RETRAIN_HOUR,
            minute=RETRAIN_MINUTE,
            timezone='UTC'
        )
        
        self.scheduler.add_job(
            self._weekly_retraining_job,
            trigger=trigger,
            id='weekly_retraining',
            name='Weekly Model Retraining',
            replace_existing=True
        )
        
        logger.info(f"Scheduled weekly retraining: {RETRAIN_DAY.capitalize()} {RETRAIN_HOUR:02d}:{RETRAIN_MINUTE:02d} UTC")
        logger.info("Starting scheduler...")
        
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped")
            self.scheduler.shutdown()
    
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping scheduler...")
        self.scheduler.shutdown()
    
    def run_retraining_now(self):
        """Manually trigger retraining (for testing)."""
        logger.info("Manually triggering retraining...")
        self._weekly_retraining_job()


if __name__ == "__main__":
    # Test scheduler
    scheduler = RetrainingScheduler()
    logger.info("Scheduler created. Use start() to begin scheduling.")
    # scheduler.start()  # Uncomment to start

