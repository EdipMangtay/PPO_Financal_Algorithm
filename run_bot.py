"""
GOD_LEVEL_TRADER_V5 - Master CLI
Main entry point with rich dashboard and training/live modes.
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
import time
import threading
from typing import Optional, Dict
from datetime import datetime
import logging
from pathlib import Path

import torch
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOP_20_COINS,
    SCALP_TIMEFRAME,
    INITIAL_BALANCE,
    CONFIDENCE_THRESHOLD,
    DASHBOARD_REFRESH_RATE
)
from data.loader import DataLoader
from models.tft import TFTModel
from models.ppo import PPOTradingAgent
from env.trading_env import TradingEnv
from trainer import Trainer
from validator import Validator
from scheduler import RetrainingScheduler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Force float32 matmul precision for RTX 5070
torch.set_float32_matmul_precision('medium')

console = Console()


class TradingDashboard:
    """Rich terminal dashboard for live trading."""
    
    def __init__(self, env: TradingEnv, tft_model: Optional[TFTModel], ppo_agent: Optional[PPOTradingAgent]):
        """Initialize dashboard."""
        self.env = env
        self.tft_model = tft_model
        self.ppo_agent = ppo_agent
        self.start_time = datetime.now()
        self.trading_paused = False
        
    def create_portfolio_panel(self) -> Panel:
        """Create portfolio metrics panel."""
        portfolio_value = self.env.portfolio_value
        balance = self.env.balance
        pnl = portfolio_value - INITIAL_BALANCE
        pnl_pct = (pnl / INITIAL_BALANCE) * 100
        
        # Calculate drawdown
        drawdown = (self.env.max_portfolio_value - portfolio_value) / self.env.max_portfolio_value * 100
        
        text = Text()
        text.append(f"Portfolio Value: ", style="bold")
        text.append(f"${portfolio_value:,.2f}\n", style="green" if pnl >= 0 else "red")
        text.append(f"Balance: ", style="bold")
        text.append(f"${balance:,.2f}\n", style="white")
        text.append(f"PnL: ", style="bold")
        text.append(f"${pnl:,.2f} ({pnl_pct:+.2f}%)\n", style="green" if pnl >= 0 else "red")
        text.append(f"Max Drawdown: ", style="bold")
        text.append(f"{drawdown:.2f}%\n", style="red" if drawdown > 15 else "yellow" if drawdown > 10 else "green")
        text.append(f"Total Trades: ", style="bold")
        text.append(f"{self.env.total_trades}\n", style="cyan")
        text.append(f"Open Positions: ", style="bold")
        text.append(f"{len(self.env.positions)}", style="cyan")
        
        return Panel(text, title="Portfolio", border_style="blue")
    
    def create_positions_table(self) -> Table:
        """Create active positions table."""
        table = Table(title="Active Positions", box=box.ROUNDED)
        table.add_column("Coin", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Entry Price", justify="right", style="green")
        table.add_column("Current Price", justify="right", style="yellow")
        table.add_column("Size", justify="right", style="white")
        table.add_column("Leverage", justify="right", style="white")
        table.add_column("PnL", justify="right", style="green")
        table.add_column("PnL %", justify="right", style="green")
        
        for coin, position in self.env.positions.items():
            current_price = self.env._get_current_price(coin)
            
            if position.position_type.value == 1:  # LONG
                pnl_pct = ((current_price - position.entry_price) / position.entry_price) * position.leverage * 100
                pos_type = "LONG"
            else:  # SHORT
                pnl_pct = ((position.entry_price - current_price) / position.entry_price) * position.leverage * 100
                pos_type = "SHORT"
            
            pnl = position.size * (pnl_pct / 100)
            
            table.add_row(
                coin,
                pos_type,
                f"${position.entry_price:.4f}",
                f"${current_price:.4f}",
                f"${position.size:.2f}",
                f"{position.leverage:.1f}x",
                f"${pnl:+.2f}",
                f"{pnl_pct:+.2f}%"
            )
        
        if len(self.env.positions) == 0:
            table.add_row("No open positions", "", "", "", "", "", "", "")
        
        return table
    
    def create_recent_trades_table(self) -> Table:
        """Create recent trades table."""
        table = Table(title="Recent Trades", box=box.ROUNDED)
        table.add_column("Coin", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Entry", justify="right", style="green")
        table.add_column("Exit", justify="right", style="yellow")
        table.add_column("PnL", justify="right", style="green")
        table.add_column("PnL %", justify="right", style="green")
        table.add_column("Reason", style="white")
        
        # Show last 10 trades
        recent_trades = self.env.trades[-10:] if len(self.env.trades) > 0 else []
        
        for trade in reversed(recent_trades):
            pos_type = "LONG" if trade.position_type.value == 1 else "SHORT"
            table.add_row(
                trade.coin,
                pos_type,
                f"${trade.entry_price:.4f}",
                f"${trade.exit_price:.4f}",
                f"${trade.pnl:+.2f}",
                f"{trade.pnl_pct:+.2f}%",
                trade.reason
            )
        
        if len(recent_trades) == 0:
            table.add_row("No trades yet", "", "", "", "", "", "")
        
        return table
    
    def create_status_panel(self) -> Panel:
        """Create status panel."""
        uptime = datetime.now() - self.start_time
        current_coin = self.env.current_coin
        current_step = self.env.current_step
        
        text = Text()
        text.append(f"Status: ", style="bold")
        text.append(f"{'PAUSED' if self.trading_paused else 'LIVE'}\n", 
                   style="red" if self.trading_paused else "green")
        text.append(f"Uptime: ", style="bold")
        text.append(f"{str(uptime).split('.')[0]}\n", style="white")
        text.append(f"Current Coin: ", style="bold")
        text.append(f"{current_coin}\n", style="cyan")
        text.append(f"Step: ", style="bold")
        text.append(f"{current_step}", style="white")
        
        return Panel(text, title="Status", border_style="green")
    
    def render(self) -> Layout:
        """Render the dashboard."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=7)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(self.create_portfolio_panel(), name="portfolio"),
            Layout(self.create_status_panel(), name="status")
        )
        
        layout["right"].split_column(
            Layout(self.create_positions_table(), name="positions"),
            Layout(self.create_recent_trades_table(), name="trades")
        )
        
        # Header
        header_text = Text("GOD_LEVEL_TRADER_V5 - Context-Aware HFT System", style="bold magenta", justify="center")
        layout["header"].update(Panel(header_text, border_style="magenta"))
        
        # Footer
        footer_text = Text(
            f"Press Ctrl+C to stop | Confidence Threshold: {CONFIDENCE_THRESHOLD} | "
            f"Timeframe: {SCALP_TIMEFRAME}",
            style="dim white",
            justify="center"
        )
        layout["footer"].update(Panel(footer_text, border_style="dim"))
        
        return layout


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self):
        """Initialize trading bot."""
        self.tft_model: Optional[TFTModel] = None
        self.ppo_agent: Optional[PPOTradingAgent] = None
        self.env: Optional[TradingEnv] = None
        self.trainer: Optional[Trainer] = None
        self.validator: Optional[Validator] = None
        self.scheduler: Optional[RetrainingScheduler] = None
        self.dashboard: Optional[TradingDashboard] = None
        self.trading_paused = False
        self.running = False
        
    def load_models(self):
        """Load pre-trained models."""
        logger.info("Loading models...")
        
        # Load TFT
        tft_path = Path("models/checkpoints/tft_pretrained.pt")
        if tft_path.exists():
            self.tft_model = TFTModel()
            # Note: Need training_data to load, will handle in training mode
            logger.info("TFT model path found (will load during initialization)")
        else:
            logger.warning("TFT model not found. Will need to train.")
        
        # Load PPO
        ppo_path = Path("models/checkpoints/ppo_trained")
        if ppo_path.exists() and self.env:
            self.ppo_agent = PPOTradingAgent(
                observation_space=self.env.observation_space,
                action_space=self.env.action_space
            )
            self.ppo_agent.load(str(ppo_path), env=self.env)
            logger.info("PPO agent loaded")
        else:
            logger.warning("PPO agent not found. Will need to train.")
    
    def train_mode(self, use_optuna: bool = True):
        """Run training mode."""
        console.print("[bold green]Starting Training Mode[/bold green]")
        
        # Step 1: Fetch Data
        console.print("[cyan]Step 1: Fetching data...[/cyan]")
        loader = DataLoader()
        data = asyncio.run(loader.fetch_recent(days=30))
        
        if not data:
            console.print("[bold red]Failed to fetch data. Exiting.[/bold red]")
            return
        
        console.print(f"[green]Fetched data for {len(data)} coins[/green]")
        
        # Step 2: Optimize TFT (optional)
        if use_optuna:
            console.print("[cyan]Step 2: Optimizing TFT hyperparameters with Optuna...[/cyan]")
            self.trainer = Trainer()
            best_params = self.trainer.optimize_tft_hyperparameters(data)
            console.print(f"[green]Best TFT params: {best_params}[/green]")
        else:
            self.trainer = Trainer()
        
        # Step 3: Pretrain TFT
        console.print("[cyan]Step 3: Pretraining TFT model...[/cyan]")
        self.tft_model = self.trainer.pretrain_tft(data, epochs=20)
        console.print("[green]TFT model pretrained[/green]")
        
        # Step 4: Create Environment
        console.print("[cyan]Step 4: Creating trading environment...[/cyan]")
        self.env = TradingEnv(data)
        
        # Step 5: Train PPO
        console.print("[cyan]Step 5: Training PPO agent...[/cyan]")
        self.ppo_agent = self.trainer.train_ppo(self.env, total_timesteps=100000)
        console.print("[green]PPO agent trained[/green]")
        
        console.print("[bold green]Training completed![/bold green]")
    
    def live_mode(self):
        """Run live trading mode."""
        console.print("[bold green]Starting Live Trading Mode[/bold green]")
        
        # Load data
        console.print("[cyan]Loading data...[/cyan]")
        loader = DataLoader()
        data = asyncio.run(loader.fetch_recent(days=7))
        
        if not data:
            console.print("[bold red]Failed to load data. Exiting.[/bold red]")
            return
        
        # Create environment
        self.env = TradingEnv(data)
        
        # Load models
        self.load_models()
        
        if self.tft_model is None or self.ppo_agent is None:
            console.print("[bold red]Models not loaded. Run training mode first.[/bold red]")
            return
        
        # Initialize scheduler
        self.trainer = Trainer(self.tft_model, self.ppo_agent)
        self.validator = Validator()
        self.scheduler = RetrainingScheduler(
            tft_model=self.tft_model,
            ppo_agent=self.ppo_agent,
            trainer=self.trainer,
            validator=self.validator,
            trading_pause_callback=self.pause_trading,
            trading_resume_callback=self.resume_trading
        )
        
        # Start scheduler in background
        scheduler_thread = threading.Thread(target=self.scheduler.start, daemon=True)
        scheduler_thread.start()
        
        # Create dashboard
        self.dashboard = TradingDashboard(self.env, self.tft_model, self.ppo_agent)
        
        # Main trading loop
        self.running = True
        obs, info = self.env.reset()
        lstm_states = None
        
        try:
            with Live(self.dashboard.render(), refresh_per_second=1/DASHBOARD_REFRESH_RATE) as live:
                while self.running:
                    if not self.trading_paused:
                        # Get TFT prediction
                        current_coin = self.env.current_coin
                        coin_data = self.env.data.get(current_coin)
                        
                        if coin_data is not None and len(coin_data) > 0:
                            try:
                                predictions, confidence = self.tft_model.predict(coin_data, current_coin)
                                atr = coin_data['ATR'].iloc[-1] if 'ATR' in coin_data.columns else 0.0
                            except Exception as e:
                                logger.warning(f"TFT prediction failed: {e}")
                                confidence = 0.5
                                atr = 0.0
                        else:
                            confidence = 0.5
                            atr = 0.0
                        
                        # Get PPO action
                        action, lstm_states = self.ppo_agent.predict(obs, lstm_states=lstm_states)
                        
                        # Step environment (using ensemble: same confidence for all timeframes if single model)
                        obs, reward, terminated, truncated, info = self.env.step(
                            action,
                            tft_confidence_15m=confidence,
                            tft_confidence_1h=confidence,
                            tft_confidence_4h=confidence,
                            atr=atr
                        )
                        
                        if terminated or truncated:
                            logger.warning("Environment terminated. Resetting...")
                            obs, info = self.env.reset()
                            lstm_states = None
                    
                    # Update dashboard
                    live.update(self.dashboard.render())
                    
                    time.sleep(DASHBOARD_REFRESH_RATE)
        
        except KeyboardInterrupt:
            console.print("\n[bold yellow]Stopping trading bot...[/bold yellow]")
            self.running = False
            if self.scheduler:
                self.scheduler.stop()
    
    def pause_trading(self):
        """Pause trading (for retraining)."""
        self.trading_paused = True
        if self.dashboard:
            self.dashboard.trading_paused = True
        logger.info("Trading paused")
    
    def resume_trading(self):
        """Resume trading (after retraining)."""
        self.trading_paused = False
        if self.dashboard:
            self.dashboard.trading_paused = False
        logger.info("Trading resumed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GOD_LEVEL_TRADER_V5 - Context-Aware HFT System")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'live'],
        required=True,
        help='Mode: train (train models) or live (run trading bot)'
    )
    parser.add_argument(
        '--no-optuna',
        action='store_true',
        help='Skip Optuna hyperparameter optimization (faster training)'
    )
    
    args = parser.parse_args()
    
    bot = TradingBot()
    
    if args.mode == 'train':
        bot.train_mode(use_optuna=not args.no_optuna)
    elif args.mode == 'live':
        bot.live_mode()
    else:
        console.print("[bold red]Invalid mode. Use 'train' or 'live'.[/bold red]")


if __name__ == "__main__":
    main()

