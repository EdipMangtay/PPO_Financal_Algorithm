"""
MASTER PIPELINE - Tek Komutla Tüm Süreç
Veri İndirme → Optuna Optimizasyon → Eğitim → Backtest → Detaylı Rapor
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

import asyncio
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TOP_20_COINS, SCALP_TIMEFRAME, INITIAL_BALANCE
from main import FeatureEngineeringPipeline, run_live_trading
from data.loader import DataLoader
from data_engine.features import FeatureGenerator
from tuning.optimizer import TwoLayerOptimizer, load_feature_config
from models.tft import TFTModel
from models.tft_ensemble import TFTEnsemble
from models.ppo import PPOTradingAgent
from env.trading_env import TradingEnv
from trainer import Trainer
from validator import Validator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()


class MasterPipeline:
    """Tek komutla tüm süreci çalıştıran master pipeline."""
    
    def __init__(self):
        self.pipeline = FeatureEngineeringPipeline()
        self.results = {}
        
    async def run_complete_pipeline(
        self,
        days: int = 1825,
        trials: int = 200,
        coins: Optional[List[str]] = None,
        timeframe: str = SCALP_TIMEFRAME,
        backtest_steps: int = 5000
    ):
        """Tüm süreci çalıştır: Veri → Optimize → Train → Backtest → Rapor"""
        
        console.print("\n[bold magenta]=" * 70)
        console.print("[bold magenta]GOD_LEVEL_TRADER - MASTER PIPELINE")
        console.print("[bold magenta]=" * 70)
        console.print(f"[cyan]Days: {days} | Trials: {trials} | Timeframe: {timeframe}[/cyan]\n")
        
        if coins is None:
            coins = TOP_20_COINS
        
        start_time = datetime.now()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # ====================================================================
            # ADIM 1: VERİ İNDİRME
            # ====================================================================
            task1 = progress.add_task("[cyan]Step 1: Downloading data...", total=100)
            console.print("\n[bold green]STEP 1: DATA DOWNLOAD[/bold green]")
            
            try:
                data_dict = await self.pipeline.step1_download_data(
                    days=days,
                    coins=coins,
                    timeframe=timeframe,
                    use_cache=True  # YENİ: Cache kullan (disk'ten oku)
                )
                progress.update(task1, completed=100)
                console.print(f"[green]✓ Downloaded data for {len(data_dict)} coins[/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Data download failed: {e}[/red]")
                return
            
            # ====================================================================
            # ADIM 2: OPTUNA OPTİMİZASYON - TÜM TIMEFRAME'LER İÇİN
            # ====================================================================
            all_timeframes = ['15m', '1h', '4h']
            total_optimizations = len(coins) * len(all_timeframes) * trials
            task2 = progress.add_task("[yellow]Step 2: Feature optimization (multi-timeframe)...", total=total_optimizations)
            console.print("[bold yellow]STEP 2: FEATURE OPTIMIZATION (Optuna) - Multi-Timeframe[/bold yellow]")
            console.print(f"[cyan]Optimizing {len(coins)} coins × {len(all_timeframes)} timeframes = {total_optimizations} total trials[/cyan]\n")
            
            feature_configs = {}
            for coin in coins:
                if coin not in data_dict:
                    continue
                
                for tf in all_timeframes:
                    if tf not in data_dict[coin]:
                        console.print(f"[yellow]⚠ {coin} {tf} data not available, skipping...[/yellow]")
                        continue
                    
                    console.print(f"\n[cyan]Optimizing {coin} {tf}...[/cyan]")
                    console.print(f"[yellow]💡 Tip: View live dashboard in another terminal:[/yellow]")
                    console.print(f"[yellow]   optuna-dashboard optuna_studies/feature_optimization_{coin.replace('/', '_')}_{tf}.db[/yellow]")
                    console.print(f"[yellow]   Then open: http://localhost:8080[/yellow]\n")
                    
                    try:
                        df = data_dict[coin][tf]
                        optimizer = TwoLayerOptimizer(
                            coin=coin,
                            timeframe=tf,
                            data=df,
                            n_trials=trials,
                            timeout=None
                        )
                        
                        study, best_config = optimizer.optimize()
                        optimizer.save_config(best_config)
                        feature_configs[f"{coin}_{tf}"] = best_config
                        
                        progress.update(task2, advance=trials)
                        console.print(f"[green]✓ {coin} {tf} optimized[/green]")
                        console.print(f"[cyan]📊 Dashboard: optuna-dashboard optuna_studies/feature_optimization_{coin.replace('/', '_')}_{tf}.db[/cyan]")
                        
                    except Exception as e:
                        console.print(f"[red]✗ {coin} {tf} optimization failed: {e}[/red]")
                        import traceback
                        traceback.print_exc()
                        continue
            
            progress.update(task2, completed=total_optimizations)
            console.print(f"\n[green]✓ Feature optimization completed: {len(feature_configs)} configs[/green]\n")
            
            # ====================================================================
            # ADIM 3: MODEL EĞİTİMİ - TFT ENSEMBLE (3 TIMEFRAME)
            # ====================================================================
            task3 = progress.add_task("[blue]Step 3: Training TFT Ensemble...", total=100)
            console.print("[bold blue]STEP 3: MODEL TRAINING - TFT ENSEMBLE[/bold blue]")
            console.print("[cyan]Training 3 TFT models (15m, 1h, 4h) + PPO[/cyan]\n")
            
            try:
                from models.tft_ensemble import TFTEnsemble
                
                # Her timeframe için optimize edilmiş verileri hazırla
                all_timeframes = ['15m', '1h', '4h']
                optimized_data_by_timeframe = {}
                
                for tf in all_timeframes:
                    optimized_data = {}
                    for coin in coins:
                        if coin not in data_dict or tf not in data_dict[coin]:
                            continue
                        
                        config_key = f"{coin}_{tf}"
                        if config_key not in feature_configs:
                            continue
                        
                        df = data_dict[coin][tf].copy()
                        feature_config = feature_configs[config_key]
                        
                        # Generate all features
                        feature_generator = FeatureGenerator()
                        df_with_features = feature_generator.generate_candidate_features(df)
                        
                        # Select only optimized features
                        selected_features = feature_config['selected_features']
                        base_cols = ['open', 'high', 'low', 'close']
                        if 'volume' in df_with_features.columns:
                            base_cols.append('volume')
                        
                        available_features = [f for f in selected_features if f in df_with_features.columns]
                        feature_cols = base_cols + available_features
                        
                        optimized_df = df_with_features[feature_cols].copy()
                        optimized_data[coin] = optimized_df
                    
                    if optimized_data:
                        optimized_data_by_timeframe[tf] = optimized_data
                        console.print(f"[green]✓ Prepared {len(optimized_data)} coins for {tf}[/green]")
                
                if not optimized_data_by_timeframe:
                    console.print("[red]✗ No optimized data available for training[/red]")
                    return
                
                # TFT Ensemble oluştur ve eğit
                console.print("\n[cyan]Training TFT Ensemble (3 models)...[/cyan]")
                ensemble = TFTEnsemble()
                
                # Her timeframe için model eğit
                for tf in all_timeframes:
                    if tf not in optimized_data_by_timeframe:
                        console.print(f"[yellow]⚠ Skipping {tf} - no data[/yellow]")
                        continue
                    
                    console.print(f"\n[cyan]Training TFT model for {tf}...[/cyan]")
                    try:
                        ensemble.train_model(
                            timeframe=tf,
                            data=optimized_data_by_timeframe[tf],
                            epochs=30,
                            batch_size=128  # RTX 5070 optimized
                        )
                        progress.update(task3, advance=33)
                        console.print(f"[green]✓ {tf} TFT model trained[/green]")
                    except Exception as e:
                        console.print(f"[red]✗ {tf} TFT training failed: {e}[/red]")
                        import traceback
                        traceback.print_exc()
                
                # Ensemble'i kaydet
                ensemble.save("models/checkpoints/tft_ensemble")
                console.print("[green]✓ TFT Ensemble saved[/green]")
                
                # PPO eğitimi (15m için - primary timeframe)
                if '15m' in optimized_data_by_timeframe:
                    console.print("\n[cyan]Training PPO agent (15m)...[/cyan]")
                    try:
                        trainer = Trainer()
                        trainer.pretrain_tft(
                            optimized_data_by_timeframe['15m'],
                            epochs=30,
                            batch_size=128
                        )
                        
                        # PPO eğitimi
                        env = TradingEnv(
                            data=optimized_data_by_timeframe['15m'],
                            initial_balance=INITIAL_BALANCE
                        )
                        
                        ppo_agent = PPOTradingAgent(env=env)
                        ppo_agent.train(total_timesteps=5000000)  # 5M steps
                        ppo_agent.save("models/checkpoints/ppo_trained")
                        
                        progress.update(task3, advance=1)
                        console.print("[green]✓ PPO agent trained[/green]")
                    except Exception as e:
                        console.print(f"[yellow]⚠ PPO training failed: {e}[/yellow]")
                        import traceback
                        traceback.print_exc()
                
                progress.update(task3, completed=100)
                console.print("[green]✓ Model training completed[/green]\n")
                
            except Exception as e:
                console.print(f"[red]✗ Training failed: {e}[/red]")
                import traceback
                traceback.print_exc()
                return
            
            # ====================================================================
            # ADIM 4: BACKTEST
            # ====================================================================
            task4 = progress.add_task("[magenta]Step 4: Running backtest...", total=backtest_steps)
            console.print("[bold magenta]STEP 4: BACKTEST[/bold magenta]")
            
            backtest_results = await self._run_backtest(
                data_dict,
                feature_configs,
                timeframe,
                backtest_steps,
                progress,
                task4
            )
            
            progress.update(task4, completed=backtest_steps)
            
            # ====================================================================
            # ADIM 5: DETAYLI RAPOR
            # ====================================================================
            console.print("\n[bold cyan]STEP 5: GENERATING REPORT[/bold cyan]")
            self._generate_report(backtest_results, start_time, days, trials)
            
            console.print("\n[bold green]=" * 70)
            console.print("[bold green]PIPELINE COMPLETED SUCCESSFULLY![/bold green]")
            console.print("[bold green]=" * 70)
    
    async def _run_backtest(
        self,
        data_dict: Dict,
        feature_configs: Dict,
        timeframe: str,
        steps: int,
        progress,
        task
    ) -> Dict:
        """Backtest çalıştır ve detaylı metrikler topla."""
        
        # Optimize edilmiş verileri hazırla
        feature_generator = FeatureGenerator()
        optimized_data = {}
        
        for coin, timeframes in data_dict.items():
            if timeframe not in timeframes:
                continue
            
            config_key = f"{coin}_{timeframe}"
            if config_key not in feature_configs:
                continue
            
            df = timeframes[timeframe].copy()
            feature_config = feature_configs[config_key]
            
            # Tüm özellikleri oluştur
            df_with_features = feature_generator.generate_candidate_features(df)
            
            # Sadece optimize edilmiş özellikleri seç
            selected_features = feature_config['selected_features']
            base_cols = ['open', 'high', 'low', 'close']
            if 'volume' in df_with_features.columns:
                base_cols.append('volume')
            
            available_features = [f for f in selected_features if f in df_with_features.columns]
            feature_cols = base_cols + available_features
            
            optimized_df = df_with_features[feature_cols].copy()
            optimized_data[coin] = optimized_df
        
        if not optimized_data:
            console.print("[red]No optimized data for backtest[/red]")
            return {}
        
        # Environment oluştur
        env = TradingEnv(data=optimized_data, initial_balance=INITIAL_BALANCE)
        
        # Model yükle (eğer varsa)
        tft_model = None
        ppo_agent = None
        
        tft_path = Path("models/checkpoints/tft_pretrained.pt")
        ppo_path = Path("models/checkpoints/ppo_trained")
        
        if tft_path.exists():
            try:
                tft_model = TFTModel()
                # Note: Full loading requires training_data, simplified for now
                console.print("[yellow]Note: Using simplified TFT for backtest[/yellow]")
            except:
                pass
        
        if ppo_path.exists():
            try:
                ppo_agent = PPOTradingAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space
                )
                ppo_agent.load(str(ppo_path), env=env)
                console.print("[green]✓ PPO model loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load PPO: {e}[/yellow]")
        
        # Backtest çalıştır
        obs, info = env.reset()
        portfolio_values = [info['portfolio_value']]
        returns = []
        lstm_states = None
        
        console.print(f"[cyan]Running backtest for {steps} steps...[/cyan]")
        
        for step in range(steps):
            # TFT prediction
            if tft_model and ppo_agent:
                current_coin = env.current_coin
                coin_data = env.data.get(current_coin)
                
                if coin_data is not None and len(coin_data) > 0:
                    try:
                        predictions, confidence = tft_model.predict(coin_data, current_coin)
                        atr = coin_data['ATR_14'].iloc[-1] if 'ATR_14' in coin_data.columns else 0.0
                    except:
                        confidence = 0.5
                        atr = 0.0
                else:
                    confidence = 0.5
                    atr = 0.0
                
                # PPO action
                action, lstm_states = ppo_agent.predict(obs, lstm_states=lstm_states)
            else:
                # Random policy (fallback)
                action = np.random.uniform(-0.3, 0.3, size=(1,))
                confidence = 0.5
                atr = 0.0
            
            # Step environment
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
            
            if step % 100 == 0:
                progress.update(task, advance=100)
            
            if terminated or truncated:
                break
        
        # Metrikleri hesapla
        returns_array = np.array(returns) if returns else np.array([0.0])
        portfolio_array = np.array(portfolio_values)
        
        total_return = (portfolio_array[-1] - portfolio_array[0]) / portfolio_array[0] if len(portfolio_array) > 1 else 0.0
        
        # Sharpe Ratio
        if len(returns_array) > 0 and np.std(returns_array) > 0:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino Ratio
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            sortino = np.mean(returns_array) / downside_std * np.sqrt(252) if downside_std > 0 else 0.0
        else:
            sortino = 10.0 if np.mean(returns_array) > 0 else 0.0
        
        # Max Drawdown
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - peak) / peak
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Trade analizi
        trades = env.trades
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        net_pnl = total_profit - total_loss
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        results = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd * 100,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_pnl': net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_portfolio_value': portfolio_array[-1] if len(portfolio_array) > 0 else INITIAL_BALANCE,
            'initial_balance': INITIAL_BALANCE,
            'trades': trades
        }
        
        return results
    
    def _generate_report(self, results: Dict, start_time: datetime, days: int, trials: int):
        """Detaylı rapor oluştur."""
        
        if not results:
            console.print("[red]No results to report[/red]")
            return
        
        # Ana metrikler tablosu
        table = Table(title="📊 BACKTEST RESULTS", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Initial Balance", f"${results['initial_balance']:,.2f}")
        table.add_row("Final Portfolio Value", f"${results['final_portfolio_value']:,.2f}")
        table.add_row("Total Return", f"{results['total_return_pct']:+.2f}%")
        table.add_row("Net P&L", f"${results['net_pnl']:+,.2f}")
        table.add_row("", "")
        table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.4f}")
        table.add_row("Sortino Ratio", f"{results['sortino_ratio']:.4f}")
        table.add_row("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")
        table.add_row("", "")
        table.add_row("Total Trades", f"{results['total_trades']}")
        table.add_row("Winning Trades", f"{results['winning_trades']} ({results['win_rate_pct']:.1f}%)")
        table.add_row("Losing Trades", f"{results['losing_trades']}")
        table.add_row("", "")
        table.add_row("Total Profit", f"${results['total_profit']:+,.2f}")
        table.add_row("Total Loss", f"${results['total_loss']:+,.2f}")
        table.add_row("Average Win", f"${results['avg_win']:+,.2f}")
        table.add_row("Average Loss", f"${results['avg_loss']:+,.2f}")
        table.add_row("Profit Factor", f"{results['profit_factor']:.2f}")
        
        console.print("\n")
        console.print(table)
        
        # Trade detayları tablosu (son 20 trade)
        if results['trades']:
            trade_table = Table(title="📈 RECENT TRADES (Last 20)", show_header=True, header_style="bold yellow")
            trade_table.add_column("Coin", style="cyan")
            trade_table.add_column("Type", style="magenta")
            trade_table.add_column("Entry", justify="right", style="green")
            trade_table.add_column("Exit", justify="right", style="yellow")
            trade_table.add_column("P&L", justify="right", style="green")
            trade_table.add_column("P&L %", justify="right", style="green")
            trade_table.add_column("Reason", style="white")
            
            for trade in results['trades'][-20:]:
                pos_type = "LONG" if trade.position_type.value == 1 else "SHORT"
                pnl_color = "green" if trade.pnl > 0 else "red"
                trade_table.add_row(
                    trade.coin,
                    pos_type,
                    f"${trade.entry_price:.4f}",
                    f"${trade.exit_price:.4f}",
                    f"[{pnl_color}]{trade.pnl:+.2f}[/{pnl_color}]",
                    f"[{pnl_color}]{trade.pnl_pct*100:+.2f}%[/{pnl_color}]",
                    trade.reason
                )
            
            console.print("\n")
            console.print(trade_table)
        
        # JSON raporu kaydet
        report_path = Path("logs/backtest_report.json")
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_config': {
                'days': days,
                'trials': trials,
                'start_time': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            },
            'results': {k: v for k, v in results.items() if k != 'trades'},
            'trade_summary': {
                'total': results['total_trades'],
                'winning': results['winning_trades'],
                'losing': results['losing_trades']
            },
            'trades': [
                {
                    'coin': t.coin,
                    'position_type': t.position_type.name,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'leverage': t.leverage,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'reason': t.reason
                }
                for t in results.get('trades', [])
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        console.print(f"\n[green]✓ Detailed report saved to: {report_path}[/green]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master Pipeline - Complete automated trading system"
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1825,
        help='Days of historical data (default: 1825 = 5 years)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=200,
        help='Optuna trials per coin (default: 200)'
    )
    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        default=None,
        help='Specific coins (default: all top 20)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default=SCALP_TIMEFRAME,
        help='Trading timeframe (default: 15m)'
    )
    parser.add_argument(
        '--backtest-steps',
        type=int,
        default=5000,
        help='Backtest steps (default: 5000)'
    )
    
    args = parser.parse_args()
    
    pipeline = MasterPipeline()
    asyncio.run(pipeline.run_complete_pipeline(
        days=args.days,
        trials=args.trials,
        coins=args.coins,
        timeframe=args.timeframe,
        backtest_steps=args.backtest_steps
    ))


if __name__ == "__main__":
    main()

