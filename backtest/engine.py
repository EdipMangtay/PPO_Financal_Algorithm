"""
Backtest Engine - Event-driven backtesting with exact metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    side: int  # 1 for LONG, -1 for SHORT
    pnl: float
    pnl_pct: float
    fees: float
    bars_held: int

class BacktestEngine:
    """Event-driven backtest engine."""
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_rate: float = 0.0004,  # 0.04% taker fee
        slippage: float = 0.0005,  # 0.05% slippage
        position_sizing: str = "fixed",  # "fixed" or "kelly"
        position_size: float = 0.1,  # 10% of balance for fixed, or kelly fraction
        enable_long: bool = True,
        enable_short: bool = True,
        max_leverage: float = 1.0,  # 1.0 = no leverage
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_balance: Starting capital
            fee_rate: Trading fee rate (per trade)
            slippage: Slippage rate (per trade)
            position_sizing: "fixed" or "kelly"
            position_size: Position size (fraction of balance or kelly fraction)
            enable_long: Allow long positions
            enable_short: Allow short positions
            max_leverage: Maximum leverage (1.0 = no leverage)
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.position_size = position_size
        self.enable_long = enable_long
        self.enable_short = enable_short
        self.max_leverage = max_leverage
        
    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        price_col: str = 'close',
        timestamp_col: str = 'timestamp'
    ) -> Dict:
        """
        Run backtest on signal series.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Series of signals (1=LONG, -1=SHORT, 0=FLAT)
            price_col: Column name for price
            timestamp_col: Column name for timestamp
        
        Returns:
            Dict with metrics and trades
        """
        # Ensure signals align with dataframe
        if len(signals) != len(df):
            raise ValueError(f"Signals length ({len(signals)}) != DataFrame length ({len(df)})")
        
        # Initialize state
        balance = self.initial_balance
        position = None  # None or Trade object
        equity_curve = [balance]
        trades = []
        
        # Ensure timestamp column exists
        if timestamp_col not in df.columns:
            df = df.copy()
            df[timestamp_col] = df.index
        
        # Run event-driven simulation
        for i in range(len(df)):
            current_price = df.iloc[i][price_col]
            current_signal = signals.iloc[i]
            current_time = df.iloc[i][timestamp_col]
            
            # Close existing position if signal changes or opposite signal
            if position is not None:
                should_close = False
                
                if current_signal == 0:  # FLAT signal
                    should_close = True
                elif current_signal != position.side:  # Opposite signal
                    should_close = True
                
                if should_close:
                    # Close position
                    exit_price = self._apply_slippage(current_price, -position.side)
                    pnl = self._calculate_pnl(
                        position.entry_price,
                        exit_price,
                        position.size,
                        position.side
                    )
                    fees = position.size * self.fee_rate * 2  # Entry + exit
                    net_pnl = pnl - fees
                    
                    balance += net_pnl
                    
                    trade = Trade(
                        entry_time=position.entry_time,
                        exit_time=current_time,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        size=position.size,
                        side=position.side,
                        pnl=net_pnl,
                        pnl_pct=(exit_price - position.entry_price) / position.entry_price * position.side * 100,
                        fees=fees,
                        bars_held=i - position.entry_bar
                    )
                    trades.append(trade)
                    position = None
            
            # Open new position if signal and no existing position
            if position is None and current_signal != 0:
                if current_signal == 1 and not self.enable_long:
                    continue
                if current_signal == -1 and not self.enable_short:
                    continue
                
                # Calculate position size
                position_value = self._calculate_position_size(balance, current_price)
                
                if position_value > 0:
                    entry_price = self._apply_slippage(current_price, current_signal)
                    fees = position_value * self.fee_rate
                    
                    position = type('Position', (), {
                        'entry_time': current_time,
                        'entry_price': entry_price,
                        'size': position_value,
                        'side': current_signal,
                        'entry_bar': i,
                    })()
            
            # Update equity (balance + unrealized PnL)
            if position is not None:
                unrealized_pnl = self._calculate_pnl(
                    position.entry_price,
                    current_price,
                    position.size,
                    position.side
                )
                equity = balance + unrealized_pnl
            else:
                equity = balance
            
            equity_curve.append(equity)
        
        # Close final position if exists
        if position is not None:
            final_price = df.iloc[-1][price_col]
            exit_price = self._apply_slippage(final_price, -position.side)
            pnl = self._calculate_pnl(
                position.entry_price,
                exit_price,
                position.size,
                position.side
            )
            fees = position.size * self.fee_rate * 2
            net_pnl = pnl - fees
            balance += net_pnl
            
            trade = Trade(
                entry_time=position.entry_time,
                exit_time=df.iloc[-1][timestamp_col],
                entry_price=position.entry_price,
                exit_price=exit_price,
                size=position.size,
                side=position.side,
                pnl=net_pnl,
                pnl_pct=(exit_price - position.entry_price) / position.entry_price * position.side * 100,
                fees=fees,
                bars_held=len(df) - 1 - position.entry_bar
            )
            trades.append(trade)
            equity_curve[-1] = balance
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, trades, df, timestamp_col)
        
        return {
            'metrics': metrics,
            'trades': trades,
            'equity_curve': equity_curve,
        }
    
    def _apply_slippage(self, price: float, side: int) -> float:
        """Apply slippage to price."""
        if side > 0:  # Long entry (buy) - slippage increases price
            return price * (1 + self.slippage)
        else:  # Short entry (sell) - slippage decreases price
            return price * (1 - self.slippage)
    
    def _calculate_pnl(self, entry_price: float, exit_price: float, size: float, side: int) -> float:
        """Calculate PnL for a trade."""
        return (exit_price - entry_price) * size * side
    
    def _calculate_position_size(self, balance: float, price: float) -> float:
        """Calculate position size in notional value."""
        if self.position_sizing == "fixed":
            return balance * self.position_size * self.max_leverage
        else:  # kelly
            # Simplified - in production would use actual kelly fraction
            return balance * self.position_size * self.max_leverage
    
    def _calculate_metrics(
        self,
        equity_curve: List[float],
        trades: List[Trade],
        df: pd.DataFrame,
        timestamp_col: str
    ) -> Dict:
        """Calculate exact backtest metrics."""
        equity_array = np.array(equity_curve)
        
        # Total return
        total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
        
        # CAGR (simplified - assumes daily bars)
        if len(df) > 0:
            start_time = df.iloc[0][timestamp_col]
            end_time = df.iloc[-1][timestamp_col]
            if pd.api.types.is_datetime64_any_dtype(type(start_time)):
                days = (end_time - start_time).days
                if days > 0:
                    years = days / 365.25
                    cagr = (equity_array[-1] / equity_array[0]) ** (1 / years) - 1
                else:
                    cagr = 0.0
            else:
                cagr = 0.0
        else:
            cagr = 0.0
        
        # Max drawdown
        peak = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Returns for Sharpe/Sortino
        returns = np.diff(equity_array) / equity_array[:-1]
        returns = returns[~np.isnan(returns)]
        
        # Sharpe ratio (annualized)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Assume daily
        else:
            sharpe = 0.0
        
        # Sortino ratio (annualized, only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        else:
            sortino = 0.0
        
        # Trade metrics
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / len(trades)
            total_profit = sum(t.pnl for t in winning_trades)
            total_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
            avg_trade_return = np.mean([t.pnl_pct for t in trades])
            expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
            
            avg_holding_period = np.mean([t.bars_held for t in trades])
        else:
            win_rate = 0.0
            total_profit = 0.0
            total_loss = 0.0
            profit_factor = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            avg_trade_return = 0.0
            expectancy = 0.0
            avg_holding_period = 0.0
        
        # Exposure (time in market)
        exposure = len([x for x in equity_array if x != equity_array[0]]) / len(equity_array) if len(equity_array) > 0 else 0.0
        
        # Turnover (simplified)
        turnover = sum(abs(t.size) for t in trades) / self.initial_balance if len(trades) > 0 else 0.0
        
        return {
            'total_return': float(total_return),
            'total_return_pct': float(total_return * 100),
            'cagr': float(cagr),
            'cagr_pct': float(cagr * 100),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown * 100),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'win_rate': float(win_rate),
            'win_rate_pct': float(win_rate * 100),
            'total_trades': len(trades),
            'winning_trades': len(winning_trades) if len(trades) > 0 else 0,
            'losing_trades': len(losing_trades) if len(trades) > 0 else 0,
            'total_profit': float(total_profit),
            'total_loss': float(total_loss),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'avg_trade_return': float(avg_trade_return),
            'expectancy': float(expectancy),
            'avg_holding_period_bars': float(avg_holding_period),
            'exposure': float(exposure),
            'exposure_pct': float(exposure * 100),
            'turnover': float(turnover),
            'final_equity': float(equity_array[-1]),
            'initial_balance': float(self.initial_balance),
        }

def save_backtest_results(
    results: Dict,
    output_dir: Path,
    timeframe: str
):
    """Save backtest results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    import json
    with open(output_dir / 'backtest_metrics.json', 'w') as f:
        json.dump(results['metrics'], f, indent=2, default=str)
    
    # Save trades
    if results['trades']:
        trades_df = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'side': 'LONG' if t.side > 0 else 'SHORT',
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'fees': t.fees,
                'bars_held': t.bars_held,
            }
            for t in results['trades']
        ])
        trades_df.to_csv(output_dir / 'trades.csv', index=False)
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'equity': results['equity_curve']
    })
    equity_df.to_csv(output_dir / 'equity.csv', index=False)
    
    logger.info(f"Saved backtest results to {output_dir}")





