"""
Realistic Backtest for 4h Model with $1000 Initial Balance
Includes Binance fees and proper position sizing
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).parent.parent

def calculate_binance_fees(trade_value, fee_tier='regular'):
    """
    Calculate Binance trading fees
    
    Fee tiers:
    - regular: 0.1% maker, 0.1% taker
    - vip1: 0.09% maker, 0.1% taker
    - bnb_discount: 0.075% (25% discount with BNB)
    """
    fee_rates = {
        'regular': 0.001,      # 0.1%
        'vip1': 0.001,         # 0.1% taker
        'bnb_discount': 0.00075  # 0.075%
    }
    
    return trade_value * fee_rates.get(fee_tier, 0.001)

def realistic_backtest_4h(
    predictions_file: Path,
    test_data_file: Path,
    initial_balance: float = 1000.0,
    max_leverage: float = 5.0,
    position_size_pct: float = 0.2,
    signal_threshold: float = 0.0001,
    fee_tier: str = 'regular'
):
    """
    Run realistic backtest for 4h model
    """
    
    # Load predictions
    preds_df = pd.read_parquet(predictions_file)
    
    # Load test data (for actual prices)
    if test_data_file.suffix == '.parquet':
        test_data = pd.read_parquet(test_data_file)
    else:
        test_data = pd.read_csv(test_data_file)
    
    # Extract prediction values (median quantile)
    if 'prediction' in preds_df.columns:
        predictions = preds_df['prediction'].values
    elif 'y_pred' in preds_df.columns:
        predictions = preds_df['y_pred'].values
    else:
        # If predictions are in columns, take median (column 3 for 0.5 quantile)
        pred_cols = [col for col in preds_df.columns if col.startswith('pred') or col.startswith('q_')]
        if len(pred_cols) >= 7:
            predictions = preds_df[pred_cols[3]].values  # Median quantile
        else:
            predictions = preds_df.iloc[:, 0].values  # First column
    
    # Align lengths
    min_len = min(len(predictions), len(test_data))
    predictions = predictions[:min_len]
    test_data = test_data.iloc[:min_len].reset_index(drop=True)
    
    # Initialize backtest state
    balance = initial_balance
    position = 0.0  # Current position size (BTC)
    entry_price = 0.0
    position_type = None  # 'long' or 'short'
    
    # Track metrics
    trades = []
    equity_curve = [initial_balance]
    
    for i in range(len(predictions)):
        current_price = test_data.loc[i, 'close']
        pred_return = predictions[i]
        
        # Generate signal
        signal = 0
        if pred_return > signal_threshold:
            signal = 1  # Long
        elif pred_return < -signal_threshold:
            signal = -1  # Short
        
        # Close existing position if signal changes
        if position != 0 and ((signal == 1 and position_type == 'short') or 
                               (signal == -1 and position_type == 'long') or 
                               signal == 0):
            # Calculate P&L
            if position_type == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # short
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Apply leverage
            leveraged_pnl_pct = pnl_pct * max_leverage
            
            # Calculate position value
            position_value = abs(position) * current_price
            pnl_usd = position_value * leveraged_pnl_pct
            
            # Exit fees
            exit_fee = calculate_binance_fees(position_value, fee_tier)
            
            # Net P&L
            net_pnl = pnl_usd - exit_fee
            
            # Update balance
            balance += net_pnl
            
            # Record trade
            trades.append({
                'entry_idx': entry_idx,
                'exit_idx': i,
                'type': position_type,
                'entry_price': entry_price,
                'exit_price': current_price,
                'position_size': abs(position),
                'pnl_pct': pnl_pct,
                'leveraged_pnl_pct': leveraged_pnl_pct,
                'pnl_usd': pnl_usd,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'net_pnl': net_pnl,
                'balance_after': balance
            })
            
            # Close position
            position = 0.0
            position_type = None
        
        # Open new position if signal and no position
        if position == 0 and signal != 0 and balance > 0:
            # Calculate position size
            risk_capital = balance * position_size_pct
            position_value = risk_capital * max_leverage  # Leveraged position
            
            # Calculate position size in BTC
            position = position_value / current_price
            
            # Entry fees
            entry_fee = calculate_binance_fees(position_value, fee_tier)
            
            # Deduct entry fee from balance
            balance -= entry_fee
            
            # Record entry
            entry_price = current_price
            entry_idx = i
            position_type = 'long' if signal == 1 else 'short'
        
        # Update equity curve
        if position != 0:
            # Mark-to-market P&L
            if position_type == 'long':
                unrealized_pnl_pct = (current_price - entry_price) / entry_price
            else:
                unrealized_pnl_pct = (entry_price - current_price) / entry_price
            
            position_value = abs(position) * current_price
            unrealized_pnl = position_value * unrealized_pnl_pct * max_leverage
            current_equity = balance + unrealized_pnl
        else:
            current_equity = balance
        
        equity_curve.append(max(0, current_equity))  # Prevent negative equity
    
    # Close any remaining position
    if position != 0:
        current_price = test_data.loc[len(test_data)-1, 'close']
        
        if position_type == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
        
        leveraged_pnl_pct = pnl_pct * max_leverage
        position_value = abs(position) * current_price
        pnl_usd = position_value * leveraged_pnl_pct
        exit_fee = calculate_binance_fees(position_value, fee_tier)
        net_pnl = pnl_usd - exit_fee
        balance += net_pnl
        
        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': len(test_data)-1,
            'type': position_type,
            'entry_price': entry_price,
            'exit_price': current_price,
            'position_size': abs(position),
            'pnl_pct': pnl_pct,
            'leveraged_pnl_pct': leveraged_pnl_pct,
            'pnl_usd': pnl_usd,
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'net_pnl': net_pnl,
            'balance_after': balance
        })
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    
    if len(trades_df) == 0:
        return {
            'error': 'No trades executed',
            'initial_balance': initial_balance,
            'final_balance': balance,
            'total_trades': 0
        }
    
    winning_trades = trades_df[trades_df['net_pnl'] > 0]
    losing_trades = trades_df[trades_df['net_pnl'] < 0]
    
    total_fees = trades_df['entry_fee'].sum() + trades_df['exit_fee'].sum()
    
    metrics = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'net_profit': balance - initial_balance,
        'net_profit_pct': ((balance - initial_balance) / initial_balance) * 100,
        'total_return': balance / initial_balance,
        
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0,
        'win_rate_pct': (len(winning_trades) / len(trades_df) * 100) if len(trades_df) > 0 else 0,
        
        'total_fees_paid': total_fees,
        'fees_pct_of_initial': (total_fees / initial_balance) * 100,
        
        'gross_profit': winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0,
        'gross_loss': abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0,
        'profit_factor': (winning_trades['net_pnl'].sum() / abs(losing_trades['net_pnl'].sum())) if len(losing_trades) > 0 and losing_trades['net_pnl'].sum() != 0 else float('inf'),
        
        'avg_win': winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0,
        'avg_loss': losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0,
        'largest_win': winning_trades['net_pnl'].max() if len(winning_trades) > 0 else 0,
        'largest_loss': losing_trades['net_pnl'].min() if len(losing_trades) > 0 else 0,
        
        'avg_trade_duration': (trades_df['exit_idx'] - trades_df['entry_idx']).mean(),
        'longest_trade': (trades_df['exit_idx'] - trades_df['entry_idx']).max(),
        'shortest_trade': (trades_df['exit_idx'] - trades_df['entry_idx']).min(),
        
        # Drawdown
        'max_equity': max(equity_curve),
        'max_drawdown': max(equity_curve) - min([equity_curve[i] for i in range(equity_curve.index(max(equity_curve)), len(equity_curve))]) if equity_curve.index(max(equity_curve)) < len(equity_curve) - 1 else 0,
        'max_drawdown_pct': ((max(equity_curve) - min([equity_curve[i] for i in range(equity_curve.index(max(equity_curve)), len(equity_curve))])) / max(equity_curve) * 100) if equity_curve.index(max(equity_curve)) < len(equity_curve) - 1 and max(equity_curve) > 0 else 0,
        
        # Risk metrics
        'sharpe_ratio': (trades_df['net_pnl'].mean() / trades_df['net_pnl'].std()) if trades_df['net_pnl'].std() > 0 else 0,
        'expectancy': trades_df['net_pnl'].mean(),
        
        'leverage_used': max_leverage,
        'position_size_pct': position_size_pct * 100,
        'fee_tier': fee_tier
    }
    
    return {
        'metrics': metrics,
        'trades': trades_df.to_dict('records'),
        'equity_curve': equity_curve
    }

def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("REALISTIC 4H BACKTEST - $1000 INITIAL BALANCE")
    print("="*60 + "\n")
    
    # Paths
    run_id = "20251230_005546"
    timeframe = "4h"
    artifacts_dir = PROJECT_ROOT / 'artifacts' / run_id / timeframe
    
    predictions_file = artifacts_dir / 'preds_test.parquet'
    
    # Load from raw data
    from utils.io import load_yaml
    config_train = load_yaml('config/train.yaml')
    config_paths = load_yaml('config/paths.yaml')
    
    from data.loader_new import load_or_resample
    date_range = config_train.get('date_range', {'start': '2023-01-01', 'end': '2024-12-31'})
    
    print("Loading full dataset...")
    df = load_or_resample(
        coin='BTC/USDT',
        target_timeframe=timeframe,
        date_range=date_range,
        data_dir=config_paths.get('data_dir', 'data/raw')
    )
    
    # Get test split (last 15%)
    test_start_idx = int(len(df) * 0.85)
    test_data = df.iloc[test_start_idx:].reset_index(drop=True)
    test_data_file = artifacts_dir / 'test_data_temp.csv'
    test_data.to_csv(test_data_file, index=False)
    print(f"Test data: {len(test_data)} bars")
    
    print(f"Predictions file: {predictions_file}")
    print(f"Test data file: {test_data_file}")
    print(f"Initial balance: $1000")
    print(f"Leverage: 5x")
    print(f"Position size: 20% of balance")
    print(f"Binance fees: 0.1% (regular tier)")
    print(f"Signal threshold: 0.0001\n")
    
    # Run backtest
    results = realistic_backtest_4h(
        predictions_file=predictions_file,
        test_data_file=test_data_file,
        initial_balance=1000.0,
        max_leverage=5.0,
        position_size_pct=0.2,
        signal_threshold=0.0001,
        fee_tier='regular'
    )
    
    if 'error' in results:
        print(f"ERROR: {results['error']}")
        return
    
    # Extract metrics
    metrics = results['metrics']
    
    # Print report
    print("="*60)
    print("BACKTEST RESULTS")
    print("="*60 + "\n")
    
    print(f"BALANCE:")
    print(f"   Initial:        ${metrics['initial_balance']:,.2f}")
    print(f"   Final:          ${metrics['final_balance']:,.2f}")
    print(f"   Net Profit:     ${metrics['net_profit']:,.2f} ({metrics['net_profit_pct']:+.2f}%)")
    print(f"   Total Return:   {metrics['total_return']:.2f}x\n")
    
    print(f"TRADES:")
    print(f"   Total Trades:   {metrics['total_trades']}")
    print(f"   Winning:        {metrics['winning_trades']}")
    print(f"   Losing:         {metrics['losing_trades']}")
    print(f"   Win Rate:       {metrics['win_rate_pct']:.2f}%\n")
    
    print(f"FEES:")
    print(f"   Total Fees:     ${metrics['total_fees_paid']:.2f}")
    print(f"   % of Initial:   {metrics['fees_pct_of_initial']:.2f}%\n")
    
    print(f"PERFORMANCE:")
    print(f"   Gross Profit:   ${metrics['gross_profit']:.2f}")
    print(f"   Gross Loss:     ${metrics['gross_loss']:.2f}")
    print(f"   Profit Factor:  {metrics['profit_factor']:.2f}\n")
    
    print(f"TRADE STATS:")
    print(f"   Avg Win:        ${metrics['avg_win']:.2f}")
    print(f"   Avg Loss:       ${metrics['avg_loss']:.2f}")
    print(f"   Largest Win:    ${metrics['largest_win']:.2f}")
    print(f"   Largest Loss:   ${metrics['largest_loss']:.2f}")
    print(f"   Expectancy:     ${metrics['expectancy']:.2f}\n")
    
    print(f"RISK:")
    print(f"   Max Drawdown:   ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)")
    print(f"   Sharpe Ratio:   {metrics['sharpe_ratio']:.2f}\n")
    
    print(f"DURATION:")
    print(f"   Avg Trade:      {metrics['avg_trade_duration']:.1f} bars (4h each)")
    print(f"   Longest Trade:  {metrics['longest_trade']} bars")
    print(f"   Shortest Trade: {metrics['shortest_trade']} bars\n")
    
    # Save results
    output_file = artifacts_dir / 'realistic_backtest_report.json'
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in metrics.items()},
            'trades_summary': {
                'total_trades': len(results['trades']),
                'first_10_trades': results['trades'][:10]
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"[OK] Report saved to: {output_file}")
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'bar': range(len(results['equity_curve'])),
        'equity': results['equity_curve']
    })
    equity_file = artifacts_dir / 'realistic_equity_curve.csv'
    equity_df.to_csv(equity_file, index=False)
    print(f"[OK] Equity curve saved to: {equity_file}")
    
    # Save trades
    if len(results['trades']) > 0:
        trades_df = pd.DataFrame(results['trades'])
        trades_file = artifacts_dir / 'realistic_trades.csv'
        trades_df.to_csv(trades_file, index=False)
        print(f"[OK] Trades saved to: {trades_file}\n")

if __name__ == '__main__':
    main()

