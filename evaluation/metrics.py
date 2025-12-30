"""
Evaluation Metrics - PROFIT-FIRST METRICS FOR CRYPTO TRADING
Prioritizes Directional Accuracy and PnL over curve-fitting metrics like R2/MAPE
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import logging

logger = logging.getLogger(__name__)

def extract_median_quantile(y_pred: np.ndarray) -> np.ndarray:
    """
    CRITICAL FIX: Extract 0.5 quantile (median) from TFT quantile predictions.
    
    TFT outputs shape: (N, decoder_len, 7) for 7 quantiles [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    We need the median (index 3 = 0.5 quantile) for point predictions.
    
    Args:
        y_pred: Predictions, shape (N, decoder_len, 7) or (N, decoder_len)
    
    Returns:
        Median predictions, shape (N, decoder_len)
    """
    if y_pred.ndim == 3 and y_pred.shape[-1] == 7:
        # Extract median quantile (index 3)
        return y_pred[:, :, 3]
    elif y_pred.ndim == 2:
        # Already extracted or single output
        return y_pred
    else:
        raise ValueError(f"Unexpected y_pred shape: {y_pred.shape}")

def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    PROFIT-FIRST METRIC: Directional Accuracy (Did we predict the sign correctly?)
    
    For crypto trading, this is MORE IMPORTANT than R2.
    > 51% = Profitable model
    > 55% = Good model
    > 60% = Excellent model
    
    Args:
        y_true: True returns (log_returns)
        y_pred: Predicted returns
    
    Returns:
        Directional accuracy (0-1)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float('nan')
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Compare signs
    true_sign = np.sign(y_true_clean)
    pred_sign = np.sign(y_pred_clean)
    
    # Directional accuracy
    correct = (true_sign == pred_sign).sum()
    total = len(true_sign)
    
    return float(correct / total) if total > 0 else 0.0

def compute_trading_pnl(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    leverage: float = 5.0,
    transaction_cost: float = 0.0004  # 0.04% per trade
) -> Dict[str, float]:
    """
    PROFIT-FIRST METRIC: Simulated Trading PnL with leverage.
    
    Strategy: Long if y_pred > 0, Short if y_pred < 0, Flat if y_pred == 0
    
    Args:
        y_true: True returns (log_returns)
        y_pred: Predicted returns
        leverage: Leverage multiplier
        transaction_cost: Slippage + fees per trade
    
    Returns:
        Dict with cumulative_pnl, sharpe_ratio, max_drawdown, win_rate
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            'cumulative_pnl': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'total_trades': 0
        }
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Generate positions
    positions = np.sign(y_pred_clean)  # +1 (Long), -1 (Short), 0 (Flat)
    
    # Calculate returns
    raw_returns = positions * y_true_clean * leverage
    
    # Apply transaction costs (when position changes)
    position_changes = np.diff(positions, prepend=0) != 0
    costs = position_changes * transaction_cost
    
    net_returns = raw_returns - costs
    
    # Cumulative metrics
    cumulative_pnl = net_returns.sum()
    
    # Sharpe ratio
    if len(net_returns) > 1:
        sharpe_ratio = np.mean(net_returns) / (np.std(net_returns) + 1e-8) * np.sqrt(252)  # Annualized
    else:
        sharpe_ratio = 0.0
    
    # Max drawdown
    cumulative_curve = np.cumsum(net_returns)
    running_max = np.maximum.accumulate(cumulative_curve)
    drawdown = running_max - cumulative_curve
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
    
    # Win rate
    winning_trades = (net_returns > 0).sum()
    total_trades = (positions != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    return {
        'cumulative_pnl': float(cumulative_pnl),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'total_trades': int(total_trades)
    }

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE safely (handles zeros). DEPRECATED for crypto log_returns."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    leverage: float = 5.0
) -> Dict[str, float]:
    """
    PROFIT-FIRST METRICS: Prioritize Directional Accuracy and PnL over R2/MAPE.
    
    CRITICAL FIX: Extract median quantile if y_pred has quantile dimension.
    
    Args:
        y_true: True values (log_returns)
        y_pred: Predicted values (may have quantile dimension)
        leverage: Leverage for PnL calculation
    
    Returns:
        Dict with profit metrics (directional_accuracy, pnl_*) + legacy metrics (mae, rmse, r2)
    """
    # CRITICAL FIX: Extract median quantile if needed
    y_pred = extract_median_quantile(y_pred)
    
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            'directional_accuracy': float('nan'),
            'pnl_cumulative': float('nan'),
            'pnl_sharpe': float('nan'),
            'pnl_max_dd': float('nan'),
            'pnl_win_rate': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
        }
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # PROFIT-FIRST METRICS (Priority)
    directional_accuracy = compute_directional_accuracy(y_true_clean, y_pred_clean)
    pnl_metrics = compute_trading_pnl(y_true_clean, y_pred_clean, leverage=leverage)
    
    # Legacy metrics (for reference, but NOT optimized)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    return {
        # PROFIT METRICS (MOST IMPORTANT)
        'directional_accuracy': float(directional_accuracy),
        'pnl_cumulative': float(pnl_metrics['cumulative_pnl']),
        'pnl_sharpe': float(pnl_metrics['sharpe_ratio']),
        'pnl_max_dd': float(pnl_metrics['max_drawdown']),
        'pnl_win_rate': float(pnl_metrics['win_rate']),
        'pnl_total_trades': int(pnl_metrics['total_trades']),
        
        # LEGACY METRICS (for reference)
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
    }

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC)
    
    Returns:
        Dict with accuracy, precision, recall, F1, AUC, confusion_matrix
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }
    
    # AUC if probabilities provided
    if y_proba is not None:
        try:
            # Binary classification
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                if y_proba.ndim == 2:
                    y_proba_binary = y_proba[:, 1]
                else:
                    y_proba_binary = y_proba
                auc = roc_auc_score(y_true, y_proba_binary)
                metrics['auc'] = float(auc)
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
    except Exception as e:
        logger.warning(f"Could not compute confusion matrix: {e}")
    
    return metrics

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'regression',
    y_proba: Optional[np.ndarray] = None,
    leverage: float = 5.0
) -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        y_proba: Predicted probabilities (for classification)
        leverage: Leverage for PnL calculation (default 5.0)
    
    Returns:
        Dict with metrics
    """
    if task_type == 'regression':
        return compute_regression_metrics(y_true, y_pred, leverage=leverage)
    elif task_type == 'classification':
        return compute_classification_metrics(y_true, y_pred, y_proba)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")





