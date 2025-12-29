"""
Evaluation Metrics - Classification and Regression
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

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE safely (handles zeros)."""
    mask = y_true != 0
    if mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Returns:
        Dict with MAE, RMSE, MAPE, R2
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'mape': float('nan'),
            'r2': float('nan'),
        }
    
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = safe_mape(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
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
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        y_proba: Predicted probabilities (for classification)
    
    Returns:
        Dict with metrics
    """
    if task_type == 'regression':
        return compute_regression_metrics(y_true, y_pred)
    elif task_type == 'classification':
        return compute_classification_metrics(y_true, y_pred, y_proba)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")





