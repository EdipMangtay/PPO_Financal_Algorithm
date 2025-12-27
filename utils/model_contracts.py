"""
TFT Model Contract Utilities - Ensures output/target/loss alignment
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def infer_task_mode(config: Dict, output_size: int, loss_class_name: str) -> str:
    """
    Infer task mode from config and model setup.
    
    Returns:
        "regression", "quantile", or "classification"
    """
    # Check explicit config
    task_config = config.get('task', {})
    if isinstance(task_config, dict):
        mode = task_config.get('mode', None)
        if mode in ['regression', 'quantile', 'classification']:
            return mode
    
    # Infer from model setup
    if output_size == 1:
        return "regression"
    elif output_size > 1 and "QuantileLoss" in loss_class_name:
        return "quantile"
    elif output_size > 1:
        # Could be multi-target regression or classification
        # Default to regression unless explicitly classification
        if "CrossEntropy" in loss_class_name or "BCE" in loss_class_name:
            return "classification"
        return "regression"
    else:
        return "regression"  # Default


def validate_tft_contract(
    output_size: int,
    loss: nn.Module,
    mode: Optional[str] = None,
    config: Optional[Dict] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate TFT output/loss contract.
    
    Returns:
        (is_valid, error_message)
    """
    loss_class_name = loss.__class__.__name__
    
    if mode is None:
        mode = infer_task_mode(config or {}, output_size, loss_class_name)
    
    # Regression mode: output_size should be 1, use regression loss
    if mode == "regression":
        if output_size != 1:
            return False, (
                f"CONTRACT VIOLATION: Regression mode requires output_size=1, "
                f"but model has output_size={output_size}. "
                f"Either set output_size=1 OR switch to quantile mode with QuantileLoss."
            )
        if "QuantileLoss" in loss_class_name:
            return False, (
                f"CONTRACT VIOLATION: Regression mode should use regression loss (MAE/RMSE/MSE), "
                f"but QuantileLoss is used. Use MAE() or RMSE() from pytorch_forecasting.metrics."
            )
        return True, None
    
    # Quantile mode: output_size should match quantiles, use QuantileLoss
    elif mode == "quantile":
        if "QuantileLoss" not in loss_class_name:
            return False, (
                f"CONTRACT VIOLATION: Quantile mode requires QuantileLoss, "
                f"but {loss_class_name} is used. Use QuantileLoss(quantiles=[...])."
            )
        # QuantileLoss expects output_size to match number of quantiles
        # Default pytorch-forecasting uses 7 quantiles
        if output_size not in [7, 9]:
            logger.warning(
                f"Quantile mode with output_size={output_size} (expected 7 or 9). "
                f"Ensure quantiles match output_size."
            )
        return True, None
    
    # Classification mode
    elif mode == "classification":
        if output_size == 1:
            return False, (
                f"CONTRACT VIOLATION: Classification mode requires output_size > 1 "
                f"(number of classes), but output_size=1."
            )
        return True, None
    
    return True, None


def normalize_prediction_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "regression",
    quantile_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize prediction and target tensors to compatible shapes.
    
    Args:
        pred: Prediction tensor [B, T, ...] or [B, T]
        target: Target tensor [B, T] or [B, T, 1]
        mode: Task mode ("regression", "quantile", "classification")
        quantile_index: For quantile mode, which quantile to extract (default: median=3 for 7 quantiles)
    
    Returns:
        (pred_aligned, target_aligned) with compatible shapes
    """
    # Handle quantile predictions: extract median if regression loss is used
    if mode == "quantile" and pred.dim() == 3 and pred.shape[-1] > 1:
        # Extract median quantile (index 3 for 7 quantiles, index 4 for 9 quantiles)
        if quantile_index is None:
            # Default to median: for 7 quantiles, index 3 is 0.5 quantile
            quantile_index = pred.shape[-1] // 2
        pred = pred[..., quantile_index]  # [B, T, 7] -> [B, T]
        logger.debug(f"Extracted quantile {quantile_index} from prediction: {pred.shape}")
    
    # Ensure both are 2D [B, T]
    if pred.dim() == 3:
        if pred.shape[-1] == 1:
            pred = pred.squeeze(-1)  # [B, T, 1] -> [B, T]
        else:
            raise ValueError(
                f"Cannot normalize prediction shape {pred.shape} to [B, T]. "
                f"Expected [B, T] or [B, T, 1] or [B, T, quantiles]."
            )
    
    if target.dim() == 3:
        if target.shape[-1] == 1:
            target = target.squeeze(-1)  # [B, T, 1] -> [B, T]
        else:
            raise ValueError(
                f"Cannot normalize target shape {target.shape} to [B, T]. "
                f"Expected [B, T] or [B, T, 1]."
            )
    
    # Ensure same shape
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch after normalization: pred={pred.shape}, target={target.shape}. "
            f"Both should be [B, T]."
        )
    
    return pred, target


def get_loss_for_mode(
    mode: str,
    quantiles: Optional[list] = None
) -> nn.Module:
    """
    Get appropriate loss function for task mode.
    
    Args:
        mode: "regression", "quantile", or "classification"
        quantiles: For quantile mode, list of quantiles (e.g., [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    
    Returns:
        Loss function
    """
    if mode == "regression":
        from pytorch_forecasting.metrics import MAE
        return MAE()
    elif mode == "quantile":
        from pytorch_forecasting.metrics import QuantileLoss
        if quantiles is None:
            # Default 7 quantiles
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        return QuantileLoss(quantiles=quantiles)
    elif mode == "classification":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def extract_prediction_tensor(output) -> torch.Tensor:
    """
    Extract prediction tensor from model output.
    
    Handles:
    - Output objects with .prediction attribute (pytorch-forecasting)
    - Dicts with "prediction" key
    - Raw tensors
    
    Args:
        output: Model output (Output object, dict, or tensor)
    
    Returns:
        Prediction tensor
    """
    if hasattr(output, 'prediction'):
        return output.prediction
    if isinstance(output, dict):
        if "prediction" in output:
            return output["prediction"]
        raise ValueError(f"Output dict missing 'prediction' key. Available: {list(output.keys())}")
    if isinstance(output, torch.Tensor):
        return output
    raise ValueError(f"Cannot extract prediction from type: {type(output)}")


def extract_target_tensor(y) -> torch.Tensor:
    """
    Extract target tensor from y (handles tuple/list like (target, weight)).
    
    Args:
        y: Target (tuple/list or tensor)
    
    Returns:
        Target tensor
    """
    if isinstance(y, (list, tuple)):
        y_true = y[0]
        if not isinstance(y_true, torch.Tensor):
            raise ValueError(f"y[0] is not a Tensor. Type: {type(y_true)}")
        return y_true
    if isinstance(y, torch.Tensor):
        return y
    raise ValueError(f"y is not tuple/list or Tensor. Type: {type(y)}")


def normalize_y_for_tft(y) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (target, weight) as float tensors with shape [B, T].
    - y can be Tensor OR tuple/list (target, weight)
    - If no weight -> weight = ones_like(target)
    - If weight is [B] -> expand to [B, T]
    - If weight is [B,1] -> expand to [B, T]
    - If weight shape mismatched -> safe fallback to ones_like(target)
    
    Args:
        y: Target from dataloader (tuple/list or tensor)
    
    Returns:
        (target, weight) both as [B, T] tensors
    """
    # Extract target with robust error handling
    if isinstance(y, (list, tuple)):
        if len(y) == 0:
            raise ValueError(f"y is empty tuple/list. Expected (target, weight) or (target,)")
        target = y[0]
        if len(y) > 1:
            weight = y[1]
        else:
            weight = None
    elif isinstance(y, torch.Tensor):
        target = y
        weight = None
    else:
        raise ValueError(
            f"y must be tuple/list or Tensor, got: {type(y)}. "
            f"If y is a tuple, it should be (target, weight) where target is a Tensor."
        )
    
    if not isinstance(target, torch.Tensor):
        raise ValueError(
            f"target must be Tensor, got: {type(target)}. "
            f"y structure: {[type(item) for item in y] if isinstance(y, (list, tuple)) else 'N/A'}"
        )
    
    # Ensure target is [B, T] with safe shape handling
    if target.ndim == 1:
        # [T] -> [1, T] (single sample)
        target = target.unsqueeze(0)
    elif target.ndim == 3:
        # [B, T, 1] -> [B, T]
        if target.shape[-1] == 1:
            target = target.squeeze(-1)
        else:
            raise ValueError(
                f"target shape {target.shape} not supported. Expected [B, T] or [B, T, 1]. "
                f"Got {target.ndim}D tensor."
            )
    elif target.ndim == 0:
        # Scalar -> [1, 1]
        target = target.unsqueeze(0).unsqueeze(0)
    
    if target.ndim != 2:
        raise ValueError(
            f"target must be [B, T] after normalization, got shape: {target.shape} "
            f"(ndim={target.ndim}). Original y type: {type(y)}"
        )
    
    B, T = target.shape
    
    # Handle weight with safe fallback
    if weight is None:
        weight = torch.ones_like(target)  # [B, T]
    elif isinstance(weight, torch.Tensor):
        try:
            if weight.ndim == 0:
                # Scalar weight -> broadcast to [B, T]
                weight = torch.full_like(target, weight.item())
            elif weight.ndim == 1:
                # [B] -> [B, T] (broadcast)
                if weight.shape[0] != B:
                    logger.warning(
                        f"weight shape {weight.shape} incompatible with target {target.shape}. "
                        f"Using ones_like(target) as fallback."
                    )
                    weight = torch.ones_like(target)
                else:
                    weight = weight.unsqueeze(1).expand(B, T)  # [B, 1] -> [B, T]
            elif weight.ndim == 2:
                if weight.shape[1] == 1:
                    # [B, 1] -> [B, T] (broadcast)
                    if weight.shape[0] != B:
                        logger.warning(
                            f"weight shape {weight.shape} incompatible with target {target.shape}. "
                            f"Using ones_like(target) as fallback."
                        )
                        weight = torch.ones_like(target)
                    else:
                        weight = weight.expand(B, T)
                elif weight.shape != (B, T):
                    logger.warning(
                        f"weight shape {weight.shape} incompatible with target {target.shape}. "
                        f"Using ones_like(target) as fallback."
                    )
                    weight = torch.ones_like(target)
                # else: weight.shape == (B, T) - perfect, use as is
            else:
                # weight.ndim > 2 - not supported, fallback
                logger.warning(
                    f"weight shape {weight.shape} (ndim={weight.ndim}) not supported. "
                    f"Expected [B], [B, 1], or [B, T]. Using ones_like(target) as fallback."
                )
                weight = torch.ones_like(target)
        except Exception as e:
            # Safe fallback on any error
            logger.warning(
                f"Error processing weight tensor {weight.shape}: {e}. "
                f"Using ones_like(target) as fallback."
            )
            weight = torch.ones_like(target)
    else:
        # weight is not None but not a Tensor - fallback
        logger.warning(
            f"weight is not a Tensor (type: {type(weight)}). Using ones_like(target) as fallback."
        )
        weight = torch.ones_like(target)
    
    # Final validation
    if weight.shape != (B, T):
        logger.warning(
            f"Final weight shape {weight.shape} != target shape {target.shape}. "
            f"Reshaping weight to match target."
        )
        weight = torch.ones_like(target)
    
    return target, weight


def compute_tft_loss(
    model: nn.Module,
    output,
    y,
    return_raw: bool = False
) -> torch.Tensor:
    """
    Canonical TFT loss computation.
    
    Extracts prediction from output, normalizes y to (target, weight),
    calls model.loss (which is an nn.Module) with tensors, and applies weighting.
    
    Args:
        model: TFT model (has model.loss as nn.Module)
        output: Model output (Output object, dict, or tensor)
        y: Target from dataloader (tuple/list or tensor)
        return_raw: If True, return raw loss before weighting
    
    Returns:
        Loss tensor (scalar on same device as model)
    """
    # Extract prediction tensor with error handling
    try:
        pred = extract_prediction_tensor(output)  # [B, T, Q] for quantile, [B, T] for regression
    except Exception as e:
        raise ValueError(
            f"Failed to extract prediction tensor from output type {type(output)}: {e}. "
            f"Output attributes: {dir(output) if hasattr(output, '__dict__') else 'N/A'}"
        )
    
    # Normalize y to (target, weight) with error handling
    try:
        target, weight = normalize_y_for_tft(y)  # Both [B, T]
    except Exception as e:
        raise ValueError(
            f"Failed to normalize y (type: {type(y)}) to (target, weight): {e}. "
            f"y structure: {[type(item) for item in y] if isinstance(y, (list, tuple)) else 'N/A'}"
        )
    
    # Validate shapes match
    if pred.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch dimension mismatch: pred.shape[0]={pred.shape[0]} vs target.shape[0]={target.shape[0]}"
        )
    if pred.shape[1] != target.shape[1]:
        raise ValueError(
            f"Time dimension mismatch: pred.shape[1]={pred.shape[1]} vs target.shape[1]={target.shape[1]}"
        )
    
    # Get loss function (model.loss is an nn.Module)
    loss_fn = model.loss if hasattr(model, 'loss') else None
    if loss_fn is None:
        raise ValueError(
            f"Model does not have loss attribute. Model type: {type(model)}, "
            f"attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}"
        )
    
    # Ensure loss_fn is on same device as pred/target
    if hasattr(loss_fn, 'to'):
        device = pred.device
        loss_fn = loss_fn.to(device)
    
    # Call loss function with tensors
    # QuantileLoss expects (y_pred, y_true) where y_pred is [B, T, Q] and y_true is [B, T]
    # Regression loss expects (y_pred, y_true) where both are [B, T]
    try:
        loss_raw = loss_fn(pred, target)
    except Exception as e:
        raise RuntimeError(
            f"Loss function {loss_fn.__class__.__name__} failed with pred shape {pred.shape} "
            f"and target shape {target.shape}: {e}"
        )
    
    if return_raw:
        return loss_raw
    
    # Validate loss_raw is a tensor
    if not isinstance(loss_raw, torch.Tensor):
        raise ValueError(
            f"Loss function returned non-tensor type: {type(loss_raw)}. "
            f"Expected torch.Tensor."
        )
    
    # Apply weighting if loss is per-sample
    # QuantileLoss returns [B, T] per-sample losses
    if loss_raw.ndim == 2:
        # [B, T] loss, apply weight and mean
        if weight.shape != loss_raw.shape:
            # Weight shape mismatch - use mean without weighting
            logger.warning(
                f"Weight shape {weight.shape} != loss shape {loss_raw.shape}. "
                f"Computing unweighted mean."
            )
            loss = loss_raw.mean()
        else:
            loss = (loss_raw * weight).mean()
    elif loss_raw.ndim == 1:
        # [B] loss, apply weight (broadcast) and mean
        if weight.ndim == 2:
            weight_1d = weight.mean(dim=1)  # [B, T] -> [B]
            if weight_1d.shape != loss_raw.shape:
                logger.warning(
                    f"Weight 1D shape {weight_1d.shape} != loss shape {loss_raw.shape}. "
                    f"Computing unweighted mean."
                )
                loss = loss_raw.mean()
            else:
                loss = (loss_raw * weight_1d).mean()
        elif weight.ndim == 1:
            if weight.shape != loss_raw.shape:
                logger.warning(
                    f"Weight shape {weight.shape} != loss shape {loss_raw.shape}. "
                    f"Computing unweighted mean."
                )
                loss = loss_raw.mean()
            else:
                loss = (loss_raw * weight).mean()
        else:
            loss = loss_raw.mean()
    elif loss_raw.ndim == 0:
        # Scalar loss
        loss = loss_raw
    else:
        # Unexpected shape - take mean
        logger.warning(
            f"Loss has unexpected shape {loss_raw.shape} (ndim={loss_raw.ndim}). "
            f"Taking mean."
        )
        loss = loss_raw.mean() if loss_raw.numel() > 0 else loss_raw
    
    # Ensure loss is scalar and on correct device
    if loss.ndim > 0:
        loss = loss.mean() if loss.numel() > 0 else loss
    
    # Ensure loss is on same device as model
    if hasattr(model, 'parameters'):
        model_device = next(model.parameters()).device
        if loss.device != model_device:
            loss = loss.to(model_device)
    
    return loss


def validate_pred_target_shapes(
    mode: str,
    pred: torch.Tensor,
    target: torch.Tensor
) -> Tuple[bool, str, Dict]:
    """
    Validate prediction and target shapes are compatible for the given mode.
    
    Args:
        mode: Task mode ("regression", "quantile", "classification")
        pred: Prediction tensor
        target: Target tensor
    
    Returns:
        (is_valid, error_message, diagnostics_dict)
        diagnostics_dict contains: shapes, mode, ndims, etc.
    """
    diagnostics = {
        'mode': mode,
        'pred_shape': list(pred.shape),
        'target_shape': list(target.shape),
        'pred_ndim': pred.ndim,
        'target_ndim': target.ndim,
    }
    
    # Quantile mode: pred [B, T, Q] vs target [B, T] is VALID
    if mode == "quantile":
        if pred.ndim == 3 and target.ndim == 2:
            # Check batch and time dimensions match
            if pred.shape[0] != target.shape[0]:
                return False, (
                    f"Batch dimension mismatch: pred.shape[0]={pred.shape[0]} "
                    f"vs target.shape[0]={target.shape[0]}"
                ), diagnostics
            if pred.shape[1] != target.shape[1]:
                return False, (
                    f"Time dimension mismatch: pred.shape[1]={pred.shape[1]} "
                    f"vs target.shape[1]={target.shape[1]}"
                ), diagnostics
            # Q dimension can be any positive value
            diagnostics['quantiles'] = pred.shape[2]
            return True, "", diagnostics
        else:
            return False, (
                f"Quantile mode requires pred.ndim=3 and target.ndim=2, "
                f"got pred.ndim={pred.ndim}, target.ndim={target.ndim}"
            ), diagnostics
    
    # Regression mode: pred [B, T] or [B, T, 1] vs target [B, T]
    elif mode == "regression":
        # Handle pred [B, T, 1] -> squeeze to [B, T]
        pred_normalized = pred
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred_normalized = pred.squeeze(-1)
            diagnostics['pred_normalized'] = True
        
        # Both should be [B, T] now
        if pred_normalized.ndim != 2 or target.ndim != 2:
            return False, (
                f"Regression mode requires pred.ndim=2 and target.ndim=2, "
                f"got pred.ndim={pred_normalized.ndim}, target.ndim={target.ndim}"
            ), diagnostics
        
        if pred_normalized.shape != target.shape:
            return False, (
                f"Shape mismatch: pred={pred_normalized.shape} vs target={target.shape}"
            ), diagnostics
        
        diagnostics['pred_shape_normalized'] = list(pred_normalized.shape)
        return True, "", diagnostics
    
    # Classification mode: pred [B, T, C] vs target [B, T] (or [B, T, 1])
    elif mode == "classification":
        # Target can be [B, T] or [B, T, 1]
        target_normalized = target
        if target.ndim == 3 and target.shape[-1] == 1:
            target_normalized = target.squeeze(-1)
        
        if pred.ndim != 3 or target_normalized.ndim != 2:
            return False, (
                f"Classification mode requires pred.ndim=3 and target.ndim=2, "
                f"got pred.ndim={pred.ndim}, target.ndim={target_normalized.ndim}"
            ), diagnostics
        
        if pred.shape[0] != target_normalized.shape[0] or pred.shape[1] != target_normalized.shape[1]:
            return False, (
                f"Batch/time mismatch: pred={pred.shape[:2]} vs target={target_normalized.shape}"
            ), diagnostics
        
        diagnostics['num_classes'] = pred.shape[2]
        return True, "", diagnostics
    
    else:
        return False, f"Unknown mode: {mode}", diagnostics


def log_shape_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    output_size: int,
    loss: nn.Module,
    mode: str,
    context: str = ""
):
    """
    Log shape diagnostics for debugging.
    
    Uses validate_pred_target_shapes to check compatibility.
    
    Args:
        pred: Prediction tensor
        target: Target tensor
        output_size: Model output_size
        loss: Loss function
        mode: Task mode
        context: Context string (e.g., "preflight", "training")
    """
    logger.info(f"Shape diagnostics ({context}):")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Output size: {output_size}")
    logger.info(f"  Loss: {loss.__class__.__name__}")
    logger.info(f"  Prediction shape: {pred.shape}")
    logger.info(f"  Target shape: {target.shape}")
    
    is_valid, error_msg, diagnostics = validate_pred_target_shapes(mode, pred, target)
    
    if is_valid:
        logger.info(f"  ✅ Shapes are compatible for {mode} mode")
        if mode == "quantile" and pred.ndim == 3:
            logger.info(f"    Quantile output: {pred.shape[2]} quantiles")
    else:
        logger.error(f"  ❌ SHAPE MISMATCH: {error_msg}")
        logger.error(f"  Expected for {mode} mode:")
        if mode == "quantile":
            logger.error(f"    pred: [B, T, Q] (Q=number of quantiles)")
            logger.error(f"    target: [B, T]")
        elif mode == "regression":
            logger.error(f"    pred: [B, T] or [B, T, 1]")
            logger.error(f"    target: [B, T]")

