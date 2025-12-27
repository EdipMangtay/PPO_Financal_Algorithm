"""
Robust Training Wrapper - Handles OOM, NaN, gradient clipping, AMP
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class SafeTrainer:
    """Training wrapper with OOM/NaN guards and AMP support."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        mixed_precision: str = "bf16",
        grad_clip: float = 1.0,
        initial_lr: float = 1e-3
    ):
        """
        Initialize safe trainer.
        
        Args:
            model: PyTorch model
            device: Device string ('cuda' or 'cpu')
            mixed_precision: 'bf16', 'fp16', or 'fp32'
            grad_clip: Gradient clipping value
            initial_lr: Initial learning rate
        """
        self.model = model
        self.device = device
        self.grad_clip = grad_clip
        self.initial_lr = initial_lr
        
        # Mixed precision setup
        self.use_amp = False
        self.scaler = None
        
        if device == "cuda" and torch.cuda.is_available():
            if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
                self.use_amp = True
                self.amp_dtype = torch.bfloat16
                logger.info("Using bfloat16 mixed precision")
            elif mixed_precision == "fp16":
                self.use_amp = True
                self.amp_dtype = torch.float16
                self.scaler = GradScaler()
                logger.info("Using float16 mixed precision with GradScaler")
            else:
                logger.info(f"Mixed precision {mixed_precision} not available, using fp32")
        else:
            logger.info("Using fp32 (CPU or CUDA not available)")
        
        # Training state
        self.nan_incidents = 0
        self.oom_incidents = 0
        self.batch_size_reductions = 0
        
    def train_step(
        self,
        batch: Tuple,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        current_batch_size: int
    ) -> Tuple[float, bool]:
        """
        Execute one training step with error handling.
        
        Returns:
            (loss_value, success)
        """
        try:
            # CRITICAL FIX: Move entire batch to device recursively
            from utils.device import move_to_device
            device = next(self.model.parameters()).device
            batch = move_to_device(batch, device)
            
            # Unpack batch
            x, y = batch
            
            # Forward pass with AMP
            optimizer.zero_grad()
            
            # CRITICAL FIX: Use canonical loss computation if model has loss, otherwise use provided loss_fn
            from utils.model_contracts import compute_tft_loss
            
            if self.use_amp:
                with autocast(dtype=self.amp_dtype):
                    output = self.model(x)
                    # Use canonical TFT loss if model has loss attribute, otherwise use provided loss_fn
                    if hasattr(self.model, 'loss') and callable(self.model.loss):
                        loss = compute_tft_loss(self.model, output, y)
                    else:
                        # Fallback: extract prediction and use provided loss_fn
                        from utils.model_contracts import extract_prediction_tensor, extract_target_tensor
                        pred = extract_prediction_tensor(output)
                        target = extract_target_tensor(y)
                        loss = loss_fn(pred, target)
            else:
                output = self.model(x)
                # Use canonical TFT loss if model has loss attribute, otherwise use provided loss_fn
                if hasattr(self.model, 'loss') and callable(self.model.loss):
                    loss = compute_tft_loss(self.model, output, y)
                else:
                    # Fallback: extract prediction and use provided loss_fn
                    from utils.model_contracts import extract_prediction_tensor, extract_target_tensor
                    pred = extract_prediction_tensor(output)
                    target = extract_target_tensor(y)
                    loss = loss_fn(pred, target)
            
            # Check for NaN/Inf
            if not torch.isfinite(loss):
                self.nan_incidents += 1
                logger.warning(f"NaN/Inf loss detected (incident #{self.nan_incidents})")
                return float('nan'), False
            
            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
            
            return loss.item(), True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.oom_incidents += 1
                logger.warning(f"OOM detected (incident #{self.oom_incidents})")
                torch.cuda.empty_cache()
                return float('nan'), False
            else:
                raise
    
    def reduce_batch_size(self, current_batch_size: int, factor: float = 0.5) -> int:
        """Reduce batch size for OOM recovery."""
        new_batch_size = max(1, int(current_batch_size * factor))
        self.batch_size_reductions += 1
        logger.info(f"Reducing batch size: {current_batch_size} -> {new_batch_size}")
        return new_batch_size

def train_with_early_stopping(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: str = "cuda",
    mixed_precision: str = "bf16",
    grad_clip: float = 1.0,
    early_stopping: Optional[Dict] = None,
    checkpoint_dir: Optional[Path] = None
) -> Dict:
    """
    Train model with early stopping and checkpointing.
    
    Returns:
        Training history dict
    """
    trainer = SafeTrainer(model, device, mixed_precision, grad_clip)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0,
        'nan_incidents': 0,
        'oom_incidents': 0,
    }
    
    patience = early_stopping.get('patience', 5) if early_stopping else None
    min_delta = early_stopping.get('min_delta', 1e-4) if early_stopping else 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            loss_val, success = trainer.train_step(batch, optimizer, loss_fn, train_loader.batch_size)
            
            if success:
                train_losses.append(loss_val)
            else:
                # OOM or NaN - skip batch
                if not np.isfinite(loss_val):
                    history['nan_incidents'] += 1
                else:
                    history['oom_incidents'] += 1
        
        avg_train_loss = np.mean(train_losses) if train_losses else float('nan')
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        if val_loader:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                # CRITICAL FIX: Use recursive device transfer
                from utils.device import move_to_device
                model_device = next(model.parameters()).device
                
                for batch in val_loader:
                    # Move entire batch to device recursively
                    batch = move_to_device(batch, model_device)
                    x, y = batch
                    
                    # Handle y (tuple/list or tensor)
                    if isinstance(y, (tuple, list)):
                        y_true = y[0] if len(y) > 0 else y
                    else:
                        y_true = y
                    
                    output = model(x)
                    
                    # CRITICAL FIX: Use canonical loss computation if model has loss, otherwise use provided loss_fn
                    from utils.model_contracts import compute_tft_loss, extract_prediction_tensor, extract_target_tensor
                    
                    if hasattr(model, 'loss') and callable(model.loss):
                        loss = compute_tft_loss(model, output, y)
                    else:
                        # Fallback: Extract prediction and use provided loss_fn
                        pred = extract_prediction_tensor(output)
                        target = extract_target_tensor(y)
                        loss = loss_fn(pred, target)
                    
                    if torch.isfinite(loss):
                        val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
            history['val_loss'].append(avg_val_loss)
            
            # Early stopping check
            if early_stopping and patience:
                if avg_val_loss < history['best_val_loss'] - min_delta:
                    history['best_val_loss'] = avg_val_loss
                    history['best_epoch'] = epoch
                    patience_counter = 0
                    
                    # Save best model
                    if checkpoint_dir:
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': avg_val_loss,
                        }, checkpoint_dir / 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        else:
            history['val_loss'].append(float('nan'))
        
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.6f}, "
            f"val_loss={history['val_loss'][-1]:.6f}"
        )
    
    # Load best model
    if checkpoint_dir and (checkpoint_dir / 'best_model.pt').exists():
        checkpoint = torch.load(checkpoint_dir / 'best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    return history

