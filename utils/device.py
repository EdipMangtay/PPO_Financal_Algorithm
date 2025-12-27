"""
Device Utilities - GPU/CPU detection and selection
"""

import logging
from typing import Optional, Union, Dict, List, Tuple, Any
import numpy as np

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

def get_device(requested: Optional[str] = None) -> str:
    """
    Get available device (cuda or cpu).
    
    Args:
        requested: Requested device ('cuda' or 'cpu')
    
    Returns:
        Device string
    """
    if not TORCH_AVAILABLE or torch is None:
        return 'cpu'
    
    if requested == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            logger.warning("CUDA requested but not available, using CPU")
            return 'cpu'
    elif requested == 'cpu':
        return 'cpu'
    else:
        # Auto-detect
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

def get_device_info(device: str) -> dict:
    """Get device information."""
    info = {
        'device': device,
        'cuda_available': False,
    }
    
    if TORCH_AVAILABLE and torch is not None:
        info['cuda_available'] = torch.cuda.is_available()
        
        if device == 'cuda' and torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['device_count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda
            info['memory_allocated'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
            info['memory_reserved'] = torch.cuda.memory_reserved(0) / 1024**3  # GB
    
    return info


def move_to_device(
    batch: Any,
    device: Union[str, torch.device],
    non_blocking: bool = True
) -> Any:
    """
    Recursively move tensors in batch to device.
    Supports: torch.Tensor, dict, list, tuple, numpy arrays.
    Preserves non-tensor objects (strings, ints, floats).
    Ensures integer index tensors are moved too.
    
    Args:
        batch: Input batch (can be tensor, dict, list, tuple, or nested)
        device: Target device (string like 'cuda' or torch.device)
        non_blocking: Use non-blocking transfer (faster for GPU)
    
    Returns:
        Batch with all tensors moved to device
    """
    if not TORCH_AVAILABLE or torch is None:
        return batch
    
    # Convert string device to torch.device
    if isinstance(device, str):
        device = torch.device(device)
    
    # Handle torch.Tensor
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    
    # Handle dict
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    
    # Handle list
    if isinstance(batch, list):
        return [move_to_device(item, device, non_blocking) for item in batch]
    
    # Handle tuple
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in batch)
    
    # Handle numpy array (convert to tensor then move)
    if isinstance(batch, np.ndarray):
        tensor = torch.from_numpy(batch)
        return tensor.to(device, non_blocking=non_blocking)
    
    # Handle other types (strings, ints, floats, None) - return as-is
    return batch


def find_device_mismatches(
    batch: Any,
    expected_device: Union[str, torch.device],
    path: str = "root"
) -> List[str]:
    """
    Recursively find tensors in batch that are on wrong device.
    
    Args:
        batch: Input batch (can be tensor, dict, list, tuple, or nested)
        expected_device: Expected device (string like 'cuda' or torch.device)
        path: Current path in batch structure (for error messages)
    
    Returns:
        List of paths (like "x['encoder_cat']") that are on wrong device
    """
    mismatches = []
    
    if not TORCH_AVAILABLE or torch is None:
        return mismatches
    
    # Convert string device to torch.device for comparison
    if isinstance(expected_device, str):
        expected_device = torch.device(expected_device)
    
    # Handle torch.Tensor
    if isinstance(batch, torch.Tensor):
        if batch.device != expected_device:
            mismatches.append(f"{path} (device: {batch.device}, expected: {expected_device})")
        return mismatches
    
    # Handle dict
    if isinstance(batch, dict):
        for k, v in batch.items():
            new_path = f"{path}['{k}']" if isinstance(path, str) and path != "root" else f"['{k}']"
            mismatches.extend(find_device_mismatches(v, expected_device, new_path))
        return mismatches
    
    # Handle list
    if isinstance(batch, list):
        for i, item in enumerate(batch):
            new_path = f"{path}[{i}]" if path != "root" else f"[{i}]"
            mismatches.extend(find_device_mismatches(item, expected_device, new_path))
        return mismatches
    
    # Handle tuple
    if isinstance(batch, tuple):
        for i, item in enumerate(batch):
            new_path = f"{path}[{i}]" if path != "root" else f"[{i}]"
            mismatches.extend(find_device_mismatches(item, expected_device, new_path))
        return mismatches
    
    # Handle numpy array (should be converted to tensor first)
    if isinstance(batch, np.ndarray):
        # This is a warning - numpy arrays should be converted
        mismatches.append(f"{path} (numpy array, should be converted to tensor)")
        return mismatches
    
    # Other types (strings, ints, floats, None) - no mismatch
    return mismatches


def model_device_sanity_check(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[bool, List[str]]:
    """
    Sanity check that model forward pass works with batch on correct device.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader to get a batch from
        device: Expected device (if None, uses model's device)
    
    Returns:
        (success, error_messages)
    """
    errors = []
    
    if not TORCH_AVAILABLE or torch is None:
        errors.append("PyTorch not available")
        return False, errors
    
    try:
        # Get model device
        if device is None:
            device = next(model.parameters()).device
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Move batch to device recursively
        batch = move_to_device(batch, device)
        
        # Check for mismatches
        mismatches = find_device_mismatches(batch, device)
        if mismatches:
            errors.append(f"Device mismatches found: {', '.join(mismatches)}")
            return False, errors
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            # Unpack batch if needed
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1] if len(batch) > 1 else None
            else:
                x, y = batch, None
            
            output = model(x)
            
            # Check output is on correct device
            if isinstance(output, torch.Tensor):
                if output.device != device:
                    errors.append(f"Output tensor on wrong device: {output.device}, expected: {device}")
            elif hasattr(output, 'prediction'):
                if isinstance(output.prediction, torch.Tensor):
                    if output.prediction.device != device:
                        errors.append(f"Output.prediction on wrong device: {output.prediction.device}, expected: {device}")
        
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Forward pass failed: {e}")
        import traceback
        errors.append(f"Traceback: {traceback.format_exc()}")
        return False, errors

