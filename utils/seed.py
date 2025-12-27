"""
Deterministic Seeding - Ensures reproducibility
"""

import random
import numpy as np
import os
import logging

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Seed set to {seed} for reproducibility")

def get_seed_info() -> dict:
    """Get current seed information."""
    info = {
        'python_hashseed': os.environ.get('PYTHONHASHSEED'),
    }
    
    if TORCH_AVAILABLE and torch is not None:
        info['torch_deterministic'] = torch.backends.cudnn.deterministic
        info['torch_benchmark'] = torch.backends.cudnn.benchmark
    else:
        info['torch_deterministic'] = None
        info['torch_benchmark'] = None
    
    return info

