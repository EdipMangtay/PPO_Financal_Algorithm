"""
Test device transfer utilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np

def test_move_to_device_tensor():
    """Test moving a simple tensor to device."""
    from utils.device import move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    tensor = torch.randn(10, 20)
    moved = move_to_device(tensor, 'cuda')
    
    assert moved.device.type == 'cuda'
    assert torch.equal(tensor.cuda(), moved)


def test_move_to_device_dict():
    """Test moving nested dict with tensors."""
    from utils.device import move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch = {
        'encoder_cont': torch.randn(2, 60, 10),
        'decoder_cont': torch.randn(2, 12, 5),
        'encoder_cat': torch.randint(0, 5, (2, 60)),
        'decoder_cat': torch.randint(0, 5, (2, 12)),
        'static_cat': torch.randint(0, 2, (2,)),
    }
    
    moved = move_to_device(batch, 'cuda')
    
    assert moved['encoder_cont'].device.type == 'cuda'
    assert moved['decoder_cont'].device.type == 'cuda'
    assert moved['encoder_cat'].device.type == 'cuda'
    assert moved['decoder_cat'].device.type == 'cuda'
    assert moved['static_cat'].device.type == 'cuda'


def test_move_to_device_tuple():
    """Test moving tuple (x, y) batch."""
    from utils.device import move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    x = {
        'encoder_cont': torch.randn(2, 60, 10),
        'decoder_cont': torch.randn(2, 12, 5),
    }
    y = (torch.randn(2, 12), torch.ones(2, 12))  # (target, weight)
    
    batch = (x, y)
    moved = move_to_device(batch, 'cuda')
    
    moved_x, moved_y = moved
    
    assert moved_x['encoder_cont'].device.type == 'cuda'
    assert moved_x['decoder_cont'].device.type == 'cuda'
    assert moved_y[0].device.type == 'cuda'
    assert moved_y[1].device.type == 'cuda'


def test_move_to_device_nested():
    """Test moving deeply nested structures."""
    from utils.device import move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch = {
        'nested': {
            'level1': {
                'level2': torch.randn(2, 10)
            }
        },
        'list_of_tensors': [torch.randn(2, 5), torch.randn(2, 3)],
        'tuple_of_tensors': (torch.randn(2, 4), torch.randn(2, 2))
    }
    
    moved = move_to_device(batch, 'cuda')
    
    assert moved['nested']['level1']['level2'].device.type == 'cuda'
    assert moved['list_of_tensors'][0].device.type == 'cuda'
    assert moved['list_of_tensors'][1].device.type == 'cuda'
    assert moved['tuple_of_tensors'][0].device.type == 'cuda'
    assert moved['tuple_of_tensors'][1].device.type == 'cuda'


def test_move_to_device_preserves_non_tensors():
    """Test that non-tensor objects are preserved."""
    from utils.device import move_to_device
    
    batch = {
        'tensor': torch.randn(2, 10),
        'string': 'test',
        'int': 42,
        'float': 3.14,
        'none': None,
        'list_mixed': [torch.randn(2, 5), 'string', 10]
    }
    
    moved = move_to_device(batch, 'cpu')  # Use CPU for this test
    
    assert moved['tensor'].device.type == 'cpu'
    assert moved['string'] == 'test'
    assert moved['int'] == 42
    assert moved['float'] == 3.14
    assert moved['none'] is None
    assert moved['list_mixed'][0].device.type == 'cpu'
    assert moved['list_mixed'][1] == 'string'
    assert moved['list_mixed'][2] == 10


def test_find_device_mismatches():
    """Test finding device mismatches in batch."""
    from utils.device import find_device_mismatches, move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create batch with mixed devices
    batch = {
        'on_cuda': torch.randn(2, 10).cuda(),
        'on_cpu': torch.randn(2, 10),  # Still on CPU
    }
    
    mismatches = find_device_mismatches(batch, torch.device('cuda'))
    assert len(mismatches) > 0, "Should find device mismatches"
    
    # Move everything to CUDA
    batch_moved = move_to_device(batch, 'cuda')
    mismatches_after = find_device_mismatches(batch_moved, torch.device('cuda'))
    assert len(mismatches_after) == 0, "Should find no mismatches after moving"


def test_find_device_mismatches_nested():
    """Test finding mismatches in nested structures."""
    from utils.device import find_device_mismatches, move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    batch = {
        'level1': {
            'level2': {
                'tensor_cpu': torch.randn(2, 10),  # On CPU
                'tensor_cuda': torch.randn(2, 10).cuda()
            }
        }
    }
    
    mismatches = find_device_mismatches(batch, torch.device('cuda'))
    assert len(mismatches) > 0, "Should find mismatches in nested structure"
    
    # Check that path is reported correctly
    cpu_paths = [m for m in mismatches if 'cpu' in m.lower() or 'tensor_cpu' in m]
    assert len(cpu_paths) > 0, "Should report path to CPU tensor"


def test_move_to_device_long_index_tensors():
    """Test moving long/integer index tensors (common in TFT)."""
    from utils.device import move_to_device
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # TFT often uses long tensors for categorical indices
    batch = {
        'encoder_cat': torch.randint(0, 10, (2, 60), dtype=torch.long),
        'decoder_cat': torch.randint(0, 10, (2, 12), dtype=torch.long),
        'static_cat': torch.randint(0, 2, (2,), dtype=torch.long),
        'time_idx': torch.arange(0, 60, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    }
    
    moved = move_to_device(batch, 'cuda')
    
    assert moved['encoder_cat'].device.type == 'cuda'
    assert moved['decoder_cat'].device.type == 'cuda'
    assert moved['static_cat'].device.type == 'cuda'
    assert moved['time_idx'].device.type == 'cuda'
    
    # Verify dtypes are preserved
    assert moved['encoder_cat'].dtype == torch.long
    assert moved['decoder_cat'].dtype == torch.long
    assert moved['static_cat'].dtype == torch.long
    assert moved['time_idx'].dtype == torch.long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




