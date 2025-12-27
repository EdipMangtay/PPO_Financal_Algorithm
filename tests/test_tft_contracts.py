"""
Test TFT model contracts: shape validation and loss computation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path

def test_validate_pred_target_shapes_quantile():
    """Test shape validation for quantile mode."""
    from utils.model_contracts import validate_pred_target_shapes
    
    # Quantile mode: pred [B, T, Q] vs target [B, T] is VALID
    B, T, Q = 2, 12, 7
    pred = torch.randn(B, T, Q)
    target = torch.randn(B, T)
    
    is_valid, error_msg, diagnostics = validate_pred_target_shapes("quantile", pred, target)
    
    assert is_valid, f"Quantile shapes should be valid: {error_msg}"
    assert diagnostics['mode'] == "quantile"
    assert diagnostics['quantiles'] == Q
    assert diagnostics['pred_shape'] == [B, T, Q]
    assert diagnostics['target_shape'] == [B, T]


def test_validate_pred_target_shapes_regression():
    """Test shape validation for regression mode."""
    from utils.model_contracts import validate_pred_target_shapes
    
    # Regression mode: pred [B, T] vs target [B, T] is VALID
    B, T = 2, 12
    pred = torch.randn(B, T)
    target = torch.randn(B, T)
    
    is_valid, error_msg, diagnostics = validate_pred_target_shapes("regression", pred, target)
    
    assert is_valid, f"Regression shapes should be valid: {error_msg}"
    assert diagnostics['mode'] == "regression"
    
    # Also test [B, T, 1] -> squeezed to [B, T]
    pred_3d = torch.randn(B, T, 1)
    is_valid_3d, _, _ = validate_pred_target_shapes("regression", pred_3d, target)
    assert is_valid_3d, "Regression should accept [B, T, 1] pred"


def test_validate_pred_target_shapes_mismatch():
    """Test shape validation catches mismatches."""
    from utils.model_contracts import validate_pred_target_shapes
    
    # Batch mismatch
    pred = torch.randn(2, 12, 7)
    target = torch.randn(3, 12)  # Different batch size
    
    is_valid, error_msg, _ = validate_pred_target_shapes("quantile", pred, target)
    assert not is_valid, "Should detect batch mismatch"
    assert "Batch dimension" in error_msg
    
    # Time mismatch
    pred = torch.randn(2, 12, 7)
    target = torch.randn(2, 10)  # Different time dimension
    
    is_valid, error_msg, _ = validate_pred_target_shapes("quantile", pred, target)
    assert not is_valid, "Should detect time dimension mismatch"
    assert "Time dimension" in error_msg


def test_extract_prediction_tensor():
    """Test extraction of prediction tensor from various output types."""
    from utils.model_contracts import extract_prediction_tensor
    
    B, T, Q = 2, 12, 7
    pred_tensor = torch.randn(B, T, Q)
    
    # Test Output object with .prediction
    class MockOutput:
        def __init__(self):
            self.prediction = pred_tensor
    
    output_obj = MockOutput()
    extracted = extract_prediction_tensor(output_obj)
    assert torch.equal(extracted, pred_tensor)
    
    # Test dict with "prediction" key
    output_dict = {"prediction": pred_tensor}
    extracted = extract_prediction_tensor(output_dict)
    assert torch.equal(extracted, pred_tensor)
    
    # Test raw tensor
    extracted = extract_prediction_tensor(pred_tensor)
    assert torch.equal(extracted, pred_tensor)


def test_extract_target_tensor():
    """Test extraction of target tensor from y."""
    from utils.model_contracts import extract_target_tensor
    
    B, T = 2, 12
    target_tensor = torch.randn(B, T)
    
    # Test tuple (target, weight)
    y_tuple = (target_tensor, torch.ones(B, T))
    extracted = extract_target_tensor(y_tuple)
    assert torch.equal(extracted, target_tensor)
    
    # Test list
    y_list = [target_tensor, torch.ones(B, T)]
    extracted = extract_target_tensor(y_list)
    assert torch.equal(extracted, target_tensor)
    
    # Test raw tensor
    extracted = extract_target_tensor(target_tensor)
    assert torch.equal(extracted, target_tensor)


def test_tft_model_loss_computation():
    """Test that TFT model loss computation works with quantile outputs."""
    from models.tft import TFTModel
    from pytorch_forecasting import TimeSeriesDataSet
    
    # Create tiny dataset
    n_samples = 200
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
    
    np.random.seed(42)
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 0.1)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.rand(n_samples) * 1000,
    })
    
    # Add required features (minimal set)
    df['RSI'] = 50.0
    df['MACD'] = 0.0
    df['MACD_signal'] = 0.0
    df['MACD_hist'] = 0.0
    df['ATR'] = 1.0
    df['BB_upper'] = prices * 1.01
    df['BB_middle'] = prices
    df['BB_lower'] = prices * 0.99
    df['Volume_MA'] = df['volume']
    
    # Create model
    model = TFTModel(
        prediction_horizon=12,
        max_encoder_length=60,
        max_decoder_length=12,
        hidden_size=64,  # Smaller for test
        device='cpu'  # Use CPU for tests
    )
    
    # Create dataset
    coin = "BTC/USDT"
    df['coin'] = coin
    df['time_idx'] = range(len(df))
    
    data_dict = {coin: df}
    dataset = model.create_dataset(data_dict, target='close')
    
    # Build model with quantile mode
    config = {
        'task': {
            'mode': 'quantile',
            'quantiles': [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        }
    }
    model.build_model(dataset, config=config)
    
    # Get one batch
    dataloader = dataset.to_dataloader(train=True, batch_size=2, num_workers=0)
    batch = next(iter(dataloader))
    
    # Move to device
    from utils.device import move_to_device
    device = next(model.model.parameters()).device
    batch = move_to_device(batch, device)
    x, y = batch
    
    # Forward pass
    model.model.eval()
    with torch.no_grad():
        output = model.model(x)
        
        # CRITICAL: Use canonical loss computation
        from utils.model_contracts import compute_tft_loss
        loss = compute_tft_loss(model.model, output, y)
        
        # Validate loss is finite
        assert torch.isfinite(loss), f"Loss should be finite, got: {loss.item()}"
        
        # Validate shapes using our validation function
        from utils.model_contracts import (
            extract_prediction_tensor,
            extract_target_tensor,
            validate_pred_target_shapes
        )
        
        pred = extract_prediction_tensor(output)
        target = extract_target_tensor(y)
        
        is_valid, error_msg, diagnostics = validate_pred_target_shapes("quantile", pred, target)
        assert is_valid, f"Shapes should be valid for quantile mode: {error_msg}\n{diagnostics}"
        
        # Verify quantile output shape
        assert pred.ndim == 3, f"Pred should be 3D [B, T, Q], got: {pred.shape}"
        assert pred.shape[2] == 7, f"Should have 7 quantiles, got: {pred.shape[2]}"
        assert target.ndim == 2, f"Target should be 2D [B, T], got: {target.shape}"


def test_normalize_y_for_tft_tensor():
    """Test normalize_y_for_tft with tensor input."""
    from utils.model_contracts import normalize_y_for_tft
    
    B, T = 2, 12
    target = torch.randn(B, T)
    
    target_norm, weight_norm = normalize_y_for_tft(target)
    
    assert target_norm.shape == (B, T)
    assert weight_norm.shape == (B, T)
    assert torch.allclose(target_norm, target)
    assert torch.allclose(weight_norm, torch.ones_like(target))


def test_normalize_y_for_tft_tuple():
    """Test normalize_y_for_tft with tuple (target, weight)."""
    from utils.model_contracts import normalize_y_for_tft
    
    B, T = 2, 12
    target = torch.randn(B, T)
    weight = torch.ones(B, T) * 0.5
    
    target_norm, weight_norm = normalize_y_for_tft((target, weight))
    
    assert target_norm.shape == (B, T)
    assert weight_norm.shape == (B, T)
    assert torch.allclose(target_norm, target)
    assert torch.allclose(weight_norm, weight)


def test_normalize_y_for_tft_weight_broadcast():
    """Test normalize_y_for_tft with weight broadcasting [B] -> [B, T]."""
    from utils.model_contracts import normalize_y_for_tft
    
    B, T = 2, 12
    target = torch.randn(B, T)
    weight_1d = torch.ones(B) * 0.5
    
    target_norm, weight_norm = normalize_y_for_tft((target, weight_1d))
    
    assert target_norm.shape == (B, T)
    assert weight_norm.shape == (B, T)
    assert torch.allclose(weight_norm, torch.ones(B, T) * 0.5)


def test_normalize_y_for_tft_weight_broadcast_2d():
    """Test normalize_y_for_tft with weight broadcasting [B, 1] -> [B, T]."""
    from utils.model_contracts import normalize_y_for_tft
    
    B, T = 2, 12
    target = torch.randn(B, T)
    weight_2d = torch.ones(B, 1) * 0.5
    
    target_norm, weight_norm = normalize_y_for_tft((target, weight_2d))
    
    assert target_norm.shape == (B, T)
    assert weight_norm.shape == (B, T)
    assert torch.allclose(weight_norm, torch.ones(B, T) * 0.5)


def test_normalize_y_for_tft_3d_target():
    """Test normalize_y_for_tft with [B, T, 1] target -> [B, T]."""
    from utils.model_contracts import normalize_y_for_tft
    
    B, T = 2, 12
    target = torch.randn(B, T, 1)
    
    target_norm, weight_norm = normalize_y_for_tft(target)
    
    assert target_norm.shape == (B, T)
    assert weight_norm.shape == (B, T)
    assert torch.allclose(target_norm, target.squeeze(-1))


def test_normalize_y_for_tft_1d_target():
    """Test normalize_y_for_tft with [T] target -> [1, T]."""
    from utils.model_contracts import normalize_y_for_tft
    
    T = 12
    target = torch.randn(T)
    
    target_norm, weight_norm = normalize_y_for_tft(target)
    
    assert target_norm.shape == (1, T)
    assert weight_norm.shape == (1, T)
    assert torch.allclose(target_norm, target.unsqueeze(0))


def test_compute_tft_loss_quantile():
    """Test compute_tft_loss with quantile mode."""
    from utils.model_contracts import compute_tft_loss
    from pytorch_forecasting.metrics import QuantileLoss
    
    B, T, Q = 2, 12, 7
    
    # Create mock model with QuantileLoss
    class MockModel:
        def __init__(self):
            self.loss = QuantileLoss()
    
    model = MockModel()
    
    # Create predictions [B, T, Q] and target [B, T]
    pred = torch.randn(B, T, Q)
    target = torch.randn(B, T)
    
    # Test with Output object
    class MockOutput:
        def __init__(self):
            self.prediction = pred
    
    output = MockOutput()
    y = target
    
    loss = compute_tft_loss(model, output, y)
    
    assert torch.is_tensor(loss)
    assert loss.ndim == 0  # Scalar
    assert torch.isfinite(loss)


def test_compute_tft_loss_quantile_with_weight():
    """Test compute_tft_loss with quantile mode and weight."""
    from utils.model_contracts import compute_tft_loss
    from pytorch_forecasting.metrics import QuantileLoss
    
    B, T, Q = 2, 12, 7
    
    class MockModel:
        def __init__(self):
            self.loss = QuantileLoss()
    
    model = MockModel()
    
    pred = torch.randn(B, T, Q)
    target = torch.randn(B, T)
    weight = torch.ones(B, T) * 0.5
    
    class MockOutput:
        def __init__(self):
            self.prediction = pred
    
    output = MockOutput()
    y = (target, weight)
    
    loss = compute_tft_loss(model, output, y)
    
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_tft_loss_regression():
    """Test compute_tft_loss with regression mode."""
    from utils.model_contracts import compute_tft_loss
    from pytorch_forecasting.metrics import MAE
    
    B, T = 2, 12
    
    class MockModel:
        def __init__(self):
            self.loss = MAE()
    
    model = MockModel()
    
    pred = torch.randn(B, T)
    target = torch.randn(B, T)
    
    class MockOutput:
        def __init__(self):
            self.prediction = pred
    
    output = MockOutput()
    y = target
    
    loss = compute_tft_loss(model, output, y)
    
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_tft_loss_dict_output():
    """Test compute_tft_loss with dict output."""
    from utils.model_contracts import compute_tft_loss
    from pytorch_forecasting.metrics import QuantileLoss
    
    B, T, Q = 2, 12, 7
    
    class MockModel:
        def __init__(self):
            self.loss = QuantileLoss()
    
    model = MockModel()
    
    pred = torch.randn(B, T, Q)
    target = torch.randn(B, T)
    
    output = {"prediction": pred}
    y = target
    
    loss = compute_tft_loss(model, output, y)
    
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_tft_loss_tensor_output():
    """Test compute_tft_loss with raw tensor output."""
    from utils.model_contracts import compute_tft_loss
    from pytorch_forecasting.metrics import QuantileLoss
    
    B, T, Q = 2, 12, 7
    
    class MockModel:
        def __init__(self):
            self.loss = QuantileLoss()
    
    model = MockModel()
    
    pred = torch.randn(B, T, Q)
    target = torch.randn(B, T)
    
    output = pred  # Raw tensor
    y = target
    
    loss = compute_tft_loss(model, output, y)
    
    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

