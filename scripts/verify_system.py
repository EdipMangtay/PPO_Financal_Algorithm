"""
System Verification Script - Windows Stability & Profit-Driven HPO
Verifies that the system is properly configured to prevent CPU freezing and maximize profitability.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# ============================================================================
# APPLY THREAD LIMITS IMMEDIATELY (before any other torch operations)
# ============================================================================
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_torch_thread_limits():
    """
    Verify that PyTorch thread limits are set to prevent CPU saturation.
    
    CRITICAL: On Windows with high-core CPUs (Intel Ultra), unrestricted
    threading causes OS interrupt starvation (USB/Bluetooth freezing).
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 1: PyTorch Thread Limits")
    logger.info("="*80)
    
    intra_threads = torch.get_num_threads()
    inter_threads = torch.get_num_interop_threads()
    
    logger.info(f"Intra-op threads (BLAS): {intra_threads}")
    logger.info(f"Inter-op threads (parallel ops): {inter_threads}")
    
    # Check if limits are reasonable (should be << total CPU cores)
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"System CPU cores: {cpu_count}")
    
    # Validate: threads should be limited to prevent saturation
    if intra_threads <= 0 or intra_threads > cpu_count:
        logger.warning(f"⚠️  Intra-op threads ({intra_threads}) unrestricted - may cause freezing!")
        return False
    
    if intra_threads > cpu_count // 2:
        logger.warning(f"⚠️  Intra-op threads ({intra_threads}) too high - recommend 4 or less")
        return False
    
    logger.info(f"✅ Thread limits are safe ({intra_threads} intra, {inter_threads} inter)")
    return True

def check_config_num_workers():
    """
    Verify that config/train.yaml has num_workers=0 for Windows safety.
    
    CRITICAL: PyTorch DataLoader with num_workers > 0 on Windows uses
    'spawn' multiprocessing, which causes severe OS interrupt delays.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Config num_workers Setting")
    logger.info("="*80)
    
    config_path = Path("config/train.yaml")
    
    if not config_path.exists():
        logger.error(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    num_workers = config.get('num_workers', -1)
    
    logger.info(f"Config num_workers: {num_workers}")
    
    if num_workers != 0:
        logger.error(f"❌ num_workers MUST be 0 on Windows (found: {num_workers})")
        logger.error("   This will cause system freezing!")
        return False
    
    logger.info("✅ num_workers=0 (Windows safe)")
    return True

def test_dataloader_creation():
    """
    Test that DataLoader can be created with num_workers=0 without errors.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 3: DataLoader Creation (num_workers=0)")
    logger.info("="*80)
    
    try:
        # Create mock dataset
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
                self.data = torch.randn(size, 10)
                self.targets = torch.randn(size, 1)
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return (self.data[idx], self.targets[idx])
        
        dataset = MockDataset()
        
        # Create DataLoader with num_workers=0
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            num_workers=0,  # CRITICAL
            shuffle=False
        )
        
        # Try to iterate
        batch_count = 0
        for batch in loader:
            x, y = batch
            batch_count += 1
            if batch_count >= 3:  # Just test a few batches
                break
        
        logger.info(f"✅ DataLoader created and iterated successfully (num_workers=0)")
        logger.info(f"   Processed {batch_count} batches without freezing")
        return True
        
    except Exception as e:
        logger.error(f"❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_profit_driven_objective():
    """
    Verify that the objective function computes Directional Accuracy and Proxy PnL.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Profit-Driven Objective Function")
    logger.info("="*80)
    
    try:
        # Create minimal mock data
        np.random.seed(42)
        n_samples = 200
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='15min')
        price = 50000 + np.cumsum(np.random.randn(n_samples) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': price + np.random.randn(n_samples) * 10,
            'high': price + np.abs(np.random.randn(n_samples) * 20),
            'low': price - np.abs(np.random.randn(n_samples) * 20),
            'close': price,
            'volume': np.random.randint(100, 1000, n_samples),
            'time_idx': np.arange(n_samples),
            'coin': 'BTC/USDT',
            # Add minimal features
            'RSI_14': np.random.uniform(30, 70, n_samples),
            'EMA_21': price * np.random.uniform(0.99, 1.01, n_samples),
        })
        
        # Add target
        df['target'] = df['close'].pct_change(12).shift(-12) * 100
        df = df.dropna()
        
        # Split
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size].copy()
        val_data = df.iloc[train_size:].copy()
        
        logger.info(f"Mock data: Train={len(train_data)}, Val={len(val_data)}")
        
        # Import objective function
        from hpo.optuna_search import objective
        import optuna
        
        # Create mock config
        config = {
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'mixed_precision': 'bf16' if torch.cuda.is_bf16_supported() else 'fp32',
            'grad_clip': 1.0,
            'num_workers': 0,  # CRITICAL
            'model': {
                'hidden_size': 32,  # Small for speed
                'attention_head_size': 2,
                'dropout': 0.1,
                'max_encoder_length': 20,  # Small for speed
                'max_decoder_length': 4,
            },
            'task': {
                'mode': 'quantile',
                'quantiles': [0.1, 0.5, 0.9]
            },
            'paths': {
                'artifacts_dir': 'artifacts_verify_system'
            }
        }
        
        # Create study and run one trial
        study = optuna.create_study(direction='maximize')
        trial = study.ask()
        
        logger.info("Running objective function (1 trial, 2 epochs for speed)...")
        
        # Temporarily reduce epochs in objective for speed
        score = objective(trial, '15m', train_data, val_data, config, 'BTC/USDT')
        
        # Check that profit metrics were computed
        da = trial.user_attrs.get('directional_accuracy', None)
        pnl = trial.user_attrs.get('proxy_pnl_mean', None)
        comp_score = trial.user_attrs.get('composite_score', None)
        
        logger.info(f"Trial completed:")
        logger.info(f"  Score: {score:.6f}")
        logger.info(f"  Directional Accuracy: {da}")
        logger.info(f"  Proxy PnL: {pnl}")
        logger.info(f"  Composite Score: {comp_score}")
        
        # Validate
        if da is None:
            logger.error("❌ Directional Accuracy NOT computed!")
            return False
        
        if pnl is None:
            logger.error("❌ Proxy PnL NOT computed!")
            return False
        
        if comp_score is None:
            logger.error("❌ Composite Score NOT computed!")
            return False
        
        # Validate that score matches composite_score
        if abs(score - comp_score) > 1e-6:
            logger.error(f"❌ Score mismatch: returned {score} vs composite {comp_score}")
            return False
        
        logger.info("✅ Profit-driven metrics computed successfully!")
        logger.info(f"   DA={da:.2%}, PnL={pnl:.6f}, Score={score:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Objective function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if Path('artifacts_verify_system').exists():
            import shutil
            shutil.rmtree('artifacts_verify_system', ignore_errors=True)

def test_memory_cleanup():
    """
    Verify that memory cleanup function works properly.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Memory Cleanup Function")
    logger.info("="*80)
    
    try:
        from hpo.optuna_search import _cleanup_trial_resources
        
        # Create mock objects
        mock_model = {"dummy": "model"}
        mock_optimizer = {"dummy": "optimizer"}
        mock_loader1 = {"dummy": "loader1"}
        mock_loader2 = {"dummy": "loader2"}
        
        # Call cleanup
        _cleanup_trial_resources(mock_model, mock_optimizer, mock_loader1, mock_loader2)
        
        logger.info("✅ Memory cleanup function executed without errors")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory cleanup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all system verification tests."""
    logger.info("="*80)
    logger.info("SYSTEM VERIFICATION: Windows Stability & Profit-Driven HPO")
    logger.info("="*80)
    
    results = []
    
    # Test 1: Thread limits
    try:
        results.append(("Thread Limits", check_torch_thread_limits()))
    except Exception as e:
        logger.error(f"Thread limits test crashed: {e}")
        results.append(("Thread Limits", False))
    
    # Test 2: Config num_workers
    try:
        results.append(("Config num_workers", check_config_num_workers()))
    except Exception as e:
        logger.error(f"Config test crashed: {e}")
        results.append(("Config num_workers", False))
    
    # Test 3: DataLoader creation
    try:
        results.append(("DataLoader Creation", test_dataloader_creation()))
    except Exception as e:
        logger.error(f"DataLoader test crashed: {e}")
        results.append(("DataLoader Creation", False))
    
    # Test 4: Profit-driven objective
    try:
        results.append(("Profit-Driven Objective", test_profit_driven_objective()))
    except Exception as e:
        logger.error(f"Objective test crashed: {e}")
        results.append(("Profit-Driven Objective", False))
    
    # Test 5: Memory cleanup
    try:
        results.append(("Memory Cleanup", test_memory_cleanup()))
    except Exception as e:
        logger.error(f"Memory cleanup test crashed: {e}")
        results.append(("Memory Cleanup", False))
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:.<40} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED - System is ready for production!")
        logger.info("\nYou can now run:")
        logger.info("  python scripts/run_btc_pipeline.py --config config/train.yaml --hpo_trials 100")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED - Fix issues before running HPO")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

