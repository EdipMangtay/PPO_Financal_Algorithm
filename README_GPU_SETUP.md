# GPU Setup - RTX 5070 (sm_120) with CUDA 12.8

## Problem

RTX 5070 uses compute capability **sm_120**, which requires PyTorch built with **CUDA 12.8 (cu128)** support. The default PyTorch installation (cu118) does not support sm_120.

## Solution: Install PyTorch with CUDA 12.8

### Step 1: Uninstall Current PyTorch

```bash
pip uninstall torch torchvision torchaudio
```

### Step 2: Install PyTorch with CUDA 12.8

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Verify Installation

```bash
python scripts/verify_env.py
```

Expected output:
```
✅ torch.__version__: 2.x.x+cu128
✅ torch.version.cuda: 12.8
✅ torch.cuda.is_available(): True
✅ Device 0: NVIDIA GeForce RTX 5070
   Compute Capability: 12.0
✅ torch.cuda.get_arch_list(): ['sm_120', ...]
✅ sm_120 is supported in this PyTorch build
✅ torch.randn(1).cuda() successful: cuda:0
```

## Troubleshooting

### If verification fails:

1. **Check CUDA Toolkit version:**
   ```bash
   nvcc --version
   ```
   Should show CUDA 12.8 or higher.

2. **Check GPU detection:**
   ```bash
   nvidia-smi
   ```
   Should show RTX 5070.

3. **Reinstall PyTorch:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```

### If CUDA 12.8 is not available:

- Use CPU mode (slower but works):
  ```bash
  # In config/train.yaml, set:
  device: "cpu"
  ```

- Or wait for official CUDA 12.8 PyTorch builds.

## Verification Script

The `scripts/verify_env.py` script checks:
- PyTorch version and CUDA version
- GPU availability and device name
- Compute capability (should be 12.0 for RTX 5070)
- Architecture list (should include sm_120)
- GPU tensor creation test

Run it anytime to verify your setup:
```bash
python scripts/verify_env.py
```
