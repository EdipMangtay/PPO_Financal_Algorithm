"""
Environment Verification Script - RTX 5070 (sm_120) CUDA Setup
Verifies PyTorch CUDA installation and GPU compatibility.
"""

import sys
import subprocess

def check_torch_installation():
    """Check PyTorch installation and CUDA support."""
    print("=" * 60)
    print("PyTorch Environment Verification")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ torch.__version__: {torch.__version__}")
        print(f"✅ torch.version.cuda: {torch.version.cuda}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✅ torch.cuda.is_available(): {cuda_available}")
        
        if cuda_available:
            # Get device info
            device_count = torch.cuda.device_count()
            print(f"✅ torch.cuda.device_count(): {device_count}")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                capability = torch.cuda.get_device_capability(i)
                print(f"✅ Device {i}: {device_name}")
                print(f"   Compute Capability: {capability[0]}.{capability[1]}")
                
                # Check if sm_120 is supported
                arch_list = torch.cuda.get_arch_list()
                print(f"✅ torch.cuda.get_arch_list(): {arch_list}")
                
                # RTX 5070 requires sm_120
                if capability == (12, 0):
                    print(f"⚠️  RTX 5070 detected (sm_120)")
                    if 'sm_120' in arch_list or '12.0' in str(arch_list):
                        print("✅ sm_120 is supported in this PyTorch build")
                    else:
                        print("❌ sm_120 is NOT supported in this PyTorch build")
                        print("   You need PyTorch with CUDA 12.8 (cu128) support")
                        print("   Current CUDA version:", torch.version.cuda)
                        return False
                
                # Test GPU tensor creation
                try:
                    test_tensor = torch.randn(1).cuda()
                    print(f"✅ torch.randn(1).cuda() successful: {test_tensor.device}")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"❌ GPU tensor creation failed: {e}")
                    return False
        else:
            print("⚠️  CUDA not available - will use CPU")
            return True  # CPU is fine
        
        return True
        
    except ImportError:
        print("❌ torch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking torch: {e}")
        return False

def print_installation_instructions():
    """Print instructions for installing PyTorch with CUDA 12.8."""
    print("\n" + "=" * 60)
    print("Installation Instructions for RTX 5070 (sm_120)")
    print("=" * 60)
    print("\n1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Install PyTorch with CUDA 12.8:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    print("\n3. Verify installation:")
    print("   python scripts/verify_env.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    success = check_torch_installation()
    
    if not success:
        print_installation_instructions()
        sys.exit(1)
    else:
        print("\n✅ Environment verification passed!")
        sys.exit(0)
