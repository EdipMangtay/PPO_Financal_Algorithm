"""
Environment Doctor - Validates system before training
Checks Python version, packages, CUDA, disk space, permissions
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import json

def check_python_version() -> Tuple[bool, str]:
    """Check Python version >= 3.8"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (requires >= 3.8)"

def check_packages() -> Dict[str, Tuple[bool, str]]:
    """Check required packages"""
    required = {
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'pyyaml': 'yaml',
        'pytorch-forecasting': 'pytorch_forecasting',
        'scikit-learn': 'sklearn',
    }
    
    results = {}
    for package_name, import_name in required.items():
        try:
            __import__(import_name)
            results[package_name] = (True, "Installed")
        except ImportError:
            results[package_name] = (False, "Missing")
    
    return results

def check_cuda() -> Dict[str, any]:
    """Check CUDA availability and GPU info"""
    result = {
        'available': False,
        'device_name': None,
        'device_count': 0,
        'cuda_version': None,
        'free_vram_mb': None,
    }
    
    try:
        import torch
        result['available'] = torch.cuda.is_available()
        
        if result['available']:
            result['device_name'] = torch.cuda.get_device_name(0)
            result['device_count'] = torch.cuda.device_count()
            result['cuda_version'] = torch.version.cuda
            
            # Estimate free VRAM
            if torch.cuda.is_available():
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
                result['free_vram_mb'] = free_mem / (1024 ** 2)
    except Exception as e:
        result['error'] = str(e)
    
    return result

def check_disk_space(path: str = ".") -> Tuple[bool, str]:
    """Check available disk space"""
    try:
        import shutil
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        
        if free_gb < 1.0:
            return False, f"{free_gb:.2f} GB free (need >= 1 GB)"
        return True, f"{free_gb:.2f} GB free"
    except Exception as e:
        return False, f"Error: {e}"

def check_write_permissions(path: str = ".") -> Tuple[bool, str]:
    """Check write permissions"""
    try:
        test_file = Path(path) / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True, "Write permissions OK"
    except Exception as e:
        return False, f"No write permission: {e}"

def check_windows_multiprocessing() -> Tuple[bool, str]:
    """Check Windows-specific multiprocessing issues"""
    if platform.system() != "Windows":
        return True, "Not Windows"
    
    # Windows multiprocessing requires if __name__ == "__main__"
    return True, "Windows detected - num_workers=0 recommended"

def run_doctor(verbose: bool = True) -> Dict:
    """Run all checks and return results"""
    results = {
        'python_version': check_python_version(),
        'packages': check_packages(),
        'cuda': check_cuda(),
        'disk_space': check_disk_space(),
        'write_permissions': check_write_permissions(),
        'windows_multiprocessing': check_windows_multiprocessing(),
        'all_checks_passed': True,
    }
    
    # Determine if all critical checks passed
    if not results['python_version'][0]:
        results['all_checks_passed'] = False
    
    missing_packages = [pkg for pkg, (ok, _) in results['packages'].items() if not ok]
    if missing_packages:
        results['all_checks_passed'] = False
    
    if not results['disk_space'][0]:
        results['all_checks_passed'] = False
    
    if not results['write_permissions'][0]:
        results['all_checks_passed'] = False
    
    if verbose:
        print("=" * 60)
        print("ENVIRONMENT DOCTOR REPORT")
        print("=" * 60)
        
        print(f"\nPython Version: {results['python_version'][1]}")
        
        print("\nPackages:")
        for pkg, (ok, msg) in results['packages'].items():
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {status} {pkg}: {msg}")
        
        print("\nCUDA:")
        cuda = results['cuda']
        if cuda['available']:
            print(f"  [OK] CUDA Available")
            print(f"    Device: {cuda['device_name']}")
            print(f"    Count: {cuda['device_count']}")
            print(f"    CUDA Version: {cuda['cuda_version']}")
            if cuda['free_vram_mb']:
                print(f"    Free VRAM: {cuda['free_vram_mb']:.0f} MB")
        else:
            print(f"  [WARN] CUDA Not Available (will use CPU)")
        
        print(f"\nDisk Space: {results['disk_space'][1]}")
        print(f"Write Permissions: {results['write_permissions'][1]}")
        print(f"OS: {results['windows_multiprocessing'][1]}")
        
        print("\n" + "=" * 60)
        if results['all_checks_passed']:
            print("[SUCCESS] ALL CHECKS PASSED")
        else:
            print("[FAILED] SOME CHECKS FAILED - Review above")
        print("=" * 60)
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()
    
    results = run_doctor(verbose=not args.quiet)
    
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    
    sys.exit(0 if results['all_checks_passed'] else 1)

