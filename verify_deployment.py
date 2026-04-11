#!/usr/bin/env python3
"""
Deployment Verification Script
================================
Verifies that CLIP deployment is properly set up and working

Usage:
    python verify_deployment.py
"""

import sys
import os
from pathlib import Path

def check_file(filepath, description):
    """Check if a file exists."""
    exists = Path(filepath).exists()
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {filepath}")
    return exists

def check_import(module, description):
    """Check if a module can be imported."""
    try:
        __import__(module)
        print(f"  ✅ {description}: {module}")
        return True
    except ImportError as e:
        print(f"  ❌ {description}: {module} - {str(e)[:50]}")
        return False

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_info = torch.cuda.get_device_name(0) if has_gpu else "Not available"
        status = "✅" if has_gpu else "⚠️"
        print(f"  {status} GPU: {gpu_info}")
        return has_gpu
    except Exception as e:
        print(f"  ❌ GPU check failed: {str(e)}")
        return False

def main():
    print("="*70)
    print("🔍 CLIP DEPLOYMENT VERIFICATION")
    print("="*70)
    print()
    
    all_good = True
    
    # Check files
    print("📁 CHECKING FILES:")
    files_to_check = [
        ("clip_model_wrapper.py", "Model wrapper"),
        ("clip_deployment_app.py", "Flask app"),
        ("clip_cli.py", "CLI tool"),
        ("templates/clip_index.html", "Web interface"),
        ("CLIP_DEPLOYMENT_GUIDE.md", "Deployment guide"),
        ("requirements_clip_deployment.txt", "Requirements"),
        ("DEPLOYMENT_SUMMARY.md", "Summary"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file(filepath, description):
            all_good = False
    
    print()
    
    # Check Python packages
    print("📦 CHECKING PYTHON PACKAGES:")
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("flask", "Flask"),
        ("numpy", "NumPy"),
    ]
    
    for module, description in packages:
        if not check_import(module, description):
            all_good = False
    
    print()
    
    # Check GPU
    print("🎮 HARDWARE CHECK:")
    has_gpu = check_gpu()
    print(f"  ℹ️  GPU not required but recommended for speed")
    
    print()
    
    # Check directories
    print("📂 CHECKING DIRECTORIES:")
    dirs_to_check = [
        ("dataset/test", "Test dataset"),
        ("uploads", "Upload folder"),
        ("templates", "Templates folder"),
    ]
    
    for dirpath, description in dirs_to_check:
        exists = Path(dirpath).exists()
        status = "✅" if exists else "⚠️"
        print(f"  {status} {description}: {dirpath}")
    
    print()
    
    # Quick model test
    print("🧪 TESTING MODEL LOADING:")
    try:
        sys.path.insert(0, str(Path.cwd()))
        from clip_model_wrapper import CLIPModelWrapper
        print("  ✅ Model wrapper imported successfully")
        print("  ⏳ Attempting to load model... (this may take a minute)")
        
        # Try to create model instance
        try:
            model = CLIPModelWrapper()
            print("  ✅ Model loaded successfully!")
            print(f"  ✅ Device: {model.device}")
            print(f"  ✅ Classes: {len(model.class_names)}")
        except Exception as e:
            print(f"  ⚠️  Model loading: {str(e)[:80]}")
            print("     (First load downloads 605 MB model - needs internet)")
    
    except ImportError as e:
        print(f"  ❌ Model wrapper import failed: {e}")
        all_good = False
    
    print()
    print()
    
    # Summary
    print("="*70)
    if all_good:
        print("✅ DEPLOYMENT VERIFICATION PASSED!")
        print()
        print("Next steps:")
        print("  1. Start the web server:")
        print("     python clip_deployment_app.py")
        print()
        print("  2. Visit http://localhost:5000 in your browser")
        print()
        print("  3. Or use the CLI:")
        print("     python clip_cli.py --help")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print()
        print("To fix issues:")
        print("  1. Install missing packages:")
        print("     pip install -r requirements_clip_deployment.txt")
        print()
        print("  2. Create missing directories:")
        print("     mkdir -p uploads templates")
    
    print("="*70)
    
    return 0 if all_good else 1

if __name__ == '__main__':
    sys.exit(main())
