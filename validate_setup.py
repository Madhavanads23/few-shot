"""
Comprehensive validation script for the few-shot learning project.
Checks all dependencies, file imports, and library versions.
"""

import sys
import importlib
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def check_library(lib_name, package_name=None):
    """Check if a library is installed."""
    if package_name is None:
        package_name = lib_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {lib_name:20} v{version}")
        return True
    except ImportError as e:
        print(f"❌ {lib_name:20} - NOT INSTALLED: {e}")
        return False

def check_file_import(file_path, import_statement):
    """Check if a Python file can be imported."""
    try:
        exec(f"from {import_statement} import *")
        print(f"✅ {file_path}")
        return True
    except Exception as e:
        print(f"❌ {file_path} - ERROR: {str(e)[:60]}")
        return False

def main():
    print("=" * 80)
    print("FEW-SHOT LEARNING PROJECT - VALIDATION REPORT")
    print("=" * 80)
    
    # Check core libraries
    print("\n📦 CORE LIBRARIES")
    print("-" * 80)
    libraries = [
        ('PyTorch', 'torch'),
        ('TorchVision', 'torchvision'),
        ('NumPy', 'numpy'),
        ('PIL', 'PIL'),
        ('Matplotlib', 'matplotlib'),
        ('Scikit-Learn', 'sklearn'),
        ('Pandas', 'pandas'),
        ('Tqdm', 'tqdm'),
    ]
    
    lib_status = []
    for name, package in libraries:
        status = check_library(name, package)
        lib_status.append(status)
    
    # Check Easy-FSL specific
    print("\n🎯 EASY FEW-SHOT LEARNING")
    print("-" * 80)
    
    try:
        from easyfsl.methods import PrototypicalNetworks, MatchingNetworks, RelationNetworks
        print("✅ PrototypicalNetworks")
        print("✅ MatchingNetworks")
        print("✅ RelationNetworks")
    except ImportError as e:
        print(f"❌ Easy-FSL methods - {e}")
    
    try:
        from easyfsl.modules import resnet12, resnet18, resnet34
        print("✅ ResNet12 backbone")
        print("✅ ResNet18 backbone")
        print("✅ ResNet34 backbone")
    except ImportError as e:
        print(f"❌ Easy-FSL backbones - {e}")
    
    # Check project modules
    print("\n📂 PROJECT MODULES")
    print("-" * 80)
    
    modules = [
        ('utils/easyfsl_integration.py', 'utils.easyfsl_integration'),
        ('utils/config.py', 'utils.config'),
        ('data/data_loader.py', 'data.data_loader'),
        ('evaluation/evaluate.py', 'evaluation.evaluate'),
        ('infer.py', 'infer'),
    ]
    
    module_status = []
    for file_path, import_stmt in modules:
        status = check_file_import(file_path, import_stmt)
        module_status.append(status)
    
    # Check file syntax
    print("\n✔️ SYNTAX CHECK")
    print("-" * 80)
    
    files_to_check = [
        'train_easyfsl.py',
        'quickstart_easyfsl.py',
        'app.py',
        'evaluate_model.py',
    ]
    
    syntax_ok = True
    for py_file in files_to_check:
        try:
            import py_compile
            py_compile.compile(py_file, doraise=True)
            print(f"✅ {py_file}")
        except py_compile.PyCompileError as e:
            print(f"❌ {py_file} - {str(e)[:50]}")
            syntax_ok = False
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_libs_ok = all(lib_status)
    all_modules_ok = all(module_status)
    
    print(f"Libraries: {'✅ ALL OK' if all_libs_ok else '⚠️  SOME MISSING'}")
    print(f"Modules:   {'✅ ALL OK' if all_modules_ok else '⚠️  SOME ISSUES'}")
    print(f"Syntax:    {'✅ ALL OK' if syntax_ok else '⚠️  SOME ERRORS'}")
    
    if all_libs_ok and all_modules_ok and syntax_ok:
        print("\n🎉 VALIDATION PASSED - Project is ready to use!")
        print("\nNext steps:")
        print("  1. Organize CIFAR-10 data in dataset/ folder")
        print("  2. Run: python train_easyfsl.py")
        print("  3. Check results in checkpoints/ and results/ folders")
        return 0
    else:
        print("\n⚠️  VALIDATION FAILED - Please fix the issues above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
