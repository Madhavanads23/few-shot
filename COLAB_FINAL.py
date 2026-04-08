"""
Colab Training Script - CORRECTED
==================================
Uses your actual classes from the repo
"""

# ============================================================================
# CELL 1: Clone Your GitHub Repo
# ============================================================================
import os
import subprocess

repo_url = "https://github.com/Madhavanads23/few-shot.git"
repo_name = "few-shot"

if not os.path.exists(repo_name):
    print(f"Cloning {repo_url}...")
    subprocess.run(["git", "clone", repo_url], check=True)
    print("✅ Repository cloned!")

# DON'T change directory yet - find dataset first!
print(f"Current directory: {os.getcwd()}")

# ============================================================================
# CELL 1a: UPLOAD & EXTRACT DATASET.ZIP
# ============================================================================
print("\n📥 UPLOADING DATASET...")
import os
from google.colab import files
import zipfile
import shutil

print("Click 'Choose Files' and select your dataset.zip\n")
uploaded = files.upload()

if uploaded:
    # Get the uploaded filename
    zip_file = list(uploaded.keys())[0]
    print(f"✅ Uploaded: {zip_file}")
    
    # Extract to /content
    print("\nExtracting dataset (this takes ~2-3 minutes)...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('/content')
    
    print("✅ Extraction complete")
    
    # Debug: Show what was extracted
    print("\n📂 Checking extracted contents:")
    for item in os.listdir('/content'):
        path = f'/content/{item}'
        if os.path.isdir(path):
            print(f"   📁 {item}/")
            try:
                contents = os.listdir(path)[:10]  # Show first 10 items
                for subitem in contents:
                    print(f"      └─ {subitem}")
            except:
                pass
    
    dataset_path = None
    
    # The ZIP structure is: dataset.zip/dataset/train and dataset.zip/dataset/test
    # So after extraction we get: /content/dataset.zip/dataset/train
    # We need to move this to: /content/dataset/train
    
    zip_extracted_path = '/content/dataset.zip/dataset'
    target_dataset_path = '/content/dataset'
    
    if os.path.exists(zip_extracted_path):
        print(f"\n✅ Found ZIP extraction at: {zip_extracted_path}")
        
        train_src = os.path.join(zip_extracted_path, 'train')
        test_src = os.path.join(zip_extracted_path, 'test')
        train_dst = os.path.join(target_dataset_path, 'train')
        test_dst = os.path.join(target_dataset_path, 'test')
        
        # Create target dataset directory if it doesn't exist
        os.makedirs(target_dataset_path, exist_ok=True)
        
        # Remove old ones if they exist
        if os.path.exists(train_dst):
            shutil.rmtree(train_dst)
        if os.path.exists(test_dst):
            shutil.rmtree(test_dst)
        
        # Move from ZIP extraction to target
        if os.path.exists(train_src):
            print(f"   Moving train/ to {train_dst}")
            shutil.move(train_src, train_dst)
        if os.path.exists(test_src):
            print(f"   Moving test/ to {test_dst}")
            shutil.move(test_src, test_dst)
        
        # Clean up the ZIP extraction folder
        if os.path.exists('/content/dataset.zip'):
            shutil.rmtree('/content/dataset.zip')
        
        dataset_path = target_dataset_path
        print(f"\n✅ Structure reorganized!")
        print(f"   New location: {dataset_path}")
    
    # Try common extraction patterns as fallback
    if not dataset_path:
        patterns = [
            '/content/dataset',                    # Direct extraction
            '/content/dataset/dataset',            # Nested structure
            '/content/data',                       # Alternative name
        ]
        
        for pattern in patterns:
            if os.path.exists(pattern):
                train_path = os.path.join(pattern, 'train')
                test_path = os.path.join(pattern, 'test')
                if os.path.exists(train_path) and os.path.exists(test_path):
                    dataset_path = pattern
                    print(f"✅ Found valid dataset at: {pattern}")
                    break
    
    if dataset_path and os.path.exists(os.path.join(dataset_path, 'train')) and os.path.exists(os.path.join(dataset_path, 'test')):
        print(f"✅ Dataset structure verified!")
        print(f"   Location: {dataset_path}")
        print(f"   Ready for training!")
    else:
        print("\n❌ Dataset structure not found!")
        print("   Dataset must have: train/ and test/ folders")
        print("   Please check /content/ contents manually")
else:
    print("⚠️  No file uploaded. You can skip this and use existing dataset.")

# ============================================================================
# CELL 1b: FIND DATASET FIRST (Before cd into repo)
# ============================================================================
print("\n🔍 SEARCHING FOR DATASET...")
import subprocess

dataset_path = None

# Search for dataset folder
result = subprocess.run(
    ['find', '/content', '-maxdepth', 3, '-name', 'dataset', '-type', 'd'],
    capture_output=True, text=True, timeout=10
)

if result.stdout:
    for path in result.stdout.strip().split('\n'):
        if path and os.path.exists(os.path.join(path, 'train')) and os.path.exists(os.path.join(path, 'test')):
            dataset_path = path
            print(f"✅ Found dataset at: {dataset_path}")
            break

if dataset_path is None:
    print("❌ Dataset not found!")
    print("Checked locations like /content/dataset, /content/*/dataset, etc.")
else:
    print(f"✅ Using dataset from: {dataset_path}")

# Now cd into repo
os.chdir(repo_name)
print(f"\nWorking directory: {os.getcwd()}")

# ============================================================================
# CELL 2: Verify Dataset Found
# ============================================================================
print("\n📂 VERIFYING DATASET")
print("=" * 60)

if dataset_path:
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    train_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    test_classes = len([d for d in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, d))])
    
    print(f"✅ Dataset found!")
    print(f"   Location: {dataset_path}")
    print(f"   Train classes: {train_classes}")
    print(f"   Test classes: {test_classes}")
else:
    print("❌ Dataset not found! Training cannot proceed.")
    print("   Please upload the dataset to /content/")

# ============================================================================
# CELL 3: Install Dependencies
# ============================================================================
!pip install -q torch torchvision tqdm scikit-learn matplotlib numpy

print("✅ Dependencies installed!")

# ============================================================================
# CELL 4: Check GPU
# ============================================================================
import torch

print(f"\n💻 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   PyTorch: {torch.__version__}\n")

# ============================================================================
# CELL 5: Import Your Modules (CORRECTED)
# ============================================================================
import sys
sys.path.insert(0, '/content/few-shot')

try:
    from data.data_loader import EpisodicDataLoader
    from training.trainer import FewShotTrainer
    from models.prototypical_network import PrototypicalNetwork
    print("✅ All modules imported successfully!")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# CELL 6: Configure Training
# ============================================================================
CONFIG = {
    'n_way': 5,
    'k_shot': 5,
    'n_query': 15,
    'epochs': 100,
    'learning_rate': 0.0001,
    'weight_decay': 1e-5,
    'scheduler_type': 'cosine',
    'use_lr_scheduler': True,
    'best_model_path': 'best_model.pt'
}

print("\n⚙️  TRAINING CONFIGURATION")
print("=" * 60)
for key, value in CONFIG.items():
    if key != 'best_model_path':
        print(f"  {key}: {value}")

# ============================================================================
# CELL 7: Load Data (CORRECTED)
# ============================================================================
print("\n📥 LOADING DATASET")
print("=" * 60)

# Use the dataset path found in previous cell
if dataset_path is None:
    print("❌ Dataset not found! Please make sure it's uploaded.")
    raise FileNotFoundError("dataset folder not found")

train_loader = EpisodicDataLoader(
    root_dir=dataset_path,
    split="train",
    n_way=CONFIG['n_way'],
    k_shot=CONFIG['k_shot'],
    n_query=CONFIG['n_query']
)

val_loader = EpisodicDataLoader(
    root_dir=dataset_path,
    split="test",
    n_way=CONFIG['n_way'],
    k_shot=CONFIG['k_shot'],
    n_query=CONFIG['n_query']
)

print("✅ Dataset loaded!")
print(f"   Location: {dataset_path}")

# ============================================================================
# CELL 8: Initialize Model
# ============================================================================
print("\n🤖 INITIALIZING MODEL")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PrototypicalNetwork(
    backbone_type='resnet34',
    feature_dim=1024,
    num_transformer_layers=6,
    similarity='hybrid'
)

model = model.to(device)

print("✅ Model initialized!")
print(f"   Architecture: Prototypical Network")
print(f"   Backbone: resnet34")
print(f"   Feature Dimension: 1024")
print(f"   Device: {device}")

# ============================================================================
# CELL 9: START TRAINING 🚀 (CORRECTED)
# ============================================================================
print("\n" + "=" * 60)
print("🚀 STARTING TRAINING")
print("=" * 60)
print(f"\nThis will take ~12-15 hours on GPU\n")

trainer = FewShotTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=CONFIG
)

# Train for specified epochs
trainer.train(num_epochs=CONFIG['epochs'])

# ============================================================================
# CELL 10: Training Complete
# ============================================================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"Best Model Accuracy: {trainer.best_accuracy*100:.2f}%")
print(f"Saved to: {trainer.best_model_path}")

print("\n📥 TO DOWNLOAD YOUR MODEL:")
print("   1. Go to Files (left sidebar)")
print("   2. Right-click 'best_model.pt' → Download")

print("\n🎉 ALL DONE!")
