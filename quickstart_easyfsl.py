# -*- coding: utf-8 -*-
"""
Quick start guide for using Easy Few-Shot Learning library with your project.
Run this to test the integration.
"""

import torch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.easyfsl_integration import EasyFSLWrapper
from utils.config import Config
from PIL import Image
import numpy as np


def test_easyfsl_basic():
    """Test basic Easy FSL functionality."""
    print("="*70)
    print("EASY FEW-SHOT LEARNING - QUICK START TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    
    # Initialize model
    print("\n✓ Initializing Prototypical Networks from easyfsl...")
    model = EasyFSLWrapper(
        method='prototypical',
        backbone='resnet12',
        num_ways=5,
        num_shots=5,
        device=device
    )
    
    print(f"  - Model: {model.model}")
    print(f"  - Backbone: ResNet12")
    print(f"  - Method: Prototypical Networks")
    
    # Create dummy data
    print("\n✓ Creating dummy data for testing...")
    support_images = torch.randn(25, 3, 84, 84)  # 5-way 5-shot
    support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 
                                    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 
                                    4, 4, 4, 4, 4])
    
    query_images = torch.randn(10, 3, 84, 84)  # 2 queries per class
    query_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    
    # Test prediction
    print("\n✓ Testing prediction...")
    predictions, probabilities = model.predict(
        support_images, support_labels, query_images
    )
    
    print(f"  - Predictions shape: {predictions.shape}")
    print(f"  - Predictions: {predictions}")
    print(f"  - Probabilities shape: {probabilities.shape}")
    print(f"  - Sample probabilities:\n{probabilities[:3]}")
    
    # Test model save
    print("\n✓ Testing model save/load...")
    test_path = 'checkpoints/test_easyfsl.pt'
    os.makedirs('checkpoints', exist_ok=True)
    model.save_model(test_path)
    print(f"  - Model saved to: {test_path}")
    
    # Test model load
    model.load_model(test_path)
    print(f"  - Model loaded successfully!")
    
    print("\n" + "="*70)
    print("✓ INTEGRATION TEST PASSED!")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Add your dataset to the 'dataset' folder")
    print("2. Run: python train_easyfsl.py")
    print("3. Or use the EasyFSLWrapper in your own scripts")
    print("\nAVAILABLE METHODS:")
    print("  - 'prototypical' (recommended)")
    print("  - 'matching'")
    print("  - 'relation'")
    print("="*70 + "\n")


def list_available_methods():
    """List all available Easy FSL methods."""
    print("="*70)
    print("EASY FEW-SHOT LEARNING - AVAILABLE METHODS")
    print("="*70)
    
    methods = {
        'prototypical': {
            'description': 'Prototypical Networks - Fast, simple, effective',
            'pros': ['Fast training', 'Interpretable', 'Good baseline'],
            'cons': ['Simple feature space']
        },
        'matching': {
            'description': 'Matching Networks - Learn similarity metric',
            'pros': ['Learnable attention', 'Can handle complex tasks'],
            'cons': ['Slower than ProtoNet']
        },
        'relation': {
            'description': 'Relation Networks - Learn relation module',
            'pros': ['Flexible metric learning', 'State-of-the-art potential'],
            'cons': ['Most complex', 'Needs careful tuning']
        }
    }
    
    for method, info in methods.items():
        print(f"\n📊 {method.upper()}")
        print(f"   {info['description']}")
        print(f"   Pros: {', '.join(info['pros'])}")
        print(f"   Cons: {', '.join(info['cons'])}")
    
    print("\n" + "="*70 + "\n")


def show_usage_examples():
    """Show usage examples."""
    print("="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    
    example1 = """
# Example 1: Basic usage
from utils.easyfsl_integration import EasyFSLWrapper

model = EasyFSLWrapper(method='prototypical', backbone='resnet12')

# Training
loss = model.train_step(support_images, support_labels, 
                        query_images, query_labels)

# Inference
predictions, probabilities = model.predict(
    support_images, support_labels, query_images
)
    """
    
    example2 = """
# Example 2: With optimizer
import torch.optim as optim

model = EasyFSLWrapper(method='prototypical')
optimizer = optim.Adam(model.backbone.parameters(), lr=1e-4)

for epoch in range(100):
    loss = model.train_step(support_imgs, support_labels,
                           query_imgs, query_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save best model
    if epoch % 10 == 0:
        model.save_model('checkpoints/best.pt')
    """
    
    print(example1)
    print(example2)
    print("="*70 + "\n")


if __name__ == '__main__':
    test_easyfsl_basic()
    list_available_methods()
    show_usage_examples()
