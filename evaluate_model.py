import torch
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models.prototypical_network import PrototypicalNetwork
from data.data_loader import EpisodicDataLoader
from evaluation.evaluate import FewShotEvaluator
from utils.config import Config

def main():
    """Evaluate the trained model."""
    
    print("=" * 70)
    print("MODEL ACCURACY EVALUATION")
    print("=" * 70)
    
    # Configuration
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n✓ Configuration:")
    print(f"  - Model: {config.backbone_type}")
    print(f"  - Few-shot: {config.n_way}-way {config.k_shot}-shot")
    print(f"  - Device: {device}")
    
    # Check if model exists
    if not os.path.exists(config.best_model_path):
        print(f"\n❌ ERROR: Model not found at {config.best_model_path}")
        print("   Please train the model first using: python train.py")
        return
    
    print(f"\n✓ Model found at: {config.best_model_path}")
    
    # Load model
    print(f"\n⏳ Loading model...")
    model = PrototypicalNetwork(
        backbone_type=config.backbone_type,
        feature_dim=config.feature_dim,
        num_transformer_layers=config.num_transformer_layers,
        similarity=config.similarity_metric
    ).to(device)
    
    checkpoint = torch.load(config.best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded successfully")
    
    # Check checkpoint info
    if 'accuracy' in checkpoint:
        print(f"✓ Best training accuracy: {checkpoint['accuracy']:.4f}")
    if 'epoch' in checkpoint:
        print(f"✓ Trained for: {checkpoint['epoch'] + 1} epoch(s)")
    
    # Load test data
    print(f"\n⏳ Loading test dataset...")
    test_loader = EpisodicDataLoader(
        root_dir=config.dataset_dir,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        split='test',
        image_size=config.image_size
    )
    print(f"✓ Test dataset loaded")
    print(f"  - Classes available: {len(test_loader.class_images)}")
    
    # Evaluate
    print(f"\n⏳ Evaluating model on test set...")
    print(f"   (Testing on {config.n_way}-way {config.k_shot}-shot episodes)\n")
    
    evaluator = FewShotEvaluator(model, test_loader, device=device)
    results = evaluator.evaluate(num_episodes=100, n_way=config.n_way)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    mean_acc = results['mean_accuracy'] * 100
    std_acc = results['std_accuracy'] * 100
    
    print(f"\n📊 TEST ACCURACY:")
    print(f"   Mean Accuracy: {mean_acc:.2f}%")
    print(f"   Std Dev:       {std_acc:.2f}%")
    print(f"   Min Accuracy:  {min(results['accuracies']) * 100:.2f}%")
    print(f"   Max Accuracy:  {max(results['accuracies']) * 100:.2f}%")
    
    # Accuracy range
    print(f"\n📈 ACCURACY INTERPRETATION:")
    if mean_acc >= 90:
        print(f"   ✓✓✓ Excellent! Model is performing very well")
    elif mean_acc >= 80:
        print(f"   ✓✓ Good! Model is performing well")
    elif mean_acc >= 70:
        print(f"   ✓ Fair. Model has room for improvement")
    elif mean_acc >= 60:
        print(f"   ⚠ Needs improvement. Consider retraining")
    else:
        print(f"   ❌ Poor performance. Model needs significant improvement")
    
    print(f"\n📋 TEST EPISODES: 100")
    print(f"   - Each episode: {config.n_way}-way classification")
    print(f"   - Support set: {config.n_way} × {config.k_shot} images")
    print(f"   - Query set:   {config.n_way} × {config.n_query} images")
    
    # Save results
    results_file = 'results/evaluation_report.txt'
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model: {config.best_model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Backbone: {config.backbone_type}\n")
        f.write(f"Feature Dimension: {config.feature_dim}\n")
        f.write(f"Transformer Layers: {config.num_transformer_layers}\n\n")
        f.write("RESULTS:\n")
        f.write(f"Mean Accuracy: {mean_acc:.2f}%\n")
        f.write(f"Std Deviation: {std_acc:.2f}%\n")
        f.write(f"Min Accuracy: {min(results['accuracies']) * 100:.2f}%\n")
        f.write(f"Max Accuracy: {max(results['accuracies']) * 100:.2f}%\n")
        f.write(f"Test Episodes: 100\n")
        f.write(f"N-way: {config.n_way}-way\n")
        f.write(f"K-shot: {config.k_shot}-shot\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    # Next steps
    print(f"\n📌 NEXT STEPS:")
    print(f"   1. If accuracy is low (<70%), retrain with:")
    print(f"      - python train.py")
    print(f"   2. Try the Flask web app:")
    print(f"      - python app.py")
    print(f"      - Visit: http://localhost:5000")
    print(f"   3. For inference on custom images:")
    print(f"      - python infer.py")
    print()

if __name__ == "__main__":
    main()
