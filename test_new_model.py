import torch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from models.prototypical_network import PrototypicalNetwork
from data.data_loader import EpisodicDataLoader
from evaluation.evaluate import FewShotEvaluator
from utils.config import Config

# Test the new model
config = Config()
config.best_model_path = 'checkpoints/best_model (1).pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 70)
print('TESTING NEW MODEL: best_model (1).pt')
print('=' * 70)

if not os.path.exists(config.best_model_path):
    print(f'ERROR: Model not found at {config.best_model_path}')
    sys.exit(1)

print(f'\nLoading model from: {config.best_model_path}')
model = PrototypicalNetwork(
    backbone_type=config.backbone_type,
    feature_dim=config.feature_dim,
    num_transformer_layers=config.num_transformer_layers,
    similarity=config.similarity_metric
).to(device)

try:
    checkpoint = torch.load(config.best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('✓ Model loaded successfully')
    print(f'Device: {device}')
    
    # Load test data
    print(f'\nLoading test dataset...')
    test_loader = EpisodicDataLoader(
        root_dir=config.dataset_dir,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        split='test',
        image_size=config.image_size
    )
    print(f'✓ Test dataset loaded')
    
    # Evaluate
    print(f'\nEvaluating model on 50 test episodes...')
    evaluator = FewShotEvaluator(model, test_loader, device=device)
    results = evaluator.evaluate(num_episodes=50, n_way=config.n_way)
    
    print(f'\n' + '=' * 70)
    print(f'EVALUATION RESULTS')
    print(f'=' * 70)
    print(f'Accuracy: {results["mean_accuracy"]:.2%}')
    print(f'Std Dev: {results["std_accuracy"]:.4f}')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
