import torch
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import EpisodicDataLoader, EpisodicBatchSampler
from models.prototypical_network import PrototypicalNetwork
from training.trainer import FewShotTrainer
from utils.config import Config

def main():
    """Main training script."""
    
    # Configuration
    config = Config()
    print(f"✓ Configuration loaded")
    print(f"  - Model: {config.backbone_type} + Transformer")
    print(f"  - Few-shot: {config.n_way}-way {config.k_shot}-shot")
    print(f"  - Feature dim: {config.feature_dim}")
    
    # Create directories
    os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Data loaders
    print(f"\n✓ Loading data...")
    train_loader = EpisodicDataLoader(
        root_dir=config.dataset_dir,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        split='train',
        image_size=config.image_size
    )
    
    val_loader = EpisodicDataLoader(
        root_dir=config.dataset_dir,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        split='test',
        image_size=config.image_size
    )
    
    train_sampler = EpisodicBatchSampler(
        train_loader,
        batch_size=1,  # Must be 1 for episodic training
        num_episodes=config.num_train_episodes
    )
    
    val_sampler = EpisodicBatchSampler(
        val_loader,
        batch_size=1,  # Must be 1 for episodic training
        num_episodes=config.num_val_episodes
    )
    
    print(f"  - Train: {len(train_sampler)} episodes")
    print(f"  - Validation: {len(val_sampler)} episodes")
    
    # Model
    print(f"\n✓ Building model...")
    model = PrototypicalNetwork(
        backbone_type=config.backbone_type,
        feature_dim=config.feature_dim,
        num_transformer_layers=config.num_transformer_layers,
        similarity=config.similarity_metric
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    
    # Trainer
    print(f"\n✓ Initializing trainer...")
    trainer = FewShotTrainer(model, train_sampler, val_sampler, 
                            config.to_dict())
    
    # Load existing checkpoint if available
    if os.path.exists(config.best_model_path):
        print(f"\n✓ Loading existing model from checkpoint...")
        trainer.load_checkpoint(config.best_model_path)
        print(f"  - Resuming fine-tuning from saved model")
    else:
        print(f"\n✓ No checkpoint found - training from scratch")
    
    # Train
    print(f"\n{'='*50}")
    print(f"Starting training...")
    print(f"{'='*50}\n")
    
    trainer.train(config.num_epochs)
    
    print(f"\n✓ Training completed!")
    print(f"  - Best model saved at: {config.best_model_path}")


if __name__ == "__main__":
    main()