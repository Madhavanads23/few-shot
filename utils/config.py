from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Config:
    """Configuration for few-shot learning project."""
    
    # Dataset
    dataset_dir: str = 'dataset'
    train_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    test_classes: List[int] = field(default_factory=lambda: [5, 6, 7, 8, 9])
    
    # Few-shot settings
    n_way: int = 5
    k_shot: int = 5
    n_query: int = 5
    image_size: int = 84
    
    # Model architecture
    backbone_type: str = 'resnet34'  # UPGRADED: Better feature extraction
    feature_dim: int = 1024  # UPGRADED: Higher dimensional embeddings
    num_transformer_layers: int = 6  # UPGRADED: Deeper transformer for better attention
    num_heads: int = 16  # UPGRADED: More attention heads for richer representations
    similarity_metric: str = 'hybrid'  # 'euclidean', 'cosine', 'hybrid'
    
    # Training
    learning_rate: float = 5e-4         # Standard rate for episodic training
    weight_decay: float = 1e-5
    num_epochs: int = 100               # 100 epochs (sufficient for convergence)
    batch_size: int = 1                 # MUST be 1 for episodic training
    num_train_episodes: int = 300       # Good balance between learning and time
    num_val_episodes: int = 50          # Validation episodes
    use_lr_scheduler: bool = True       # Enable learning rate scheduling
    scheduler_type: str = 'cosine'      # 'cosine' or 'step'
    
    # Device
    device: str = 'cuda'  # Use 'cuda' if available, otherwise falls back to 'cpu'
    seed: int = 42
    
    # Paths
    best_model_path: str = 'checkpoints/best_model.pt'
    results_dir: str = 'results'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.__dict__