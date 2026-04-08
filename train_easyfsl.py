"""
Training script using Easy Few-Shot Learning library.
This demonstrates how to use easyfsl for few-shot learning on CIFAR-10 or other datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.easyfsl_integration import EasyFSLWrapper, DataTransforms
from data.data_loader import EpisodicDataLoader
from utils.config import Config


class EasyFSLTrainer:
    """Trainer using Easy Few-Shot Learning methods."""
    
    def __init__(self, config, method='prototypical'):
        """
        Initialize trainer.
        
        Args:
            config: Config object
            method: 'prototypical', 'matching', or 'relation'
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = EasyFSLWrapper(
            method=method,
            backbone='resnet12',
            num_ways=config.n_way,
            num_shots=config.k_shot,
            device=self.device
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.backbone.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        if config.use_lr_scheduler:
            if config.scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=config.num_epochs
                )
            else:
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=config.num_epochs // 3,
                    gamma=0.5
                )
        else:
            self.scheduler = None
        
        self.best_val_acc = 0.0
        self.training_history = {'train_loss': [], 'val_acc': []}
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.backbone.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", 
                   total=self.config.num_train_episodes)
        
        for i in range(self.config.num_train_episodes):
            # Create episode
            support_images, support_labels, query_images, query_labels = next(iter(train_loader))
            
            # Forward pass
            loss = self.model.train_step(
                support_images, support_labels,
                query_images, query_labels
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.update(1)
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        pbar.close()
        
        avg_loss = total_loss / num_batches
        self.training_history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.backbone.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(self.config.num_val_episodes):
                support_images, support_labels, query_images, query_labels = next(iter(val_loader))
                
                # Predict
                predictions, _ = self.model.predict(
                    support_images, support_labels, query_images
                )
                
                correct += (predictions.cpu() == query_labels).sum().item()
                total += len(query_labels)
        
        accuracy = 100.0 * correct / total
        self.training_history['val_acc'].append(accuracy)
        
        return accuracy
    
    def train(self, train_loader, val_loader):
        """Train model for multiple epochs."""
        print(f"\n{'='*70}")
        print(f"TRAINING with Easy Few-Shot Learning ({self.model.method})")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Method: {self.model.method}")
        print(f"Few-shot: {self.config.n_way}-way {self.config.k_shot}-shot")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_acc = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print stats
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(self.config.best_model_path)
                print(f"  ✓ New best model saved! (acc: {val_acc:.2f}%)")
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"Model saved to: {self.config.best_model_path}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, path):
        """Save training checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
    
    def save_history(self, path):
        """Save training history."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def main():
    """Main training function."""
    
    # Configuration
    config = Config()
    
    # Create directories
    os.makedirs(os.path.dirname(config.best_model_path), exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Data loaders
    print(f"\n✓ Loading data from: {config.dataset_dir}")
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
    
    # Create trainer
    trainer = EasyFSLTrainer(config, method='prototypical')
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save history
    history_path = os.path.join(config.results_dir, 'training_history_easyfsl.json')
    trainer.save_history(history_path)
    print(f"✓ Training history saved to: {history_path}")


if __name__ == '__main__':
    main()
