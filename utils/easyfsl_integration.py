"""
Integration module for Easy Few-Shot Learning library with your project.
Provides utilities to work with easyfsl's prototypical networks and other methods.
"""

import torch
import torch.nn as nn

# Import with proper error handling
from easyfsl.methods import PrototypicalNetworks, MatchingNetworks, RelationNetworks
from easyfsl.modules import resnet12, resnet18, resnet34
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class EasyFSLWrapper:
    """Wrapper to use Easy Few-Shot Learning methods with your project."""
    
    def __init__(self, method='prototypical', backbone='resnet12', 
                 num_ways=5, num_shots=5, device='cpu'):
        """
        Initialize Easy FSL wrapper.
        
        Args:
            method: 'prototypical', 'matching', or 'relation'
            backbone: 'resnet12', 'resnet18', 'resnet34'
            num_ways: n_way for few-shot learning
            num_shots: k_shot for few-shot learning
            device: 'cuda' or 'cpu' (defaults to cpu for compatibility)
        """
        # Ensure CUDA is available if requested
        if device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = torch.device(device)
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.method = method
        
        # Create backbone
        if backbone == 'resnet12':
            self.backbone = resnet12()
        elif backbone == 'resnet18':
            self.backbone = resnet18()
        elif backbone == 'resnet34':
            self.backbone = resnet34()
        else:
            raise ValueError(f"Backbone {backbone} not supported. Use 'resnet12', 'resnet18', or 'resnet34'")
        
        self.backbone = self.backbone.to(self.device)
        
        # Create method
        if method == 'prototypical':
            self.model = PrototypicalNetworks(self.backbone).to(self.device)
        elif method == 'matching':
            self.model = MatchingNetworks(self.backbone).to(self.device)
        elif method == 'relation':
            self.model = RelationNetworks(self.backbone).to(self.device)
        else:
            raise ValueError(f"Method {method} not supported")
    
    def train_step(self, support_images, support_labels, query_images, query_labels):
        """
        Single training step.
        
        Args:
            support_images: (n_way * k_shot, C, H, W)
            support_labels: (n_way * k_shot,)
            query_images: (n_query_per_class * n_way, C, H, W)
            query_labels: (n_query_per_class * n_way,)
            
        Returns:
            loss: scalar loss value
        """
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)
        
        # Process support set
        self.model.process_support_set(support_images, support_labels)
        
        # Forward pass
        output = self.model(query_images)
        
        # Compute loss (cross-entropy)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, query_labels)
        
        return loss
    
    def predict(self, support_images, support_labels, query_images):
        """
        Make predictions on query images.
        
        Args:
            support_images: (n_way * k_shot, C, H, W)
            support_labels: (n_way * k_shot,)
            query_images: (n_query, C, H, W)
            
        Returns:
            predictions: (n_query,) predicted class indices
            probabilities: (n_query, n_way) class probabilities
        """
        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        
        with torch.no_grad():
            # Process support set
            self.model.process_support_set(support_images, support_labels)
            
            # Forward pass
            logits = self.model(query_images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)
        
        return predictions, probabilities
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'method': self.method,
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])


class DataTransforms:
    """Standard transforms for few-shot learning."""
    
    @staticmethod
    def get_transforms(image_size=84, training=True):
        """Get data transforms."""
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if training:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize
            ])
        
        return transform
