import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import random

class EpisodicDataLoader:
    """
    Few-shot episodic data loader supporting N-way K-shot learning.
    
    Generates episodes dynamically:
    - Support set: N classes × K images (N-way K-shot)
    - Query set: N classes × Q images (query images per class)
    """
    
    def __init__(self, root_dir, n_way=5, k_shot=5, n_query=5, 
                 split='train', image_size=84):
        """
        Args:
            root_dir: path to dataset root
            n_way: number of classes per episode
            k_shot: number of samples per class in support set
            n_query: number of query samples per class
            split: 'train' or 'test'
            image_size: 84 for standard few-shot setup
        """
        self.root_dir = root_dir
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.split = split
        self.image_size = image_size
        
        # Image transformations
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        # Load class structure
        self.class_dirs = self._load_class_structure()
        self.class_images = self._load_images()
    
    def _load_class_structure(self):
        """Load class directories from folder structure."""
        split_dir = os.path.join(self.root_dir, self.split)
        class_dirs = {}
        
        for class_name in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_name)
            if os.path.isdir(class_path):
                class_dirs[class_name] = class_path
        
        return class_dirs
    
    def _load_images(self):
        """Load all image paths for each class."""
        class_images = {}
        
        for class_name, class_path in self.class_dirs.items():
            image_files = [f for f in os.listdir(class_path) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_paths = [os.path.join(class_path, f) for f in image_files]
            class_images[class_name] = image_paths
        
        return class_images
    
    def generate_episode(self):
        """
        Generate a single few-shot episode.
        
        Returns:
            support_images: (n_way * k_shot, 3, image_size, image_size)
            support_labels: (n_way * k_shot,)
            query_images: (n_way * n_query, 3, image_size, image_size)
            query_labels: (n_way * n_query,)
        """
        # Sample N-way classes
        selected_classes = random.sample(list(self.class_images.keys()), self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for label, class_name in enumerate(selected_classes):
            images = self.class_images[class_name]
            
            # Sample K+Q images
            sampled_indices = random.sample(range(len(images)), 
                                           self.k_shot + self.n_query)
            
            # Support set
            for i in range(self.k_shot):
                img_path = images[sampled_indices[i]]
                img = self._load_image(img_path)
                support_images.append(img)
                support_labels.append(label)
            
            # Query set
            for i in range(self.n_query):
                img_path = images[sampled_indices[self.k_shot + i]]
                img = self._load_image(img_path)
                query_images.append(img)
                query_labels.append(label)
        
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return support_images, support_labels, query_images, query_labels
    
    def _load_image(self, img_path):
        """Load and transform image."""
        img = Image.open(img_path).convert('RGB')
        return self.transform(img)
    
    def __len__(self):
        """Number of episodes per epoch."""
        return 100
    
    def __iter__(self):
        """Iterate through episodes."""
        for _ in range(len(self)):
            yield self.generate_episode()


class EpisodicBatchSampler:
    """Wrapper for episode sampling in few-shot learning."""
    
    def __init__(self, data_loader, batch_size=1, num_episodes=100):
        """
        Args:
            data_loader: EpisodicDataLoader
            batch_size: MUST be 1 for episodic training (each episode is independent)
            num_episodes: number of episodes to generate
        """
        self.data_loader = data_loader
        # CRITICAL: batch_size must always be 1 for episodic learning
        # Each episode is a complete classification task with its own support/query sets
        self.batch_size = 1
        self.num_episodes = num_episodes
    
    def __iter__(self):
        """Iterate through episodes."""
        for _ in range(self.num_episodes):
            # Generate single episode (NOT concatenated with others)
            support_img, support_lbl, query_img, query_lbl = \
                self.data_loader.generate_episode()
            
            yield (support_img, support_lbl, query_img, query_lbl)
    
    def __len__(self):
        return self.num_episodes