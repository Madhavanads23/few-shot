import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import CNNBackbone
from .transformer_encoder import TransformerEncoder, FeatureFusion, CrossAttentionModule

class PrototypicalNetwork(nn.Module):
    """
    Few-shot learner using Prototypical Networks with hybrid similarity.
    """
    
    def __init__(self, backbone_type='resnet34', feature_dim=1024, 
                 num_transformer_layers=6, similarity='hybrid'):
        """
        Args:
            backbone_type: CNN backbone type
            feature_dim: embedding dimension
            num_transformer_layers: number of transformer layers
            similarity: 'euclidean', 'cosine', or 'hybrid'
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.similarity = similarity
        
        # CNN backbone
        self.cnn_backbone = CNNBackbone(backbone_type, feature_dim=feature_dim)
        
        # Transformer encoder for self-attention within support set
        self.transformer = TransformerEncoder(
            input_dim=feature_dim,
            num_heads=16,
            num_layers=num_transformer_layers
        )
        
        # Cross-attention module between support and query (NEW)
        self.cross_attention = CrossAttentionModule(
            feature_dim=feature_dim,
            num_heads=16,
            num_layers=2  # 2-layer cross-attention
        )
        
        # Feature fusion
        self.fusion = FeatureFusion(feature_dim, fusion_type='add')
        
        # Temperature for scaling
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def encode(self, x):
        """Extract embeddings using CNN only (transformer applied at episode level)."""
        # CNN features
        cnn_features = self.cnn_backbone(x)  # (batch, feature_dim)
        
        # L2 normalization
        embeddings = F.normalize(cnn_features, p=2, dim=-1)
        
        return embeddings
    
    def compute_prototypes(self, support_features, support_labels, n_way):
        """
        Compute class prototypes.
        
        Args:
            support_features: (n_way * k_shot, feature_dim)
            support_labels: (n_way * k_shot,)
            n_way: number of classes
        
        Returns:
            prototypes: (n_way, feature_dim)
        """
        prototypes = []
        
        for i in range(n_way):
            class_features = support_features[support_labels == i]
            prototype = class_features.mean(dim=0)
            # L2 normalize
            prototype = F.normalize(prototype, p=2, dim=-1)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_way, feature_dim)
        return prototypes
    
    def compute_similarity(self, query_features, prototypes):
        """
        Compute cosine similarity between query and prototypes.
        
        Args:
            query_features: (n_query * n_way, feature_dim) - L2 normalized
            prototypes: (n_way, feature_dim) - L2 normalized
        
        Returns:
            similarities: (n_query * n_way, n_way)
        """
        # For L2-normalized features, cosine similarity = dot product
        similarities = torch.mm(query_features, prototypes.t())
        
        # Scale by temperature
        similarities = similarities / self.temperature
        
        return similarities
    
    def forward(self, support_images, support_labels, query_images, n_way, k_shot=5):
        """
        Forward pass for few-shot learning with cross-attention.
        
        Args:
            support_images: (n_way * k_shot, 3, H, W)
            support_labels: (n_way * k_shot,)
            query_images: (n_way * n_query, 3, H, W)
            n_way: number of classes
            k_shot: number of support samples per class
        
        Returns:
            logits: (n_way * n_query, n_way)
        """
        # Encode support and query images (CNN only, L2 normalized)
        support_features = self.encode(support_images)  # (n_way*k_shot, feature_dim)
        query_features = self.encode(query_images)      # (n_way*n_query, feature_dim)
        
        # Reshape support features for transformer attention within each class
        # (n_way, k_shot, feature_dim) - group by class
        support_features_seq = support_features.view(n_way, k_shot, -1)
        
        # Apply transformer to refine support features
        # Process each class's support set independently through transformer
        refined_support_list = []
        for i in range(n_way):
            class_features = support_features_seq[i]  # (k_shot, feature_dim)
            
            # Add batch dimension for transformer: (1, k_shot, feature_dim)
            class_features_unsqueezed = class_features.unsqueeze(0)
            
            # Apply transformer: allows k_shot samples to attend to each other
            refined = self.transformer(class_features_unsqueezed)  # (1, k_shot, feature_dim)
            
            # Remove batch dimension and append
            refined_support_list.append(refined.squeeze(0))  # (k_shot, feature_dim)
        
        # Concatenate back: (n_way*k_shot, feature_dim)
        support_features = torch.cat(refined_support_list, dim=0)
        
        # Ensure L2 normalization after transformer
        support_features = F.normalize(support_features, p=2, dim=-1)
        
        # CROSS-ATTENTION: Refine query features by attending to support set (NEW)
        # Add batch dimension for cross-attention: (1, n_query*n_way, feature_dim)
        query_features_batch = query_features.unsqueeze(0)
        support_features_batch = support_features.unsqueeze(0)
        
        # Apply cross-attention: queries attend to support
        refined_query_features = self.cross_attention(
            query_features_batch, 
            support_features_batch
        )  # (1, n_query*n_way, feature_dim)
        
        # Remove batch dimension and normalize
        query_features = refined_query_features.squeeze(0)
        query_features = F.normalize(query_features, p=2, dim=-1)
        
        # Compute prototypes from refined support features
        prototypes = self.compute_prototypes(support_features, support_labels, n_way)
        
        # Compute similarities (logits) with refined query features
        logits = self.compute_similarity(query_features, prototypes)
        
        return logits
    
    def predict(self, support_images, support_labels, query_images, n_way):
        """Predict class labels for query images."""
        logits = self.forward(support_images, support_labels, 
                             query_images, n_way)
        predictions = torch.argmax(logits, dim=1)
        confidence = torch.max(torch.softmax(logits, dim=1), dim=1)[0]
        
        return predictions, confidence, logits