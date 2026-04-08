import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for global attention on CNN features.
    """
    
    def __init__(self, input_dim=512, num_heads=8, num_layers=2, 
                 dim_feedforward=2048, dropout=0.1):
        """
        Args:
            input_dim: input feature dimension
            num_heads: number of attention heads
            num_layers: number of transformer layers
            dim_feedforward: dimension of feedforward network
            dropout: dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.positional_encoding = PositionalEncoding(input_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(input_dim)
        )
        
        # Store attention maps for visualization
        self.attention_maps = {}
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, input_dim) where seq_len is N*K or N*Q
            return_attention: whether to return attention weights
        
        Returns:
            embeddings: (batch, seq_len, input_dim)
            attention: dict of attention maps if return_attention=True
        """
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionModule(nn.Module):
    """
    Cross-attention between support and query sets for better matching.
    Allows queries to attend to all support samples for refined matching.
    """
    
    def __init__(self, feature_dim=1024, num_heads=16, num_layers=2, 
                 dim_feedforward=2048, dropout=0.1):
        """
        Args:
            feature_dim: dimension of features
            num_heads: number of attention heads
            num_layers: number of cross-attention layers
            dim_feedforward: feedforward network dimension
            dropout: dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_layers)
        ])
        
        # Feedforward networks
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, feature_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_norm_layers = nn.ModuleList([
            nn.LayerNorm(feature_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, query_features, support_features):
        """
        Apply cross-attention: queries attend to support set.
        
        Args:
            query_features: (batch, n_query*n_way, feature_dim) - query set
            support_features: (batch, n_way*k_shot, feature_dim) - support set
        
        Returns:
            refined_query: (batch, n_query*n_way, feature_dim) - refined queries
        """
        x = query_features
        
        for i, (cross_attn, norm, ff, final_norm) in enumerate(
            zip(self.cross_attention_layers, self.norm_layers, 
                self.feedforward_layers, self.final_norm_layers)):
            
            # Cross-attention: queries (Q) attend to support set (K, V)
            attn_out, _ = cross_attn(x, support_features, support_features)
            
            # Residual connection + Layer norm
            x = norm(x + attn_out)
            
            # Feedforward
            ff_out = self.feedforward_layers[i](x)
            
            # Residual connection + Layer norm
            x = final_norm(x + ff_out)
        
        return x


class FeatureFusion(nn.Module):
    """Fuse CNN and Transformer features."""
    
    def __init__(self, feature_dim=512, fusion_type='concat'):
        """
        Args:
            feature_dim: dimension of input features
            fusion_type: 'concat', 'add', or 'attention'
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = nn.Linear(feature_dim * 2, feature_dim)
        elif fusion_type == 'add':
            self.fusion = nn.Identity()
        elif fusion_type == 'attention':
            self.fusion = nn.MultiheadAttention(feature_dim, num_heads=8, 
                                               batch_first=True)
    
    def forward(self, cnn_features, transformer_features):
        """Fuse CNN and Transformer features."""
        if self.fusion_type == 'concat':
            return self.fusion(torch.cat([cnn_features, transformer_features], 
                                        dim=-1))
        elif self.fusion_type == 'add':
            return cnn_features + transformer_features
        elif self.fusion_type == 'attention':
            fused, _ = self.fusion(transformer_features, cnn_features, 
                                   cnn_features)
            return fused