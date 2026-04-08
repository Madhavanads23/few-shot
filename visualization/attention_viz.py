import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    """Visualize Transformer attention maps."""
    
    def __init__(self, model):
        """
        Args:
            model: PrototypicalNetwork with Transformer
        """
        self.model = model
        self.attention_weights = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        # Hook into transformer encoder
        for module in self.model.transformer.transformer_encoder.layers:
            def forward_hook(m, input, output):
                self.attention_weights.append(output)
            
            module.self_attn.register_forward_hook(forward_hook)
    
    def extract_attention_weights(self, images, support_labels=None, n_way=None):
        """Extract attention weights for input images."""
        self.attention_weights = []
        
        with torch.no_grad():
            # Just encode the images to extract attention
            self.model.encode(images)
        
        return self.attention_weights
    
    def visualize_attention(self, attention_weights, num_heads=8):
        """
        Visualize attention weights.
        
        Args:
            attention_weights: list of attention tensors from model
            num_heads: number of attention heads
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        if len(attention_weights) > 0:
            attn = attention_weights[0]  # First layer
            
            # Average over batch
            attn_avg = attn.mean(dim=0)  # (num_heads, seq_len, seq_len)
            
            for head_idx in range(min(num_heads, 8)):
                ax = axes[head_idx // 4, head_idx % 4]
                
                att_map = attn_avg[head_idx].cpu().numpy()
                sns.heatmap(att_map, ax=ax, cmap='Blues', 
                           cbar=True, square=True)
                ax.set_title(f'Head {head_idx}')
                ax.set_xlabel('Keys')
                ax.set_ylabel('Queries')
        
        plt.tight_layout()
        return fig


def plot_attention_distribution(attention_weights, save_path=None):
    """Plot attention weight distribution."""
    fig, ax = plt.subplots()
    
    for layer_idx, weights in enumerate(attention_weights):
        flat_weights = weights.flatten().cpu().numpy()
        ax.hist(flat_weights, bins=50, alpha=0.5, 
               label=f'Layer {layer_idx}')
    
    ax.set_xlabel('Attention Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Attention Weight Distribution')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()
    return fig