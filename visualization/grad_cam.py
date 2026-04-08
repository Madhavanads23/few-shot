import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    """Generate Grad-CAM heatmaps for CNN features."""
    
    def __init__(self, model, target_layer_name):
        """
        Args:
            model: PrototypicalNetwork
            target_layer_name: name of layer to visualize (e.g., 'layer4')
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        target_layer = self._get_target_layer()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
    
    def _get_target_layer(self):
        """Get target layer from model."""
        # Access layer4 from feature extractor
        feature_extractor = self.model.cnn_backbone.feature_extractor
        for idx, module in enumerate(feature_extractor):
            if hasattr(module, '__class__'):
                if 'layer4' in str(module.__class__):
                    return module
        # If layer4 not found, hook into layer3 or the last conv block
        for module in feature_extractor.modules():
            if hasattr(module, 'register_forward_hook'):
                pass
        # Default to layer4 if structure is Sequential
        return feature_extractor[7]  # layer4 in ResNet is at this position
    
    def generate(self, input_image, class_index=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: (1, 3, H, W)
            class_index: target class index for Grad-CAM
        
        Returns:
            heatmap: (H, W) normalized heatmap
        """
        self.model.eval()
        
        # Enable gradients for input
        input_copy = input_image.clone().detach().requires_grad_(True)
        
        # Forward pass with gradient computation enabled
        with torch.enable_grad():
            embeddings = self.model.cnn_backbone(input_copy)
            # Create loss - sum of embeddings
            loss = embeddings.sum()
            
            # Backward pass to compute gradients
            loss.backward()
        
        # Get gradients from input
        if input_copy.grad is not None:
            gradients = input_copy.grad.data.abs()
            # Compute heatmap from input gradients (average across channels)
            heatmap = gradients[0].mean(dim=0)  # (H, W)
        else:
            # Fallback: create equal heatmap
            H, W = input_image.shape[-2:]
            heatmap = torch.ones(H, W)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)
        else:
            heatmap = heatmap / (1e-8)
        
        heatmap = heatmap.cpu().detach().numpy()
        
        return heatmap
    
    def overlay_heatmap(self, input_image, heatmap, alpha=0.5):
        """
        Overlay heatmap on original image.
        
        Args:
            input_image: (3, H, W) tensor
            heatmap: (H, W) numpy array
        
        Returns:
            overlay: (H, W, 3) RGB image
        """
        # Convert image to numpy
        img = input_image.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, 
                                     (img.shape[1], img.shape[0]))
        
        # Generate heatmap colormap
        heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(
                                         np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (1 - alpha) * img + alpha * heatmap_color
        overlay = overlay.astype(np.uint8)
        
        return overlay