import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.prototypical_network import PrototypicalNetwork
from data.data_loader import EpisodicDataLoader
from visualization.grad_cam import GradCAM
from visualization.attention_viz import AttentionVisualizer
from utils.config import Config
import matplotlib.pyplot as plt

class FewShotInferencer:
    """Inference on few-shot learning model."""
    
    def __init__(self, model_path, config):
        """
        Args:
            model_path: path to saved model
            config: Config object
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model
        self.model = PrototypicalNetwork(
            backbone_type=config.backbone_type,
            feature_dim=config.feature_dim,
            num_transformer_layers=config.num_transformer_layers,
            similarity=config.similarity_metric
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")
        
        # Explainability modules
        self.grad_cam = GradCAM(self.model, 'layer4')
        self.attention_viz = AttentionVisualizer(self.model)
    
    def predict_with_explanation(self, support_images, support_labels, 
                                 query_image, n_way):
        """
        Predict class label with explanations.
        
        Args:
            support_images: (n_way * k_shot, 3, H, W)
            support_labels: (n_way * k_shot,)
            query_image: (1, 3, H, W)
            n_way: number of classes
        
        Returns:
            result: dict with prediction, confidence, heatmap, attention
        """
        with torch.no_grad():
            support_images = support_images.to(self.device)
            support_labels = support_labels.to(self.device)
            query_image = query_image.to(self.device)
            
            # Predict
            predictions, confidence, logits = self.model.predict(
                support_images, support_labels, query_image, n_way)
            
            pred_class = predictions[0].item()
            conf_score = confidence[0].item()
            
            # Grad-CAM
            heatmap = self.grad_cam.generate(query_image, class_index=pred_class)
            overlay = self.grad_cam.overlay_heatmap(query_image[0], heatmap)
            
            result = {
                'predicted_class': pred_class,
                'confidence': conf_score,
                'logits': logits[0].cpu().numpy(),
                'heatmap': heatmap,
                'overlay': overlay
            }
        
        return result
    
    def visualize_result(self, query_image, result, class_names=None):
        """Visualize prediction result with explanations."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = query_image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[0].imshow(img)
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(result['heatmap'], cmap='hot')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(result['overlay'])
        pred_class = result['predicted_class']
        conf = result['confidence']
        class_name = class_names[pred_class] if class_names else f"Class {pred_class}"
        axes[2].set_title(f'Prediction: {class_name}\nConfidence: {conf:.2f}')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    config = Config()
    inferencer = FewShotInferencer('checkpoints/best_model (1).pt', config)
    
    # Load test data
    val_loader = EpisodicDataLoader(
        root_dir=config.dataset_dir,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.n_query,
        split='test',
        image_size=config.image_size
    )
    
    # Generate episode
    support_img, support_lbl, query_img, query_lbl = \
        val_loader.generate_episode()
    
    # Predict with explanation
    result = inferencer.predict_with_explanation(
        support_img, support_lbl, query_img[0:1], config.n_way)
    
    print(f"✓ Prediction: Class {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Logits: {result['logits']}")
    
    # Visualize
    fig = inferencer.visualize_result(query_img[0:1], result)
    plt.savefig('result.png')
    print(f"✓ Result saved to result.png")