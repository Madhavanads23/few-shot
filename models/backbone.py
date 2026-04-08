import torch
import torch.nn as nn
import torchvision.models as models

class CNNBackbone(nn.Module):
    """
    CNN backbone for feature extraction.
    Supports ResNet18 and MobileNetV2.
    """
    
    def __init__(self, backbone_type='resnet18', pretrained=True, 
                 feature_dim=512):
        """
        Args:
            backbone_type: 'resnet18' or 'mobilenet'
            pretrained: use ImageNet pretrained weights
            feature_dim: output feature dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        if backbone_type == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            # Extract features from conv layers (remove FC layer)
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool,
                nn.Flatten()
            )
            backbone_out_dim = 512  # ResNet18 outputs 512 features
        
        elif backbone_type == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            # Extract features from conv layers (remove FC layer)
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool,
                nn.Flatten()
            )
            backbone_out_dim = 512  # ResNet34 outputs 512 features
        
        elif backbone_type == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            # Extract features from conv layers (remove FC layer)
            self.feature_extractor = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
                resnet.avgpool,
                nn.Flatten()
            )
            backbone_out_dim = 2048  # ResNet50 outputs 2048 features
            
        elif backbone_type == 'mobilenet':
            mobilenet = models.mobilenet_v2(pretrained=pretrained)
            # Extract features from conv layers (remove FC layer)
            self.feature_extractor = nn.Sequential(
                mobilenet.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            backbone_out_dim = 1280  # MobileNetV2 outputs 1280 features
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")
        
        # Projection to feature_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 3, H, W)
        
        Returns:
            features: (batch, feature_dim)
        """
        features = self.feature_extractor(x)
        embeddings = self.projection(features)
        return embeddings
    
    def extract_features(self, x):
        """Extract features before projection."""
        # Return features before flattening if needed for Grad-CAM
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            # Stop before flatten to preserve spatial dimensions for visualization
            if isinstance(layer, nn.Flatten):
                # Return flattened features
                return x
        return x