import torch
import torch.nn as nn
import torch.nn.functional as F
from .saai_aligner import SemanticAdversarialAligner
from .backbones import make_vgg_backbone, make_resnet_backbone, FeatureAdapter

class SAAICrowdCounter(nn.Module):
    """Complete SAAI-enhanced crowd counting model with semantic alignment"""
    
    def __init__(self, backbone_name='vgg16', pretrained=True, feature_dim=512):
        super(SAAICrowdCounter, self).__init__()
        
        # Dual-stream backbone networks
        if backbone_name == 'vgg16':
            self.rgb_backbone = make_vgg_backbone(pretrained)
            self.thermal_backbone = make_vgg_backbone(pretrained)
            backbone_channels = 512
        elif backbone_name == 'resnet50':
            self.rgb_backbone = make_resnet_backbone(pretrained)
            self.thermal_backbone = make_resnet_backbone(pretrained)
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Feature adaptation to standard dimension
        if backbone_channels != feature_dim:
            self.rgb_adapter = FeatureAdapter(backbone_channels, feature_dim)
            self.thermal_adapter = FeatureAdapter(backbone_channels, feature_dim)
        else:
            self.rgb_adapter = nn.Identity()
            self.thermal_adapter = nn.Identity()
        
        # SAAI semantic adversarial alignment module
        self.saai_aligner = SemanticAdversarialAligner(feature_dim=feature_dim)
        
        # Feature fusion module (inspired by broker modality approach)
        self.fusion_module = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Multi-scale regression head for density prediction
        self.regression_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)  # Ensure non-negative density values
        )
        
    def forward(self, rgb, thermal):
        """
        Forward pass through SAAI-enhanced crowd counter
        Args:
            rgb: [B, 3, H, W] RGB images
            thermal: [B, 3, H, W] Thermal images
        Returns:
            density_map: [B, 1, H', W'] Predicted density maps
            domain predictions for adversarial training
        """
        # Extract features from dual-stream backbones
        rgb_features = self.rgb_backbone(rgb)
        thermal_features = self.thermal_backbone(thermal)
        
        # Adapt feature dimensions if needed
        rgb_features = self.rgb_adapter(rgb_features)
        thermal_features = self.thermal_adapter(thermal_features)
        
        # SAAI semantic adversarial alignment
        rgb_aligned, thermal_aligned, domain_pred_rgb, domain_pred_thermal = \
            self.saai_aligner(rgb_features, thermal_features)
        
        # Multi-modal feature fusion
        fused_features = torch.cat([rgb_aligned, thermal_aligned], dim=1)
        fused_features = self.fusion_module(fused_features)
        
        # Density map prediction
        density_map = self.regression_head(fused_features)
        
        return density_map, domain_pred_rgb, domain_pred_thermal
