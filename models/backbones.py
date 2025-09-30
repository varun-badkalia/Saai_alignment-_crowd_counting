import torch
import torch.nn as nn
import torchvision.models as models

def make_vgg_backbone(pretrained=True):
    """Create VGG16 backbone for feature extraction"""
    backbone = models.vgg16(pretrained=pretrained).features
    # Remove last maxpool to maintain spatial resolution for dense prediction
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    return backbone

def make_resnet_backbone(pretrained=True):
    """Create ResNet50 backbone for feature extraction"""
    resnet = models.resnet50(pretrained=pretrained)
    # Remove final avgpool and fc layers
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    return backbone

class FeatureAdapter(nn.Module):
    """Adapter to match feature dimensions across different backbones"""
    
    def __init__(self, in_channels, out_channels=512):
        super(FeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)
