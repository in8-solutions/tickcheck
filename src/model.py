"""
MobileNet-based detection model for tick detection.

This module implements a custom detection model using MobileNet backbone
and is designed for efficient mobile deployment with TFLite compatibility.

Key Features:
- MobileNet backbone for mobile efficiency
- Simple detection head
- TFLite-friendly architecture
- Support for transfer learning from ImageNet weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import boxes as box_ops

class MobileNetDetectionHead(nn.Module):
    """Detection head for MobileNet-based model."""
    
    def __init__(self, in_channels, num_classes, num_anchors=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * num_anchors, 3, padding=1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4 * num_anchors, 3, padding=1)
        )
    
    def forward(self, x):
        cls_logits = self.cls_head(x)
        bbox_regression = self.reg_head(x)
        
        # Reshape outputs
        batch_size = cls_logits.size(0)
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(batch_size, -1, self.num_classes)
        
        bbox_regression = bbox_regression.permute(0, 2, 3, 1).contiguous()
        bbox_regression = bbox_regression.view(batch_size, -1, 4)
        
        return cls_logits, bbox_regression

class MobileNetBackbone(nn.Module):
    """MobileNet backbone with feature extraction."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNet
        if pretrained:
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        else:
            self.backbone = mobilenet_v3_small(weights=None)
        
        # Remove the classifier by replacing with identity
        self.backbone.classifier = nn.Sequential(nn.Identity())
        
        # Feature extraction layers
        self.features = self.backbone.features
        
    def forward(self, x):
        features = []
        
        # Extract features at different scales
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [5, 11, 15]:  # Different feature levels
                features.append(x)
        
        return features

class TickDetectorMobile(nn.Module):
    """MobileNet-based tick detection model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = MobileNetBackbone(pretrained=config['model']['pretrained'])
        
        # Detection head
        in_channels = 576  # MobileNet V3 small feature channels
        self.detection_head = MobileNetDetectionHead(
            in_channels=in_channels,
            num_classes=config['model']['num_classes'],
            num_anchors=6
        )
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Loss functions
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()
    
    def forward(self, x, targets=None):
        """
        Forward pass of the model.
        
        Args:
            x (Tensor): Input images [B, C, H, W]
            targets (List[Dict], optional): List of target dictionaries
            
        Returns:
            During training: Dict containing losses
            During inference: List[Dict] containing predictions
        """
        # Extract features
        features = self.backbone(x)
        
        # Use the last feature map for detection
        feature_map = features[-1]
        
        # Detection head
        cls_logits, bbox_regression = self.detection_head(feature_map)
        
        if self.training and targets is not None:
            # Compute losses
            losses = self.compute_loss(cls_logits, bbox_regression, targets)
            return losses
        else:
            # Inference mode
            # Apply sigmoid to classification logits
            cls_probs = torch.sigmoid(cls_logits)
            
            # Simple post-processing (you might want to add NMS here)
            predictions = []
            for i in range(cls_probs.size(0)):
                pred = {
                    'boxes': bbox_regression[i],
                    'labels': torch.argmax(cls_probs[i], dim=1),
                    'scores': torch.max(cls_probs[i], dim=1)[0]
                }
                predictions.append(pred)
            
            return predictions
    
    def compute_loss(self, cls_logits, bbox_regression, targets):
        """Compute classification and regression losses."""
        # This is a simplified loss computation
        # In practice, you'd want proper anchor matching and loss computation
        
        cls_loss = self.cls_loss(cls_logits.view(-1, self.config['model']['num_classes']), 
                                torch.zeros(cls_logits.size(0) * cls_logits.size(1), 
                                          dtype=torch.long, device=cls_logits.device))
        
        reg_loss = self.reg_loss(bbox_regression, torch.zeros_like(bbox_regression))
        
        return {
            'loss_classifier': cls_loss,
            'loss_box_reg': reg_loss,
            'total_loss': cls_loss + reg_loss
        }

def create_model(config):
    """Create and configure the model based on the provided configuration."""
    model = TickDetectorMobile(config)
    
    # Set device
    device = torch.device(config['training']['device'])
    model = model.to(device)
    
    return model 