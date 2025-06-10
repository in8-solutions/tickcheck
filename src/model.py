"""
RetinaNet model implementation for tick detection.

This module implements a custom RetinaNet model based on PyTorch's implementation.
The model uses a ResNet50-FPN backbone and is designed for efficient single-stage
object detection with balanced handling of class imbalance through Focal Loss.

Key Features:
- ResNet50 backbone with Feature Pyramid Network (FPN)
- Custom anchor generation
- Focal Loss for handling class imbalance
- Support for transfer learning from ImageNet weights
"""

import math
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.resnet import ResNet50_Weights

class CustomRetinaNet(nn.Module):
    """Custom RetinaNet implementation for tick detection.
    
    This class implements a RetinaNet model with:
    1. ResNet50-FPN backbone (optionally pretrained on ImageNet)
    2. Custom anchor generation with multiple scales and aspect ratios
    3. Focal Loss for handling class imbalance
    4. Configurable backbone freezing for transfer learning
    
    The model follows the architecture described in the RetinaNet paper:
    "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)
    
    Args:
        num_classes (int): Number of object classes to detect
        pretrained (bool): Whether to use ImageNet pretrained backbone
        freeze_backbone (bool): Whether to freeze the backbone for transfer learning
    """
    def __init__(self, num_classes, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Create backbone with updated weights parameter
        backbone = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
            trainable_layers=0 if freeze_backbone else 5
        )
        
        # Create anchor generator
        anchor_generator = AnchorGenerator(
            sizes=tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) 
                  for x in [32, 64, 128, 256, 512]),
            aspect_ratios=tuple((0.5, 1.0, 2.0) for _ in range(5))
        )
        
        # Create head
        head = RetinaNetHead(
            backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            num_classes,
            norm_layer=nn.BatchNorm2d
        )
        
        # Create RetinaNet model
        self.model = RetinaNet(
            backbone,
            num_classes,
            anchor_generator=anchor_generator,
            head=head,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            topk_candidates=1000
        )
        
        # Initialize weights
        if pretrained:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classification and regression layers."""
        for name, param in self.model.head.named_parameters():
            if "classification" in name:
                if param.dim() > 1:
                    torch.nn.init.normal_(param, mean=0.0, std=0.01)
                else:
                    torch.nn.init.constant_(param, -math.log((1 - 0.01) / 0.01))
            elif "bbox_regression" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.01)
    
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should not be None")
        
        try:
            return self.model(images, targets)
        except Exception as e:
            print(f"Error in model forward pass: {str(e)}")
            raise e

def create_model(config):
    """Create and configure the RetinaNet model based on config settings.
    
    This function:
    1. Creates a CustomRetinaNet instance with specified parameters
    2. Moves the model to the appropriate device (GPU/CPU)
    3. Prints device information for debugging
    
    Args:
        config (dict): Configuration dictionary containing model settings
            Required keys:
            - model.num_classes: Number of object classes
            - model.pretrained: Whether to use pretrained backbone
            - model.freeze_backbone: Whether to freeze backbone
            - model.device: Device to place model on
    
    Returns:
        CustomRetinaNet: Configured model instance on specified device
    """
    model = CustomRetinaNet(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        freeze_backbone=config['model']['freeze_backbone']
    )
    
    # Move model to appropriate device
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        
    model = model.to(device)
    
    return model 