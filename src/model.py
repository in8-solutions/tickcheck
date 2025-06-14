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
import torch.nn.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead

class RetinaNetHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels, num_anchors
        )
        
    def forward(self, x):
        cls_logits = []
        bbox_regression = []
        
        # x is a list of feature maps from FPN
        for feature in x:
            # Ensure feature is 4D (batch, channels, height, width)
            if len(feature.shape) == 3:
                feature = feature.unsqueeze(0)
            cls_logits.append(self.classification_head(feature))
            bbox_regression.append(self.regression_head(feature))
            
        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression
        }
        
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        """
        Compute the losses for the RetinaNet head.
        
        Args:
            targets (List[Dict]): List of target dictionaries
            head_outputs (Dict): Dictionary containing classification and regression outputs
            anchors (List[Tensor]): List of anchor boxes for each feature level
            matched_idxs (List[Tensor]): List of matched indices for each feature level
            
        Returns:
            Dict containing classification and regression losses
        """
        cls_logits = head_outputs["cls_logits"]
        bbox_regression = head_outputs["bbox_regression"]
        
        # Compute classification loss
        cls_loss = self.classification_head.compute_loss(
            targets, cls_logits, anchors, matched_idxs
        )
        
        # Compute regression loss
        reg_loss = self.regression_head.compute_loss(
            targets, bbox_regression, anchors, matched_idxs
        )
        
        return {
            "loss_classifier": cls_loss,
            "loss_box_reg": reg_loss
        }

class TickDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create the internal RetinaNet model with COCO weights
        self.model = retinanet_resnet50_fpn(
            weights=None,  # Don't use pretrained weights initially
            num_classes=2,  # Our 2 classes: background and tick
            min_size=config['model']['min_size'],
            max_size=config['model']['max_size'],
            box_score_thresh=config['model']['box_score_thresh'],
            box_nms_thresh=config['model']['box_nms_thresh'],
            box_detections_per_img=config['model']['box_detections_per_img']
        )
        
        # Update anchor sizes and ratios
        anchor_sizes = config['model']['anchor_sizes']
        anchor_ratios = config['model']['anchor_ratios']
        
        # Create anchor generator with updated sizes
        anchor_generator = AnchorGenerator(
            sizes=tuple((s,) for s in anchor_sizes),
            aspect_ratios=tuple((r,) for r in anchor_ratios)
        )
        
        # Update the anchor generator in the head
        self.model.head.anchor_generator = anchor_generator
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
                
    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        
        Args:
            images (List[Tensor]): List of images to process
            targets (List[Dict], optional): List of target dictionaries containing:
                - boxes (Tensor[N, 4]): Ground truth boxes in [x1, y1, x2, y2] format
                - labels (Tensor[N]): Class labels for each box (0 for background, 1 for tick)
                
        Returns:
            During training:
                Dict containing losses
            During inference:
                List[Dict] containing predictions with boxes, labels, and scores
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
            
        try:
            # Convert images to list if they're not already
            if not isinstance(images, list):
                images = [images]
                
            # Convert targets to list if they're not already
            if targets is not None and not isinstance(targets, list):
                targets = [targets]
                
            # Forward pass through the model
            return self.model(images, targets)
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Images shape: {[img.shape for img in images]}")
            if targets is not None:
                print(f"Targets: {targets}")
            raise

def create_model(config):
    """Create and configure the model based on the provided configuration."""
    model_config = config['model']
    
    # Create model with pretrained weights
    model = TickDetector(config)
    
    # Set device
    device = torch.device(config['training']['device'])
    model = model.to(device)
    
    return model 