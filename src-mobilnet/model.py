"""
Mobile Model Architecture for Tick Detection
Lightweight models suitable for mobile deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Any
import logging

from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class MobileTickClassifier(nn.Module):
    """
    Mobile-optimized binary classifier for tick detection
    """
    
    def __init__(self, 
                 architecture: str = "mobilenet_v3_small",
                 num_classes: int = 2,
                 pretrained: bool = True,
                 dropout_rate: float = 0.2,
                 use_attention: bool = False):
        
        super().__init__()
        
        self.architecture = architecture
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Build backbone
        self.backbone = self._build_backbone(architecture, pretrained)
        
        # Get feature dimension
        if architecture == "mobilenet_v3_small":
            feature_dim = 576  # MobileNetV3-Small output
        elif architecture == "efficientnet_b0":
            feature_dim = 1280  # EfficientNet-B0 output
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Optional attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 8, 1, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Created {architecture} model with {self._count_parameters()} parameters")
    
    def _build_backbone(self, architecture: str, pretrained: bool) -> nn.Module:
        """Build the backbone network"""
        
        if architecture == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=pretrained)
            # Remove the classifier head
            backbone = nn.Sequential(*list(model.children())[:-1])
            
        elif architecture == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            # Remove the classifier head
            backbone = nn.Sequential(*list(model.children())[:-1])
            
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        return backbone
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing:
            - 'logits': Raw logits (batch_size, num_classes)
            - 'probabilities': Softmax probabilities (batch_size, num_classes)
            - 'attention_weights': Optional attention weights
        """
        
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # Classification
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'attention_weights': attention_weights
        }


class LightweightTickClassifier(nn.Module):
    """
    Ultra-lightweight classifier for very constrained mobile environments
    """
    
    def __init__(self, 
                 input_size: int = 224,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        
        super().__init__()
        
        # Very lightweight architecture
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 1
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        logger.info(f"Created lightweight model with {self._count_parameters()} parameters")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        features = self.features(x)
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probabilities
        }


def create_model(config: Dict[str, Any] = None) -> nn.Module:
    """
    Factory function to create a model based on configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    
    if config is None:
        config = MODEL_CONFIG
    
    architecture = config.get('architecture', 'mobilenet_v3_small')
    num_classes = config.get('num_classes', 2)
    pretrained = config.get('pretrained', True)
    dropout_rate = config.get('dropout_rate', 0.2)
    use_attention = config.get('use_attention', False)
    
    if architecture == "lightweight":
        return LightweightTickClassifier(
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        return MobileTickClassifier(
            architecture=architecture,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probabilities shape: {output['probabilities'].shape}")
    print(f"Probabilities sum: {output['probabilities'].sum(dim=1)}") 