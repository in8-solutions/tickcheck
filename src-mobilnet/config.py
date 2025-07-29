"""
Configuration for Mobile Tick Detection Model
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs" / "mobile"

# Data configuration
DATA_CONFIG = {
    "input_size": (224, 224),  # Mobile-optimized input size
    "crop_padding": 0.3,  # Padding around bounding boxes for positive samples
    "negative_crops_per_image": 3,  # Increased with relaxed overlap detection
    "train_split": 0.8,
    "val_split": 0.2,
    "max_chunk": 7,  # Only use chunks 1 through this number
    "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
    "std": [0.229, 0.224, 0.225],
}

# Model configuration
MODEL_CONFIG = {
    "architecture": "mobilenet_v3_small",  # Options: mobilenet_v3_small, efficientnet_b0
    "num_classes": 2,  # Binary: no tick (0), tick present (1)
    "pretrained": True,
    "dropout_rate": 0.2,
    "use_attention": False,  # Optional attention mechanism
}

# Training configuration
TRAINING_CONFIG = {
    "device": "cuda",
    "num_epochs": 100,  # Longer training for better convergence
    "batch_size": 256,  # Much larger batch size for 32GB GPU utilization
    "learning_rate": 0.0005,  # Lower learning rate for more stable training
    "weight_decay": 0.0001,
    "early_stopping_patience": 15,  # More patience for longer training
    "min_epochs": 25,  # Train longer before early stopping
    
    # Learning rate scheduling
    "lr_scheduler": {
        "name": "reduce_on_plateau",
        "factor": 0.3,  # More aggressive reduction
        "patience": 8,  # More patience before reducing LR
        "min_lr": 0.000001,  # Lower minimum LR
    },
    
    # Data loading
    "num_workers": 12,  # Increased for 24-core system
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,  # Enable for better performance with more workers
    
    # Mixed precision
    "use_amp": True,
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "train": {
        "horizontal_flip": True,
        "vertical_flip": False,  # Ticks are usually oriented naturally
        "rotate": {
            "enabled": True,
            "limit": 30,  # Smaller rotation for mobile realism
        },
        "scale": {
            "enabled": True,
            "min": 0.8,
            "max": 1.2,
        },
        "brightness": {
            "enabled": True,
            "limit": 0.2,
        },
        "contrast": {
            "enabled": True,
            "limit": 0.2,
        },
        "blur": {
            "enabled": True,
            "limit": 2,  # Lighter blur for mobile
        },
        "noise": {
            "enabled": True,
            "limit": 0.05,
        },
    },
    "val": {
        "horizontal_flip": False,
        "vertical_flip": False,
        "rotate": {"enabled": False},
        "scale": {"enabled": False},
        "brightness": {"enabled": False},
        "contrast": {"enabled": False},
        "blur": {"enabled": False},
        "noise": {"enabled": False},
    }
}

# Output configuration
OUTPUT_CONFIG = {
    "checkpoint_dir": OUTPUT_DIR / "checkpoints",
    "model_dir": OUTPUT_DIR / "models",
    "curves_dir": OUTPUT_DIR / "curves",
    "evaluation_dir": OUTPUT_DIR / "evaluation",
    "export_dir": OUTPUT_DIR / "export",
}

# Mobile export configuration
EXPORT_CONFIG = {
    "quantization": True,
    "pruning": False,  # Can be enabled for further size reduction
    "knowledge_distillation": False,  # Use RetinaNet as teacher
    "target_size_mb": 5,  # Target model size
    "target_inference_ms": 50,  # Target inference time
}

# Create output directories
for path in OUTPUT_CONFIG.values():
    path.mkdir(parents=True, exist_ok=True) 