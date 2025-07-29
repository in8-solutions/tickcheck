"""
Utility functions for Mobile Tick Detection
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_epochs: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.epoch = 0
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop"""
        self.epoch += 1
        
        if self.epoch < self.min_epochs:
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_loss = float('inf')
        self.epoch = 0


def save_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metric_value: float,
                   filepath: Path,
                   metric_name: str = 'loss') -> None:
    """Save model checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        metric_name: metric_value,
        'model_config': {
            'architecture': getattr(model, 'architecture', 'unknown'),
            'num_classes': getattr(model, 'num_classes', 2)
        }
    }
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   filepath: Path) -> Dict[str, Any]:
    """Load model checkpoint"""
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    logger.info(f"Epoch: {checkpoint['epoch']}")
    logger.info(f"Best metric: {checkpoint.get('loss', checkpoint.get('f1', 'unknown'))}")
    
    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def freeze_layers(model: nn.Module, num_layers: int = 0) -> None:
    """Freeze the first num_layers of the model"""
    if num_layers == 0:
        return
    
    # Get all parameters
    params = list(model.parameters())
    
    # Freeze the first num_layers
    for i, param in enumerate(params):
        if i < num_layers:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    logger.info(f"Froze first {num_layers} layers")


def unfreeze_all_layers(model: nn.Module) -> None:
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True
    
    logger.info("Unfroze all layers")


def get_layer_groups(model: nn.Module) -> list:
    """Get parameter groups for different learning rates"""
    # This is a simple implementation - can be customized based on model architecture
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    return [
        {'params': backbone_params, 'lr': 0.0001},  # Lower LR for backbone
        {'params': classifier_params, 'lr': 0.001}   # Higher LR for classifier
    ]


def calculate_inference_time(model: nn.Module, 
                           input_size: tuple = (1, 3, 224, 224),
                           num_runs: int = 100,
                           device: str = 'cpu') -> Dict[str, float]:
    """Calculate model inference time"""
    
    model.eval()
    model.to(device)
    
    # Warm up
    dummy_input = torch.randn(input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if device == 'cuda':
        start_time.record()
    else:
        start_time = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    if device == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    else:
        import time
        total_time = time.time() - start_time
    
    avg_time = total_time / num_runs
    fps = 1.0 / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'fps': fps,
        'total_time_s': total_time
    }


def print_model_summary(model: nn.Module) -> None:
    """Print a summary of the model"""
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Size: {get_model_size_mb(model):.2f} MB")
    
    # Print layer information
    print("\nLayer Information:")
    total_params = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            total_params += params
            print(f"  {name}: {params:,} parameters")
    
    print(f"\nTotal trainable parameters: {total_params:,}")


if __name__ == "__main__":
    # Test utilities
    from model import create_model
    
    model = create_model()
    print_model_summary(model)
    
    # Test inference time
    inference_stats = calculate_inference_time(model, device='cpu')
    print(f"\nInference time: {inference_stats['avg_inference_time_ms']:.2f} ms")
    print(f"FPS: {inference_stats['fps']:.2f}") 