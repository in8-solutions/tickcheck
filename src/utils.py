import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.ops import box_convert
import os
import json

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directories(config):
    """Create necessary directories for training."""
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['training']['output_dir'], 'training_curves'), exist_ok=True)

def collate_fn(batch):
    """Custom collate function for data loader."""
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images), targets

def visualize_prediction(image, boxes, scores, labels, class_names, threshold=0.5):
    """Visualize detection results on an image."""
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Filter detections based on threshold
    mask = scores >= threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Plot each box
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.cpu().numpy()
        class_name = class_names[label.item()]
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='red', linewidth=2
        ))
        plt.text(
            x1, y1 - 5,
            f'{class_name}: {score:.2f}',
            color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.axis('off')
    return plt.gcf()

def save_checkpoint(model, optimizer, epoch, config, filename):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_loss': model.train_loss if hasattr(model, 'train_loss') else None,
        'val_loss': model.val_loss if hasattr(model, 'val_loss') else None,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'history': model.history if hasattr(model, 'history') else None
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['config']

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

def plot_training_curves(history, output_dir):
    """Plot and save training curves."""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot learning rate
    plt.subplot(2, 1, 2)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    
    # Save the plot
    curves_dir = os.path.join(output_dir, 'training_curves')
    os.makedirs(curves_dir, exist_ok=True)  # Ensure directory exists
    plt.savefig(os.path.join(curves_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save raw data for future reference
    with open(os.path.join(curves_dir, 'training_history.json'), 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'learning_rates': history['learning_rates'],
            'epochs': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']),
            'best_epoch': history['val_loss'].index(min(history['val_loss'])) + 1
        }, f, indent=4) 