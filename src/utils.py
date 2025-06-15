"""
Utility functions for tick detection model.

This module provides various utility functions used throughout the project for:
- Configuration management
- Directory handling
- Data loading
- Visualization
- Checkpoint management
- Metric tracking
- Training curve plotting

Key Features:
- YAML configuration loading
- Directory structure management
- Custom data collation
- Detection visualization
- Model checkpointing
- Training metrics tracking
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.ops import box_convert
import os
import json
from typing import List, Tuple
from copy import deepcopy
from math import ceil
import shutil
from datetime import datetime

def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML configuration file
    
    Returns:
        dict: Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_directories(config):
    """
    Create necessary directories for training.
    
    Args:
        config: Configuration dictionary
    """
    # Create output directories
    os.makedirs(config['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # Create temp directory for annotations
    os.makedirs('temp_annotations', exist_ok=True)

def collate_fn(batch):
    """Custom collate function for object detection data loading.
    
    This function handles batching of images and targets for the data loader,
    ensuring proper stacking of images while keeping target dictionaries separate.
    
    Args:
        batch (list): List of (image, target) tuples
    
    Returns:
        tuple: (stacked_images, list_of_targets)
    """
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return torch.stack(images), targets

def visualize_prediction(image, boxes, scores, labels, class_names, threshold=0.5):
    """Visualize object detection results on an image.
    
    This function:
    1. Denormalizes the image
    2. Draws bounding boxes around detections
    3. Adds class labels and confidence scores
    4. Handles confidence thresholding
    
    Args:
        image (torch.Tensor): The input image tensor
        boxes (torch.Tensor): Detected bounding boxes
        scores (torch.Tensor): Detection confidence scores
        labels (torch.Tensor): Class labels for each detection
        class_names (list): List of class names for label mapping
        threshold (float): Confidence threshold for showing detections
    
    Returns:
        matplotlib.figure.Figure: Figure with visualized detections
    """
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
    """Save model checkpoint with training state.
    
    Saves:
    - Model state dict
    - Optimizer state dict
    - Current epoch
    - Configuration
    - Training metrics
    - Learning rate
    - Training history
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer
        epoch (int): Current epoch number
        config (dict): Configuration dictionary
        filename (str): Path to save checkpoint
    """
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
    """Load model checkpoint and restore training state.
    
    Args:
        model (nn.Module): The model to load weights into
        optimizer (torch.optim.Optimizer): The optimizer to restore state
        filename (str): Path to checkpoint file
    
    Returns:
        tuple: (epoch_number, config_dict)
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['config']

class AverageMeter:
    """Computes and stores the average and current value of metrics.
    
    This class provides:
    - Running average calculation
    - Current value tracking
    - Simple update mechanism
    
    Useful for tracking metrics like loss values during training.
    """
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
    """Plot training curves and save to file."""
    plt.figure(figsize=(12, 8))
    
    # Create two subplots with different y-axis scales
    gs = plt.GridSpec(2, 1, height_ratios=[1, 3])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # Plot first epoch separately with different scale
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot([1], [history['train_loss'][0]], 'b-', label='Training Loss')
    ax1.plot([1], [history['val_loss'][0]], 'r-', label='Validation Loss')
    ax1.set_ylabel('Loss (First Epoch)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot remaining epochs with appropriate scale
    ax2.plot(epochs[1:], history['train_loss'][1:], 'b-', label='Training Loss')
    ax2.plot(epochs[1:], history['val_loss'][1:], 'r-', label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Add a break in the y-axis
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
    
    # Add the break mark
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Save raw data
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

def split_coco_annotations(annotation_file: str, 
                         chunk_size: int = 1000,
                         output_dir: str = "data/chunks") -> List[str]:
    """
    Split a COCO annotation file into smaller chunks, each with its own directory.
    
    Args:
        annotation_file: Path to the original COCO annotation file
        chunk_size: Number of images per chunk
        output_dir: Base directory to save the chunks
        
    Returns:
        List of paths to the created chunk directories
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original annotation file
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create a mapping of image_id to annotations
    image_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    # Calculate number of chunks needed
    num_images = len(coco_data['images'])
    num_chunks = ceil(num_images / chunk_size)
    
    chunk_dirs = []
    
    # Create chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, num_images)
        
        # Create chunk directory
        chunk_dir = os.path.join(output_dir, f'chunk_{chunk_idx + 1:03d}')
        chunk_images_dir = os.path.join(chunk_dir, 'images')
        os.makedirs(chunk_images_dir, exist_ok=True)
        
        # Create a new COCO structure for this chunk
        chunk_data = {
            'categories': deepcopy(coco_data['categories']),
            'images': [],
            'annotations': []
        }
        
        # Add optional fields if they exist
        if 'info' in coco_data:
            chunk_data['info'] = deepcopy(coco_data['info'])
            # Add chunk number to info if it exists
            if 'description' in chunk_data['info']:
                chunk_data['info']['description'] = f"{chunk_data['info']['description']} - Chunk {chunk_idx + 1}/{num_chunks}"
            else:
                chunk_data['info']['description'] = f"Chunk {chunk_idx + 1}/{num_chunks}"
        else:
            chunk_data['info'] = {
                'description': f"Chunk {chunk_idx + 1}/{num_chunks}",
                'date_created': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        if 'licenses' in coco_data:
            chunk_data['licenses'] = deepcopy(coco_data['licenses'])
        
        # Add images for this chunk
        chunk_images = coco_data['images'][start_idx:end_idx]
        chunk_data['images'] = chunk_images
        
        # Add corresponding annotations
        for img in chunk_images:
            img_id = img['id']
            if img_id in image_to_anns:
                chunk_data['annotations'].extend(image_to_anns[img_id])
        
        # Save chunk annotations
        chunk_annotations_file = os.path.join(chunk_dir, 'annotations.json')
        with open(chunk_annotations_file, 'w') as f:
            json.dump(chunk_data, f, indent=2)
        
        # Copy images to chunk directory
        for img in chunk_images:
            src_path = os.path.join('data/images', img['file_name'])
            dst_path = os.path.join(chunk_images_dir, img['file_name'])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
        
        chunk_dirs.append(chunk_dir)
        
        print(f"Created chunk {chunk_idx + 1}/{num_chunks} in {chunk_dir}")
        print(f"  - {len(chunk_images)} images")
        print(f"  - {len(chunk_data['annotations'])} annotations")
    
    return chunk_dirs 