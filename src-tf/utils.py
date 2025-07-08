"""
Utility functions for TensorFlow tick detection model.

This module provides utility functions for:
- Configuration loading and management
- Directory creation and file management
- Metric tracking and averaging
- Checkpoint saving and loading
- General helper functions
"""

import os
import yaml
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_directories(dirs: List[str]) -> None:
    """Create directories if they don't exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

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

def to_python_type(obj):
    """Recursively convert numpy/tf scalars and arrays to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, tf.Tensor):
        arr = obj.numpy()
        if hasattr(arr, 'size') and arr.size == 1:
            return arr.item() if hasattr(arr, 'item') else arr
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        return arr
    else:
        return obj

def save_checkpoint(model, optimizer, epoch, loss, config, filepath: str) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model weights
    model.save_weights(filepath)
    
    # Save training state
    checkpoint_info = {
        'epoch': int(epoch),  # Convert to Python int
        'loss': float(loss),  # Convert to Python float
        'optimizer_state': optimizer.get_weights(),
        'config': config
    }
    checkpoint_info = to_python_type(checkpoint_info)
    
    checkpoint_path = filepath.replace('.h5', '_info.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_info, f, indent=2)

def load_checkpoint(model, optimizer, filepath: str) -> Tuple[int, float]:
    """Load model checkpoint."""
    # Load model weights
    model.load_weights(filepath)
    
    # Load training state
    checkpoint_path = filepath.replace('.h5', '_info.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint_info = json.load(f)
        
        # Restore optimizer state
        optimizer.set_weights(checkpoint_info['optimizer_state'])
        
        return checkpoint_info['epoch'], checkpoint_info['loss']
    
    return 0, float('inf')

def collate_fn(batch):
    """Custom collate function for TensorFlow dataset."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    return images, targets

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes."""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Apply non-maximum suppression to bounding boxes."""
    if len(boxes) == 0:
        return [], []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by scores in descending order
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        ious = []
        for box in remaining_boxes:
            iou = calculate_iou(current_box, box)
            ious.append(iou)
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][np.array(ious) < iou_threshold]
    
    return boxes[keep], scores[keep]

def plot_training_curves(history, output_dir: str) -> None:
    """Plot training curves and save to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16)
    
    # Loss curves
    axes[0, 0].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Classification loss
    if 'classification_loss' in history:
        axes[0, 1].plot(history['classification_loss'], label='Training Classification Loss')
        if 'val_classification_loss' in history:
            axes[0, 1].plot(history['val_classification_loss'], label='Validation Classification Loss')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Regression loss
    if 'regression_loss' in history:
        axes[1, 0].plot(history['regression_loss'], label='Training Regression Loss')
        if 'val_regression_loss' in history:
            axes[1, 0].plot(history['val_regression_loss'], label='Validation Regression Loss')
        axes[1, 0].set_title('Regression Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, and F1 score for object detection."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_boxes, pred_scores, gt_boxes in zip(predictions['boxes'], predictions['scores'], ground_truth['boxes']):
        # Apply NMS to predictions
        if len(pred_boxes) > 0:
            nms_boxes, nms_scores = non_max_suppression(pred_boxes, pred_scores)
        else:
            nms_boxes, nms_scores = [], []
        
        # Count ground truth boxes
        num_gt = len(gt_boxes)
        false_negatives += num_gt
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred_box, pred_score in zip(nms_boxes, nms_scores):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
                true_positives += 1
                false_negatives -= 1
                matched_gt.add(best_gt_idx)
            else:
                false_positives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    } 