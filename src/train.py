"""
Training script for tick detection model.

This module implements the training loop and validation process for the tick detection
model. It handles:
- Data loading and batching
- Training and validation loops
- Mixed precision training
- Checkpointing
- Training curve visualization
- Memory optimization

Key Features:
- Automatic mixed precision (AMP) training
- GPU memory optimization
- Training progress visualization
- Configurable training parameters
- Support for quick testing mode
"""

# Set matplotlib backend to non-interactive to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import math
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import torch._dynamo
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import traceback
import random
from typing import List, Dict, Tuple
import argparse
import shutil
import time
import gc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils import load_config, create_directories, collate_fn, AverageMeter, save_checkpoint
from dataset import DetectionDataset, get_transform
from model import create_model
from transforms import get_transform
from dataset import MultiChunkDataset

def train_one_epoch(model, optimizer, data_loader, device, scaler=None):
    """Train the model for one epoch with performance monitoring."""
    model.train()
    loss_hist = AverageMeter()
    cls_loss_hist = AverageMeter()  # Track classification loss
    reg_loss_hist = AverageMeter()  # Track regression loss
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Progress bar - simplified description
    pbar = tqdm(data_loader, total=len(data_loader), leave=False)
    
    # Initialize timing
    end = time.time()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        try:
            # Move data to device
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Measure forward/backward time
            batch_start = time.time()
            
            # Forward pass with mixed precision where available
            if scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass with gradient scaling
                scaler.scale(losses).backward()
                
                # Add gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
                
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            loss_hist.update(losses.item())
            # RetinaNet uses these keys for its losses
            cls_loss_hist.update(loss_dict['classification'].item())
            reg_loss_hist.update(loss_dict['bbox_regression'].item())
            
            # Measure batch time
            batch_time.update(time.time() - batch_start)
            
            # Update progress bar - simplified
            pbar.set_description(f'Loss: {loss_hist.avg:.4f}')
            
            # Update end time for next iteration
            end = time.time()
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            raise e
    
    metrics = {
        'loss': loss_hist.avg,
        'cls_loss': cls_loss_hist.avg,
        'reg_loss': reg_loss_hist.avg,
        'time_per_batch': batch_time.avg,
        'data_time': data_time.avg,
        'iterations_per_second': 1.0/batch_time.avg if batch_time.avg > 0 else 0
    }
    
    return metrics

def eval_forward_retinanet(model, images, targets):
    """
    Manual forward pass for RetinaNet in eval mode that computes both losses and predictions.
    This allows us to get proper eval mode behavior while still computing losses for training decisions.
    
    Args:
        model: RetinaNet model
        images: List of image tensors
        targets: List of target dictionaries
    
    Returns:
        tuple: (losses_dict, predictions_list)
    """
    model.eval()
    
    with torch.no_grad():
        # Get predictions (this is what eval mode normally does)
        predictions = model(images)
        
        # For now, we'll use a simpler approach - just get the predictions
        # and compute a basic loss approximation
        # The full manual loss computation requires deeper access to RetinaNet internals
        
        # Create a simple loss approximation based on predictions vs targets
        total_loss = 0.0
        cls_loss = 0.0
        reg_loss = 0.0
        
        for pred, target in zip(predictions, targets):
            # Simple loss approximation: penalize based on detection count mismatch
            pred_count = len(pred['boxes'])
            target_count = len(target['boxes'])
            
            # Classification loss: penalize wrong number of detections
            cls_loss += abs(pred_count - target_count) * 0.1
            
            # Regression loss: if we have detections, penalize based on confidence
            if pred_count > 0:
                avg_confidence = pred['scores'].mean()
                reg_loss += (1.0 - avg_confidence) * 0.1
            
            total_loss += cls_loss + reg_loss
        
        # Normalize by number of images
        num_images = len(images)
        losses = {
            'loss_classifier': torch.full((1,), cls_loss / num_images, device=images[0].device, dtype=torch.float32),
            'loss_box_reg': torch.full((1,), reg_loss / num_images, device=images[0].device, dtype=torch.float32)
        }
        
        return losses, predictions

def validate(model, data_loader, device):
    """Validate the model using eval mode with manual loss computation."""
    loss_hist = AverageMeter()
    cls_loss_hist = AverageMeter()  # Track classification loss
    reg_loss_hist = AverageMeter()  # Track regression loss
    
    with torch.no_grad(), autocast('cuda'):
        for images, targets in tqdm(data_loader, desc='Validating', leave=False):
            images = [image.to(device, non_blocking=True) for image in images]
            
            # Filter out non-dictionary targets and move valid ones to device
            valid_targets = []
            for t in targets:
                if isinstance(t, dict):
                    # Move tensors to device, leave other values as-is
                    valid_target = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in t.items()}
                    valid_targets.append(valid_target)
                else:
                    print(f"Warning: Skipping non-dictionary target: {type(t)} - {t}")
            
            if not valid_targets:
                print("Warning: No valid targets in batch, skipping...")
                continue
            
            try:
                # Use eval_forward to get losses in eval mode
                loss_dict, predictions = eval_forward_retinanet(model, images, valid_targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Update metrics
                loss_hist.update(losses.item())
                cls_loss_hist.update(loss_dict['loss_classifier'].item())
                reg_loss_hist.update(loss_dict['loss_box_reg'].item())
                
            except Exception as e:
                print(f"Error in eval_forward: {str(e)}")
                raise e  # Re-raise the exception to fail fast and fix the issue
            
            # Clear some memory
            del images, targets, valid_targets, loss_dict, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return loss_hist.avg

def calculate_image_level_metrics(predictions, ground_truth):
    """
    Calculate metrics treating each image as a binary classification problem.
    
    Args:
        predictions: List of prediction dicts from model
        ground_truth: List of ground truth dicts
    
    Returns:
        dict: Image-level metrics
    """
    image_predictions = []
    image_ground_truth = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Image has tick if any detection above threshold
        has_tick_pred = len(pred['boxes']) > 0
        has_tick_gt = len(gt['boxes']) > 0
        
        image_predictions.append(has_tick_pred)
        image_ground_truth.append(has_tick_gt)
    
    # Calculate binary classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        image_ground_truth, image_predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(image_ground_truth, image_predictions)
    
    return {
        'image_precision': precision,
        'image_recall': recall,
        'image_f1': f1,
        'image_accuracy': accuracy
    }

def calculate_center_distance_metrics(predictions, ground_truth, max_distance=50):
    """
    Calculate metrics based on center point distances rather than IoU.
    More suitable for small objects like ticks.
    """
    def get_center(box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    
    def calculate_distance(center1, center2):
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, gt in zip(predictions, ground_truth):
        pred_centers = [get_center(box) for box in pred['boxes']]
        gt_centers = [get_center(box) for box in gt['boxes']]
        
        # Match predictions to ground truth based on distance
        matched_gt = set()
        for pred_center in pred_centers:
            best_match = None
            best_distance = float('inf')
            
            for i, gt_center in enumerate(gt_centers):
                if i not in matched_gt:
                    distance = calculate_distance(pred_center, gt_center)
                    if distance < best_distance and distance <= max_distance:
                        best_distance = distance
                        best_match = i
            
            if best_match is not None:
                matched_gt.add(best_match)
                true_positives += 1
            else:
                false_positives += 1
        
        false_negatives += len(gt_centers) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'distance_precision': precision,
        'distance_recall': recall,
        'distance_f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def confidence_analysis(predictions, ground_truth, confidence_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Analyze performance across different confidence thresholds."""
    results = {}
    
    for threshold in confidence_thresholds:
        filtered_predictions = []
        for pred in predictions:
            # Filter by confidence threshold
            mask = pred['scores'] >= threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'scores': pred['scores'][mask],
                'labels': pred['labels'][mask]
            }
            filtered_predictions.append(filtered_pred)
        
        # Calculate metrics at this threshold
        metrics = calculate_image_level_metrics(filtered_predictions, ground_truth)
        results[f'threshold_{threshold}'] = metrics
    
    return results

def validate_model(model, data_loader, device, confidence_threshold=0.5):
    """
    Proper validation function that handles eval mode correctly.
    This function calculates detection metrics, not losses.
    """
    model.eval()  # Set to evaluation mode
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Evaluating', leave=False):
            images = [image.to(device, non_blocking=True) for image in images]
            
            # Filter out non-dictionary targets
            valid_targets = []
            for t in targets:
                if isinstance(t, dict):
                    valid_targets.append(t)
                else:
                    print(f"Warning: Skipping non-dictionary target in validate_model: {type(t)} - {t}")
            
            if not valid_targets:
                print("Warning: No valid targets in batch for validate_model, skipping...")
                continue
            
            # Get predictions (not losses) - model must be in eval mode
            predictions = model(images)
            
            # Process predictions for metric calculation
            for pred, target in zip(predictions, valid_targets):
                # Filter by confidence threshold
                mask = pred['scores'] >= confidence_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask].cpu().numpy(),
                    'scores': pred['scores'][mask].cpu().numpy(),
                    'labels': pred['labels'][mask].cpu().numpy()
                }
                
                all_predictions.append(filtered_pred)
                all_ground_truth.append({
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                })
            
            # Clear memory
            del images, targets, valid_targets, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate various metrics
    metrics = {}
    
    # Image-level metrics
    image_metrics = calculate_image_level_metrics(all_predictions, all_ground_truth)
    metrics.update(image_metrics)
    
    # Distance-based metrics
    distance_metrics = calculate_center_distance_metrics(all_predictions, all_ground_truth)
    metrics.update(distance_metrics)
    
    # Confidence analysis
    confidence_metrics = confidence_analysis(all_predictions, all_ground_truth)
    metrics.update(confidence_metrics)
    
    return metrics

def plot_training_curves(history, output_dir):
    """Plot and save training progress visualization.
    
    Creates and saves plots showing:
    1. Training and validation loss over time
    2. Learning rate schedule
    3. Validation metrics over time
    4. Saves raw data for future analysis
    
    Args:
        history (dict): Training history containing:
            - train_loss: List of training losses
            - val_loss: List of validation losses
            - learning_rates: List of learning rates
            - val_metrics: List of validation metrics dictionaries
        output_dir (str): Directory to save plots and data
    """
    """Plot and save training curves."""
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Loss curves
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Learning rate
    plt.subplot(3, 2, 2)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Image-level metrics
    if 'val_metrics' in history and history['val_metrics']:
        plt.subplot(3, 2, 3)
        epochs = range(1, len(history['val_metrics']) + 1)
        image_f1 = [metrics['image_f1'] for metrics in history['val_metrics']]
        image_precision = [metrics['image_precision'] for metrics in history['val_metrics']]
        image_recall = [metrics['image_recall'] for metrics in history['val_metrics']]
        
        plt.plot(epochs, image_f1, label='Image F1', marker='o')
        plt.plot(epochs, image_precision, label='Image Precision', marker='s')
        plt.plot(epochs, image_recall, label='Image Recall', marker='^')
        plt.title('Image-Level Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
    
    # Plot 4: Distance-based metrics
    if 'val_metrics' in history and history['val_metrics']:
        plt.subplot(3, 2, 4)
        epochs = range(1, len(history['val_metrics']) + 1)
        distance_f1 = [metrics['distance_f1'] for metrics in history['val_metrics']]
        distance_precision = [metrics['distance_precision'] for metrics in history['val_metrics']]
        distance_recall = [metrics['distance_recall'] for metrics in history['val_metrics']]
        
        plt.plot(epochs, distance_f1, label='Distance F1', marker='o')
        plt.plot(epochs, distance_precision, label='Distance Precision', marker='s')
        plt.plot(epochs, distance_recall, label='Distance Recall', marker='^')
        plt.title('Distance-Based Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
    
    # Plot 5: Detection counts
    if 'val_metrics' in history and history['val_metrics']:
        plt.subplot(3, 2, 5)
        epochs = range(1, len(history['val_metrics']) + 1)
        true_positives = [metrics['true_positives'] for metrics in history['val_metrics']]
        false_positives = [metrics['false_positives'] for metrics in history['val_metrics']]
        false_negatives = [metrics['false_negatives'] for metrics in history['val_metrics']]
        
        plt.plot(epochs, true_positives, label='True Positives', marker='o', color='green')
        plt.plot(epochs, false_positives, label='False Positives', marker='s', color='red')
        plt.plot(epochs, false_negatives, label='False Negatives', marker='^', color='orange')
        plt.title('Detection Counts Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
    
    # Plot 6: Confidence threshold analysis (from last epoch)
    if 'val_metrics' in history and history['val_metrics']:
        plt.subplot(3, 2, 6)
        last_metrics = history['val_metrics'][-1]
        thresholds = []
        f1_scores = []
        
        for key, value in last_metrics.items():
            if key.startswith('threshold_') and 'image_f1' in value:
                threshold = float(key.split('_')[1])
                thresholds.append(threshold)
                f1_scores.append(value['image_f1'])
        
        if thresholds:
            plt.plot(thresholds, f1_scores, marker='o', linewidth=2, markersize=8)
            plt.title('F1 Score vs Confidence Threshold (Last Epoch)')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Image F1 Score')
            plt.grid(True)
            plt.ylim(0, 1)

    plt.tight_layout()
    
    # Save the plot
    curves_dir = Path(output_dir) / 'training_curves'
    curves_dir.mkdir(exist_ok=True)
    plt.savefig(curves_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save raw data for future reference
    with open(curves_dir / 'training_history.json', 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'learning_rates': history['learning_rates'],
            'val_metrics': history.get('val_metrics', []),
            'epochs': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']),
            'best_epoch': history['val_loss'].index(min(history['val_loss'])) + 1
        }, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Train tick detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test with a small subset of data')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Modify config for quick test
    if args.quick_test:
        config['training']['num_epochs'] = 2  # Reduce epochs for quick test
        config['training']['batch_size'] = 4  # Smaller batch size for quick test
        print("\nRunning in quick test mode:")
        print("- Using subset of multiple chunks")
        print(f"- {config['training']['num_epochs']} epochs")
        print(f"- Batch size: {config['training']['batch_size']}")
    
    # Create output directories
    create_directories(config)
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    if config['training']['lr_scheduler']['name'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            min_lr=config['training']['lr_scheduler']['min_lr']
        )
    
    # Create transforms
    train_transform = get_transform(config, train=True)
    val_transform = get_transform(config, train=False)
    
    # Create datasets
    if args.quick_test:
        # Use a small subset of multiple chunks for quick testing
        print("Quick test mode - using subset of multiple chunks")
        # Use first 3 chunks but only first 20 images from each
        test_chunks = config['data']['train_paths'][:3]
        test_anns = config['data']['train_annotations'][:3]
        
        train_dataset = MultiChunkDataset(
            image_dirs=test_chunks,
            annotation_files=test_anns,
            transform=train_transform
        )
        
        # Take only first 20 images from each chunk for quick testing
        total_test_images = min(20 * len(test_chunks), len(train_dataset))
        train_dataset = Subset(train_dataset, range(total_test_images))
        
        # Split into train/val
        val_size = int(len(train_dataset) * config['data']['val_split'])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        # Use all specified chunks
        print("Training with all specified chunks")
        
        train_dataset = MultiChunkDataset(
            image_dirs=config['data']['train_paths'],
            annotation_files=config['data']['train_annotations'],
            transform=train_transform
        )
        
        # Create validation dataset from a portion of the training data
        val_size = int(len(train_dataset) * config['data']['val_split'])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val images")
    print(f"Batch size: {config['training']['batch_size']} | Batches per epoch: {len(train_dataset) // config['training']['batch_size']}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['training']['prefetch_factor'],
        persistent_workers=config['training']['persistent_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['training']['prefetch_factor'],
        persistent_workers=config['training']['persistent_workers']
    )
    
    # Initialize mixed precision training
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': [],
        'val_metrics': []  # Add validation metrics history
    }
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, scaler)
        history['train_loss'].append(train_metrics['loss'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validate - use both functions for comprehensive evaluation
        val_loss = validate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        
        # Compute evaluation metrics
        val_metrics = validate_model(model, val_loader, device, confidence_threshold=0.5)
        history['val_metrics'].append(val_metrics)
        
        # Print summary for this epoch
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_loss:.4f} | Image F1: {val_metrics['image_f1']:.4f}")
        
        # Update learning rate based on validation loss
        if config['training']['lr_scheduler']['name'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(f"  ✓ New best model! (loss: {val_loss:.4f})")
        
        # Create checkpoint filename
        checkpoint_filename = os.path.join(
            config['output']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch + 1}.pt'
        )
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            config=config,
            filename=checkpoint_filename
        )
        
        # Save best model if needed
        if is_best:
            best_model_filename = os.path.join(
                config['output']['checkpoint_dir'],
                'best_model.pt'
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                config=config,
                filename=best_model_filename
            )
        
        # Plot training curves
        plot_training_curves(history, config['output']['output_dir'])
        
        # Early stopping
        if epoch >= config['training']['min_epochs'] and patience_counter >= config['training']['early_stopping_patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining completed!")
    
    # Cleanup data loaders
    del train_loader
    del val_loader
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    print("Cleanup complete, exiting...")
    sys.exit(0)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 
