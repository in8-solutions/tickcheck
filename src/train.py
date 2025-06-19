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

from utils import load_config, create_directories, collate_fn, AverageMeter, save_checkpoint
from dataset import DetectionDataset, get_transform
from model import create_model
from transforms import get_transform
from dataset import MultiChunkDataset

def print_gpu_memory():
    """Print current GPU memory usage if significant memory is being used.
    
    This function helps monitor GPU memory consumption during training,
    printing both allocated and cached memory in megabytes.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        if allocated > 1000:  # Only print if using significant memory
            print(f"GPU memory: {allocated:.0f}MB allocated, {cached:.0f}MB cached")

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
    
    # Progress bar
    pbar = tqdm(data_loader, total=len(data_loader))
    
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
            
            # Update progress bar
            it_per_sec = 1.0 / batch_time.avg if batch_time.avg > 0 else 0
            pbar.set_description(
                f'Loss: {loss_hist.avg:.4f} (cls: {cls_loss_hist.avg:.4f}, reg: {reg_loss_hist.avg:.4f}) | '
                f'Time: {batch_time.avg:.3f}s | '
                f'Data: {data_time.avg:.3f}s | '
                f'it/s: {it_per_sec:.2f}'
            )
            
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

def validate(model, data_loader, device):
    """Validate the model on the validation dataset.
    
    This function:
    1. Computes model loss on validation data
    2. Uses no_grad for memory efficiency
    3. Implements mixed precision inference
    4. Handles memory cleanup after validation
    
    Args:
        model (nn.Module): The model to validate
        data_loader (DataLoader): Validation data loader
        device (torch.device): Device to validate on
    
    Returns:
        float: Average validation loss
    """
    """Validate the model by computing losses on the validation set."""
    model.train()  # We need training mode to get losses
    loss_hist = AverageMeter()
    cls_loss_hist = AverageMeter()  # Track classification loss
    reg_loss_hist = AverageMeter()  # Track regression loss
    
    with torch.no_grad(), autocast('cuda'):
        for images, targets in tqdm(data_loader, desc='Validating'):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Get losses (model must be in train mode)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Update metrics
            loss_hist.update(losses.item())
            cls_loss_hist.update(loss_dict['classification'].item())
            reg_loss_hist.update(loss_dict['bbox_regression'].item())
            
            # Clear some memory
            del images, targets, loss_dict, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print(f"\nValidation Loss: {loss_hist.avg:.4f}")
    print(f"Classification Loss: {cls_loss_hist.avg:.4f}")
    print(f"Regression Loss: {reg_loss_hist.avg:.4f}")
    
    return loss_hist.avg

def plot_training_curves(history, output_dir):
    """Plot and save training progress visualization.
    
    Creates and saves plots showing:
    1. Training and validation loss over time
    2. Learning rate schedule
    3. Saves raw data for future analysis
    
    Args:
        history (dict): Training history containing:
            - train_loss: List of training losses
            - val_loss: List of validation losses
            - learning_rates: List of learning rates
        output_dir (str): Directory to save plots and data
    """
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
            'epochs': len(history['train_loss']),
            'best_val_loss': min(history['val_loss']),
            'best_epoch': history['val_loss'].index(min(history['val_loss'])) + 1
        }, f, indent=4)

def load_chunk_data(chunk_dir: str, val_ratio: float = 0.2, random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Load and split data from a chunk directory into training and validation sets.
    
    Args:
        chunk_dir: Path to chunk directory containing images and annotations.json
        val_ratio: Ratio of images to use for validation
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_images, val_images, train_anns, val_anns)
    """
    # Set random seed
    random.seed(random_seed)
    
    # Load annotations
    with open(os.path.join(chunk_dir, 'annotations.json'), 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to annotations mapping
    image_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)
    
    # Calculate number of validation images
    num_images = len(coco_data['images'])
    num_val = int(num_images * val_ratio)
    
    # Randomly select validation images
    val_indices = set(random.sample(range(num_images), num_val))
    
    # Split images and annotations
    train_images = []
    val_images = []
    train_anns = []
    val_anns = []
    
    for i, img in enumerate(coco_data['images']):
        if i in val_indices:
            val_images.append(img)
            if img['id'] in image_to_anns:
                val_anns.extend(image_to_anns[img['id']])
        else:
            train_images.append(img)
            if img['id'] in image_to_anns:
                train_anns.extend(image_to_anns[img['id']])
    
    return train_images, val_images, train_anns, val_anns

def get_chunk_dirs(data_dir: str = "data") -> List[str]:
    """
    Get list of chunk directories in the data directory.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        List of chunk directory paths
    """
    chunk_dirs = []
    for item in os.listdir(data_dir):
        if item.startswith('chunk_'):
            chunk_path = os.path.join(data_dir, item)
            if os.path.isdir(chunk_path):
                chunk_dirs.append(chunk_path)
    return sorted(chunk_dirs)

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
        print("\nQuick test mode - using subset of multiple chunks:")
        # Use first 3 chunks but only first 20 images from each
        test_chunks = config['data']['train_paths'][:3]
        test_anns = config['data']['train_annotations'][:3]
        
        for i, (img_dir, ann_file) in enumerate(zip(test_chunks, test_anns)):
            print(f"Chunk {i+1}: {img_dir}")
        
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
        print("\nTraining with all specified chunks:")
        for i, (img_dir, ann_file) in enumerate(zip(config['data']['train_paths'], config['data']['train_annotations'])):
            print(f"Chunk {i+1}: {img_dir}")
        
        train_dataset = MultiChunkDataset(
            image_dirs=config['data']['train_paths'],
            annotation_files=config['data']['train_annotations'],
            transform=train_transform
        )
        
        # Create validation dataset from a portion of the training data
        val_size = int(len(train_dataset) * config['data']['val_split'])
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"\nDataset sizes:")
    print(f"Total images: {len(train_dataset) + len(val_dataset)}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Batches per epoch: {len(train_dataset) // config['training']['batch_size']}")
    
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
        'learning_rates': []
    }
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, scaler)
        history['train_loss'].append(train_metrics['loss'])
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validate
        val_loss = validate(model, val_loader, device)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        if config['training']['lr_scheduler']['name'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        
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
