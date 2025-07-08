"""
Training script for TensorFlow tick detection model.

This module implements the training loop and validation process for the tick detection
model using TensorFlow/Keras. It handles:
- Data loading and batching
- Training and validation loops
- Mixed precision training
- Checkpointing
- Training curve visualization
- Memory optimization
"""

import os
import math
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
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

from utils import load_config, create_directories, AverageMeter, save_checkpoint, plot_training_curves, load_checkpoint
from dataset import create_train_val_datasets
from model import create_model
from transforms import get_transform

def setup_device(device_config):
    """Setup TensorFlow device based on configuration."""
    device = device_config.lower()
    
    if device == 'mps':
        # Check if MPS is available (macOS 12.3+ with Apple Silicon)
        if tf.config.list_physical_devices('GPU'):
            print("Using MPS (Metal Performance Shaders) for Apple Silicon")
            # Enable memory growth for MPS
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print("MPS memory growth enabled")
                except RuntimeError as e:
                    print(f"Memory growth setting failed: {e}")
            return '/GPU:0'
        else:
            print("MPS not available, falling back to CPU")
            return '/CPU:0'
    elif device == 'cuda':
        if tf.config.list_physical_devices('GPU'):
            print("Using CUDA GPU")
            return '/GPU:0'
        else:
            print("CUDA not available, falling back to CPU")
            return '/CPU:0'
    else:
        print("Using CPU")
        return '/CPU:0'

def train_one_epoch(model, optimizer, data_loader, device, mixed_precision=False):
    """Train the model for one epoch with performance monitoring."""
    loss_hist = AverageMeter()
    cls_loss_hist = AverageMeter()
    reg_loss_hist = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Progress bar - handle unknown dataset length
    try:
        total_batches = len(data_loader)
        pbar = tqdm(data_loader, total=total_batches, leave=False)
    except TypeError:
        # Dataset length is unknown (e.g., with padded_batch)
        pbar = tqdm(data_loader, leave=False)
    
    # Initialize timing
    end = time.time()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        try:
            # Convert to TensorFlow tensors
            if isinstance(images, list):
                images = tf.stack(images)
            
            # Move data to device
            with tf.device(device):
                images = tf.cast(images, tf.float32)
                
                # Use batched targets directly
                batch_boxes = targets['boxes']
                batch_labels = targets['labels']
                targets_tf = {
                    'boxes': batch_boxes,
                    'labels': batch_labels
                }
            
            # Measure forward/backward time
            batch_start = time.time()
            
            # Forward pass with mixed precision
            if mixed_precision:
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    losses = model.compute_loss(targets_tf, predictions)
                    total_loss = losses['total_loss']
                # Scale loss for mixed precision if needed (optional, depending on optimizer)
                if hasattr(optimizer, 'get_scaled_loss'):
                    scaled_loss = optimizer.get_scaled_loss(total_loss)
                    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
                    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
                else:
                    gradients = tape.gradient(total_loss, model.trainable_variables)
                # Gradient clipping
                gradients, _ = tf.clip_by_global_norm(gradients, 20.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            else:
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    losses = model.compute_loss(targets_tf, predictions)
                    total_loss = losses['total_loss']
                # Backward pass
                gradients = tape.gradient(total_loss, model.trainable_variables)
                # Gradient clipping
                gradients, _ = tf.clip_by_global_norm(gradients, 20.0)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Update metrics
            loss_hist.update(total_loss.numpy())
            cls_loss_hist.update(losses['classification_loss'].numpy())
            reg_loss_hist.update(losses['regression_loss'].numpy())
            
            # Measure batch time
            batch_time.update(time.time() - batch_start)
            
            # Update progress bar
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

def validate(model, data_loader, device):
    """Validate the model."""
    loss_hist = AverageMeter()
    cls_loss_hist = AverageMeter()
    reg_loss_hist = AverageMeter()
    
    for images, targets in tqdm(data_loader, desc='Validating', leave=False):
        try:
            # Convert to TensorFlow tensors
            if isinstance(images, list):
                images = tf.stack(images)
            
            with tf.device(device):
                images = tf.cast(images, tf.float32)
                
                # Use batched targets directly
                batch_boxes = targets['boxes']
                batch_labels = targets['labels']
                targets_tf = {
                    'boxes': batch_boxes,
                    'labels': batch_labels
                }
                
                # Forward pass
                predictions = model(images, training=False)
                losses = model.compute_loss(targets_tf, predictions)
                
                # Update metrics
                loss_hist.update(losses['total_loss'].numpy())
                cls_loss_hist.update(losses['classification_loss'].numpy())
                reg_loss_hist.update(losses['regression_loss'].numpy())
                
        except Exception as e:
            print(f"Error in validation batch: {str(e)}")
            continue
    
    metrics = {
        'loss': loss_hist.avg,
        'cls_loss': cls_loss_hist.avg,
        'reg_loss': reg_loss_hist.avg
    }
    
    return metrics

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TensorFlow tick detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with small dataset')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Set up directories
        create_directories([
            config['output']['checkpoint_dir'],
            config['output']['training_curves'],
            config['output']['output_dir']
        ])
        
        # Setup device
        device = setup_device(config['training']['device'])
        print(f"Using device: {device}")
        
        # Set mixed precision
        if config['training']['use_amp']:
            print("Enabling mixed precision training")
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Create datasets
        print("Creating datasets...")
        train_dataset, val_dataset = create_train_val_datasets(config, args.quick_test)
        
        # Create model
        print("Creating model...")
        model = create_model(config)
        
        # Create optimizer - use legacy optimizer for Mac to avoid slow performance
        if config['training']['device'].lower() in ['mps', 'cpu']:
            # Use legacy optimizer for Mac
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=config['training']['learning_rate'],
                decay=config['training']['weight_decay']
            )
        else:
            # Use regular optimizer for CUDA
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            min_lr=config['training']['lr_scheduler']['min_lr'],
            verbose=True
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        best_loss = float('inf')
        if args.resume:
            print(f"Resuming from checkpoint: {args.resume}")
            start_epoch, best_loss = load_checkpoint(model, optimizer, args.resume)
        
        # Training history
        history = {
            'loss': [],
            'val_loss': [],
            'classification_loss': [],
            'val_classification_loss': [],
            'regression_loss': [],
            'val_regression_loss': [],
            'lr': []
        }
        
        # Training loop
        print("Starting training...")
        
        # Adjust training parameters for quick test
        if args.quick_test:
            print("Quick test mode: Reducing epochs and batch size")
            num_epochs = min(5, config['training']['num_epochs'])  # Max 5 epochs for quick test
            min_epochs = min(2, config['training']['min_epochs'])  # Max 2 min epochs
        else:
            num_epochs = config['training']['num_epochs']
            min_epochs = config['training']['min_epochs']
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = train_one_epoch(
                model, optimizer, train_dataset, device, 
                mixed_precision=config['training']['use_amp']
            )
            
            # Validate
            val_metrics = validate(model, val_dataset, device)
            
            # Calculate epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Update learning rate
            current_lr = optimizer.learning_rate.numpy()
            
            # Save history
            history['loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['classification_loss'].append(train_metrics['cls_loss'])
            history['val_classification_loss'].append(val_metrics['cls_loss'])
            history['regression_loss'].append(train_metrics['reg_loss'])
            history['val_regression_loss'].append(val_metrics['reg_loss'])
            history['lr'].append(current_lr)
            
            # Print metrics with timing
            print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            print(f"Train CLS: {train_metrics['cls_loss']:.4f}, Val CLS: {val_metrics['cls_loss']:.4f}")
            print(f"Train REG: {train_metrics['reg_loss']:.4f}, Val REG: {val_metrics['reg_loss']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Epoch Time: {epoch_time:.1f}s")
            
            # Estimate data capacity for laptop
            if args.quick_test:
                samples_per_epoch = 100  # Updated for doubled quick test size
                time_per_sample = epoch_time / samples_per_epoch
                print(f"Time per sample: {time_per_sample:.2f}s")
                print(f"Estimated samples per hour: {3600/time_per_sample:.0f}")
                print(f"Estimated samples per 20min: {(1200/time_per_sample):.0f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                config['output']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.h5'
            )
            save_checkpoint(model, optimizer, epoch + 1, val_metrics['loss'], config, checkpoint_path)
            
            # Save best model
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_checkpoint_path = os.path.join(
                    config['output']['checkpoint_dir'],
                    'best_model.h5'
                )
                save_checkpoint(model, optimizer, epoch + 1, best_loss, config, best_checkpoint_path)
                print(f"New best model saved with loss: {best_loss:.4f}")
            
            # Plot training curves
            if (epoch + 1) % 5 == 0 or args.quick_test:  # Plot more frequently in quick test
                plot_training_curves(history, config['output']['training_curves'])
            
            # Early stopping check
            if epoch >= min_epochs:
                # Check if validation loss has been increasing
                recent_losses = history['val_loss'][-config['training']['early_stopping_patience']:]
                if len(recent_losses) >= config['training']['early_stopping_patience']:
                    if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
        
        # Final plot
        plot_training_curves(history, config['output']['training_curves'])
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 