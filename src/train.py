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

from utils import load_config, create_directories, collate_fn, AverageMeter, save_checkpoint
from dataset import DetectionDataset, get_transform
from model import create_model

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        if allocated > 1000:  # Only print if using significant memory
            print(f"GPU memory: {allocated:.0f}MB allocated, {cached:.0f}MB cached")

def train_one_epoch(model, optimizer, data_loader, device, scaler=None):
    model.train()
    loss_hist = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Progress bar
    pbar = tqdm(data_loader, total=len(data_loader))
    
    # Create CUDA events for timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Record the start time for the first iteration
    start.record()
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Record the end of data loading
        end.record()
        torch.cuda.synchronize()  # Make sure the events are synchronized
        
        if batch_idx > 0:  # Skip first batch timing since we don't have a valid start time
            data_time.update(start.elapsed_time(end) / 1000.0)
        
        try:
            # Move data to device
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Record the start of computation
            start.record()
            torch.cuda.synchronize()
            
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
            
            # Record the end of computation
            end.record()
            torch.cuda.synchronize()  # Make sure the events are synchronized
            
            # Update timing metrics
            batch_time.update(end.elapsed_time(start) / 1000.0)
            
            # Update progress bar
            pbar.set_description(
                f'Loss: {loss_hist.avg:.4f} | '
                f'Time: {batch_time.avg:.3f}s | '
                f'Data: {data_time.avg:.3f}s | '
                f'it/s: {1.0/batch_time.avg:.2f}'
            )
            
            # Clear some memory
            del images, targets, loss_dict, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Start timing next iteration's data loading
            start.record()
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            raise e
    
    metrics = {
        'loss': loss_hist.avg,
        'time_per_batch': batch_time.avg,
        'data_time': data_time.avg,
        'iterations_per_second': 1.0/batch_time.avg
    }
    
    return metrics

def validate(model, data_loader, device):
    """Validate the model by computing losses on the validation set."""
    model.train()  # We need training mode to get losses
    loss_hist = AverageMeter()
    
    with torch.no_grad(), autocast('cuda'):
        for images, targets in tqdm(data_loader, desc='Validating'):
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            
            # Get losses (model must be in train mode)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_hist.update(losses.item())
            
            # Clear some memory
            del images, targets, loss_dict, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return loss_hist.avg

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

def main():
    # Check for quick test mode
    QUICK_TEST = os.environ.get('QUICK_TEST', '').lower() == 'true'
    
    try:
        # Load configuration
        config = load_config('config.yaml')
        
        if QUICK_TEST:
            print("Running in QUICK TEST mode!")
            print("Using 10 images, 5 training iterations, 2 validation iterations")
        
        # Create necessary directories
        create_directories(config)
        
        # Set CUDA memory management configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'epochs_without_improvement': 0
        }
        
        # Set device and optimize CUDA settings
        device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            print(f'Using device: {device}')
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            
            # CUDA optimizations
            if config['training'].get('allow_tf32', False):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Set memory allocator settings
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.memory_stats(device=device)
        
        # Create transforms and datasets
        train_transform = get_transform(config, train=True)
        val_transform = get_transform(config, train=False)
        
        # Create dataset
        full_dataset = DetectionDataset(
            config['data']['train_path'],
            config['data']['train_annotations'],
            transforms=train_transform
        )
        
        # Create subset with first N images
        num_images = 10 if QUICK_TEST else 5000
        dataset = Subset(full_dataset, range(num_images))
        print(f"Using first {num_images} images out of {len(full_dataset)} total images")
        
        # Split dataset
        val_size = int(len(dataset) * config['data']['val_split'])
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Update val_dataset transform
        val_dataset.dataset.dataset.transforms = val_transform  # Note: double dataset due to Subset + random_split
        
        # DataLoader settings
        loader_kwargs = {
            'batch_size': config['training']['batch_size'],
            'num_workers': config['training']['num_workers'],
            'collate_fn': collate_fn,
            'pin_memory': config['training']['pin_memory'],
            'prefetch_factor': config['training']['prefetch_factor'],
            'persistent_workers': config['training']['persistent_workers']
        }
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        
        # Create model
        model = create_model(config)
        
        # Convert model to channels last format if configured
        if config['training'].get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        
        # Move model to device
        model = model.to(device)
        
        # Create optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize gradient scaler for mixed precision training
        if config['training'].get('mixed_precision', False) and device.type == 'cuda':
            scaler = GradScaler()
        else:
            scaler = None
        
        # Training loop
        best_val_loss = float('inf')
        min_epochs = 1 if QUICK_TEST else config['training'].get('min_epochs', 20)
        patience = 1 if QUICK_TEST else config['training'].get('patience', 5)
        num_epochs = 1 if QUICK_TEST else config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Train one epoch
            if QUICK_TEST:
                # Only run 5 iterations for quick test
                losses = None
                for i, (images, targets) in enumerate(train_loader):
                    if i >= 5:  # Stop after 5 iterations
                        break
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()
                    print(f'Quick test iteration {i+1}/5, loss: {losses.item():.4f}')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if losses is None:
                    raise RuntimeError("No training iterations completed")
                train_metrics = {'loss': losses.item()}
            else:
                train_metrics = train_one_epoch(model, optimizer, train_loader, device, scaler)
            
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            if QUICK_TEST:
                # Only run 2 iterations for validation in quick test
                val_losses = []
                for i, (images, targets) in enumerate(val_loader):
                    if i >= 2:  # Stop after 2 iterations
                        break
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with torch.no_grad():
                        loss_dict = model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        val_losses.append(losses.item())
                    print(f'Quick test validation iteration {i+1}/2, loss: {losses.item():.4f}')
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if not val_losses:
                    raise RuntimeError("No validation iterations completed")
                val_loss = sum(val_losses) / len(val_losses)
            else:
                val_loss = validate(model, val_loader, device)
            
            history['val_loss'].append(val_loss)
            
            # Store current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            print(f'Train Loss: {train_metrics["loss"]:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Plot and save training curves after each epoch
            plot_training_curves(history, config['training']['output_dir'])
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history['best_val_loss'] = val_loss
                history['epochs_without_improvement'] = 0
                save_checkpoint(model, optimizer, epoch + 1, config, os.path.join(config['training']['checkpoint_dir'], 'latest_model.pth'))
                
                # Save best model
                if val_loss < history['best_val_loss']:
                    history['best_val_loss'] = val_loss
                    history['epochs_without_improvement'] = 0
                    save_checkpoint(model, optimizer, epoch + 1, config, os.path.join(config['training']['checkpoint_dir'], 'best_model.pth'))
            else:
                history['epochs_without_improvement'] += 1
            
            # Early stopping check, but only after minimum epochs
            if epoch + 1 >= min_epochs and history['epochs_without_improvement'] >= patience:
                print(f'\nEarly stopping triggered. No improvement for {patience} epochs.')
                break
        
        # Plot training curves
        plot_training_curves(history, config['training']['output_dir'])
        
        print('\nTraining completed!')
        print(f'Best validation loss: {best_val_loss:.4f}')
        
    finally:
        # Cleanup
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        if 'model' in locals():
            model.cpu()
            del model
        if 'optimizer' in locals():
            del optimizer
        torch.cuda.empty_cache()
        
        # Force close the workers
        if torch.multiprocessing.current_process().daemon:
            os._exit(0)
        else:
            sys.exit(0)

if __name__ == '__main__':
    main() 
