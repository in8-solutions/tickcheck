"""
Training script for Mobile Tick Detection Model
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    DATA_DIR, OUTPUT_CONFIG, TRAINING_CONFIG, 
    MODEL_CONFIG, DATA_CONFIG, AUGMENTATION_CONFIG
)
from data_pipeline import create_data_loaders
from model import create_model, get_model_size_mb
from utils import EarlyStopping, save_checkpoint, load_checkpoint

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileTrainer:
    """Training class for mobile tick detection model"""
    
    def __init__(self, config: Dict = None):
        self.config = config or TRAINING_CONFIG
        self.device = torch.device(self.config['device'])
        
        # Initialize components
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.early_stopping = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def setup_model(self):
        """Initialize model, loss function, and optimizer"""
        
        # Create model
        self.model = create_model(MODEL_CONFIG)
        self.model.to(self.device)
        
        # Loss function with class weights to improve recall
        # Weight positive samples (ticks) higher since false positives are acceptable
        class_weights = torch.tensor([1.0, 2.0]).to(self.device)  # [negative, positive]
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler_config = self.config['lr_scheduler']
        if scheduler_config['name'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
        
        # Mixed precision training
        if self.config['use_amp']:
            self.scaler = GradScaler()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            min_epochs=self.config['min_epochs']
        )
        
        logger.info(f"Model size: {get_model_size_mb(self.model):.2f} MB")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            if self.config['use_amp']:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs['logits'], labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config['use_amp']:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Collect metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict:
        """Validate for one epoch"""
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                if self.config['use_amp']:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs['logits'], labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs['logits'], labels)
                
                # Collect results
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=1)
                probabilities = outputs['probabilities']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _calculate_metrics(self, labels: List, predictions: List, probabilities: Optional[List] = None) -> Dict:
        """Calculate classification metrics"""
        
        labels = np.array(labels)
        predictions = np.array(predictions)
        
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        accuracy = accuracy_score(labels, predictions)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }
        
        # ROC-AUC if probabilities available
        if probabilities is not None:
            probabilities = np.array(probabilities)
            if len(probabilities.shape) > 1:
                # Use probability of positive class
                pos_probabilities = probabilities[:, 1]
            else:
                pos_probabilities = probabilities
            
            try:
                auc = roc_auc_score(labels, pos_probabilities)
                metrics['auc'] = auc
            except ValueError:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        
        logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Log results
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch + 1}/{self.config['num_epochs']} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_metrics'].append(train_metrics)
            self.training_history['val_metrics'].append(val_metrics)
            self.training_history['learning_rates'].append(current_lr)
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics['loss'],
                    OUTPUT_CONFIG['checkpoint_dir'] / 'best_model.pth'
                )
                logger.info(f"✓ New best model! (loss: {val_metrics['loss']:.4f})")
            
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics['f1'],
                    OUTPUT_CONFIG['checkpoint_dir'] / 'best_f1_model.pth',
                    metric_name='f1'
                )
                logger.info(f"✓ New best F1 model! (F1: {val_metrics['f1']:.4f})")
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info("Early stopping triggered!")
                break
        
        # Save final model
        save_checkpoint(
            self.model, self.optimizer, epoch, val_metrics['loss'],
            OUTPUT_CONFIG['checkpoint_dir'] / 'final_model.pth'
        )
        
        # Save training history
        self._save_training_history()
        
        # Plot training curves
        self._plot_training_curves()
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time / 3600:.2f} hours")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
    
    def _save_training_history(self):
        """Save training history to JSON"""
        history_file = OUTPUT_CONFIG['curves_dir'] / 'training_history.json'
        
        # Convert numpy arrays to lists for JSON serialization
        history = {}
        for key, value in self.training_history.items():
            if key in ['train_metrics', 'val_metrics']:
                # Convert metrics dictionaries
                history[key] = []
                for metrics in value:
                    metrics_dict = {}
                    for metric_key, metric_value in metrics.items():
                        if isinstance(metric_value, np.ndarray):
                            metrics_dict[metric_key] = metric_value.tolist()
                        else:
                            metrics_dict[metric_key] = metric_value
                    history[key].append(metrics_dict)
            else:
                history[key] = value
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training history saved to {history_file}")
    
    def _plot_training_curves(self):
        """Plot training curves"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # F1 Score
            train_f1 = [m['f1'] for m in self.training_history['train_metrics']]
            val_f1 = [m['f1'] for m in self.training_history['val_metrics']]
            axes[0, 1].plot(train_f1, label='Train F1')
            axes[0, 1].plot(val_f1, label='Val F1')
            axes[0, 1].set_title('Training and Validation F1 Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision and Recall
            val_precision = [m['precision'] for m in self.training_history['val_metrics']]
            val_recall = [m['recall'] for m in self.training_history['val_metrics']]
            axes[1, 0].plot(val_precision, label='Val Precision')
            axes[1, 0].plot(val_recall, label='Val Recall')
            axes[1, 0].set_title('Validation Precision and Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Learning rate
            axes[1, 1].plot(self.training_history['learning_rates'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_CONFIG['curves_dir'] / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to {OUTPUT_CONFIG['curves_dir'] / 'training_curves.png'}")
            
        except Exception as e:
            logger.warning(f"Failed to plot training curves: {e}")


def main():
    """Main training function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Mobile Tick Detection Model')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run in quick test mode with limited data and epochs')
    args = parser.parse_args()
    
    if args.quick_test:
        logger.info("Running in quick test mode:")
        logger.info("- Using limited data (20 train batches, 5 val batches)")
        logger.info("- 2 epochs")
        logger.info("- Batch size: 128")
        
        # Create limited data loaders for quick testing
        train_loader, val_loader = create_data_loaders(
            DATA_DIR,
            batch_size=128,
            num_workers=2
        )
        
        # Limit the number of batches for quick testing
        class LimitedDataLoader:
            def __init__(self, loader, max_batches=10):
                self.loader = loader
                self.max_batches = max_batches
                self.batch_count = 0
            
            def __iter__(self):
                for batch in self.loader:
                    if self.batch_count >= self.max_batches:
                        break
                    self.batch_count += 1
                    yield batch
            
            def __len__(self):
                return min(len(self.loader), self.max_batches)
        
        limited_train_loader = LimitedDataLoader(train_loader, max_batches=20)
        limited_val_loader = LimitedDataLoader(val_loader, max_batches=5)
        
        logger.info(f"Limited train batches: {len(limited_train_loader)}")
        logger.info(f"Limited val batches: {len(limited_val_loader)}")
        
        # Use quick test config
        quick_config = TRAINING_CONFIG.copy()
        quick_config['num_epochs'] = 2
        quick_config['batch_size'] = 128
        quick_config['early_stopping_patience'] = 3
        quick_config['min_epochs'] = 1
        
        # Initialize trainer
        trainer = MobileTrainer(quick_config)
        trainer.setup_model()
        
        # Start training
        trainer.train(limited_train_loader, limited_val_loader)
        
    else:
        # Full training mode
        logger.info("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(
            DATA_DIR,
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=TRAINING_CONFIG['num_workers']
        )
        
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Initialize trainer
        trainer = MobileTrainer(TRAINING_CONFIG)
        trainer.setup_model()
        
        # Start training
        trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main() 