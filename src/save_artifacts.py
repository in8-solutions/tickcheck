#!/usr/bin/env python3
"""
Script to save off training artifacts for archiving.

This script bundles key files from a successful training run:
- Best model checkpoint
- Training curves and history
- Configuration file
- Evaluation results (if available)
- Training logs and metrics

Usage:
    python src/save_artifacts.py --name "training_run_v1" --output "archives/"
"""

import os
import shutil
import json
import argparse
from datetime import datetime
from pathlib import Path
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_best_checkpoint(checkpoint_dir):
    """Find the best checkpoint based on validation loss."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Look for best_model.pth first
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        return best_model_path
    
    # Look for checkpoint files and find the one with best validation loss
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Load training history to find best epoch
    history_path = Path("outputs/training_curves/training_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        best_epoch = history.get('best_epoch', 1)
        best_checkpoint = checkpoint_dir / f"checkpoint_epoch_{best_epoch}.pt"
        if best_checkpoint.exists():
            return best_checkpoint
    
    # Fallback to latest checkpoint
    return max(checkpoint_files, key=lambda x: x.stat().st_mtime)


def create_artifact_bundle(name, output_dir, include_evaluation=True):
    """Create a bundle of training artifacts."""
    config = load_config()
    
    # Create output directory
    output_path = Path(output_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating artifact bundle: {output_path}")
    
    # Files to copy
    files_to_copy = [
        ("config.yaml", "config.yaml"),
        ("outputs/training_curves/training_history.json", "training_history.json"),
        ("outputs/training_curves/training_curves.png", "training_curves.png"),
    ]
    
    # Copy files
    for src, dst in files_to_copy:
        src_path = Path(src)
        dst_path = output_path / dst
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ Copied {src} -> {dst}")
        else:
            print(f"  ✗ Missing {src}")
    
    # Copy best checkpoint
    checkpoint_dir = config['output']['checkpoint_dir']
    best_checkpoint = find_best_checkpoint(checkpoint_dir)
    if best_checkpoint:
        checkpoint_dst = output_path / "best_model.pt"
        shutil.copy2(best_checkpoint, checkpoint_dst)
        print(f"  ✓ Copied {best_checkpoint.name} -> best_model.pt")
    else:
        print(f"  ✗ No checkpoint found in {checkpoint_dir}")
    
    # Copy evaluation results if available
    if include_evaluation:
        eval_dir = Path("outputs/evaluation")
        if eval_dir.exists():
            eval_dst = output_path / "evaluation"
            shutil.copytree(eval_dir, eval_dst, dirs_exist_ok=True)
            print(f"  ✓ Copied evaluation results")
        else:
            print(f"  - No evaluation results found")
    
    # Create metadata file
    metadata = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "config": config,
        "files_included": [dst for _, dst in files_to_copy] + ["best_model.pt"]
    }
    
    if include_evaluation and eval_dir.exists():
        metadata["files_included"].append("evaluation/")
    
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"  ✓ Created metadata.json")
    
    # Create a summary report
    summary_path = output_path / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Training Run Summary: {name}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Training history summary
        history_path = Path("outputs/training_curves/training_history.json")
        if history_path.exists():
            with open(history_path, 'r') as history_file:
                history = json.load(history_file)
            
            f.write("Training Results:\n")
            f.write(f"  Total epochs: {history.get('epochs', 'N/A')}\n")
            f.write(f"  Best validation loss: {history.get('best_val_loss', 'N/A'):.4f}\n")
            f.write(f"  Best epoch: {history.get('best_epoch', 'N/A')}\n")
            f.write(f"  Final training loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"  Final validation loss: {history['val_loss'][-1]:.4f}\n\n")
        
        # Model configuration summary
        f.write("Model Configuration:\n")
        f.write(f"  Backbone: {config['model']['backbone']}\n")
        f.write(f"  Box score threshold: {config['model']['box_score_thresh']}\n")
        f.write(f"  Box NMS threshold: {config['model']['box_nms_thresh']}\n")
        f.write(f"  Learning rate: {config['training']['learning_rate']}\n")
        f.write(f"  Weight decay: {config['training']['weight_decay']}\n")
        f.write(f"  Batch size: {config['training']['batch_size']}\n")
    
    print(f"  ✓ Created summary.txt")
    
    print(f"\nArtifact bundle created successfully!")
    print(f"Location: {output_path}")
    print(f"Size: {get_directory_size(output_path):.1f} MB")
    
    return output_path


def get_directory_size(directory):
    """Calculate the size of a directory in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB


def main():
    parser = argparse.ArgumentParser(description='Save training artifacts for archiving')
    parser.add_argument('--name', help='Name for the artifact bundle (auto-generated if not provided)')
    parser.add_argument('--output', default='archives/', help='Output directory for artifacts')
    parser.add_argument('--no-evaluation', action='store_true', help='Skip evaluation results')
    
    args = parser.parse_args()
    
    # Auto-generate name if not provided
    if not args.name:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.name = f"training_run_{timestamp}"
        print(f"Auto-generated bundle name: {args.name}")
    
    # Create bundle
    bundle_path = create_artifact_bundle(
        args.name, 
        args.output, 
        include_evaluation=not args.no_evaluation
    )
    
    print(f"\nTo restore this training run:")
    print(f"1. Copy the files from {bundle_path} to your project")
    print(f"2. Use the saved config.yaml and best_model.pt")
    print(f"3. Review training_history.json for performance metrics")


if __name__ == "__main__":
    main() 