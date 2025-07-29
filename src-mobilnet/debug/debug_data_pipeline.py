#!/usr/bin/env python3
"""
Debug Data Pipeline
Check what's actually being created by the data pipeline
"""

import json
from pathlib import Path
from data_pipeline import TickBinaryDataset, get_transforms
from config import DATA_DIR, DATA_CONFIG

def debug_data_pipeline():
    """Debug the data pipeline to see what's actually being created"""
    
    print("Debugging data pipeline...")
    
    # Find chunk directories
    chunk_dirs = []
    max_chunk = DATA_CONFIG.get('max_chunk', 7)
    for chunk_num in range(1, max_chunk + 1):
        chunk_dir = DATA_DIR / f"chunk_{chunk_num:03d}"
        if chunk_dir.exists():
            chunk_dirs.append(chunk_dir)
    
    print(f"Using chunks: {[d.name for d in chunk_dirs]}")
    
    # Collect image paths and annotation files (same as in create_data_loaders)
    train_image_paths = []
    train_annotation_paths = []
    
    for chunk_dir in chunk_dirs:
        images_dir = chunk_dir / "images"
        ann_file = chunk_dir / "annotations.json"
        if images_dir.exists() and ann_file.exists():
            image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')]
            train_image_paths.extend(image_files)
            train_annotation_paths.extend([ann_file] * len(image_files))
    
    print(f"Total images found: {len(train_image_paths)}")
    
    # Test with the full dataset to see if the issue is with subset selection
    print("Testing with full dataset...")
    test_image_paths = train_image_paths
    test_annotation_paths = train_annotation_paths
    
    # Create dataset with current settings
    try:
        dataset = TickBinaryDataset(
            image_paths=test_image_paths,
            annotation_files=test_annotation_paths,
            transform=get_transforms('train'),
            mode='train',
            crop_padding=DATA_CONFIG['crop_padding'],
            negative_crops_per_image=DATA_CONFIG['negative_crops_per_image']
        )
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Analyze the samples
        positive_count = 0
        negative_count = 0
        
        for i, sample in enumerate(dataset.samples):
            if sample['label'] == 1:
                positive_count += 1
            else:
                negative_count += 1
            
            # Show first few samples
            if i < 10:
                print(f"Sample {i}: label={sample['label']}, crop={sample['crop_coords']}, path={sample['image_path'].name}")
        
        print(f"\nActual sample distribution:")
        print(f"Positive samples: {positive_count}")
        print(f"Negative samples: {negative_count}")
        print(f"Total samples: {positive_count + negative_count}")
        if positive_count + negative_count > 0:
            print(f"Positive ratio: {positive_count/(positive_count + negative_count)*100:.1f}%")
        
        # Check a few actual samples
        print(f"\nChecking first 5 actual samples:")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: label={sample['label'].item()}, image_shape={sample['image'].shape}")
            
            # Check if image is all zeros or has very low variance
            img_tensor = sample['image']
            img_mean = img_tensor.mean().item()
            img_std = img_tensor.std().item()
            print(f"  Image stats: mean={img_mean:.4f}, std={img_std:.4f}")
            
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("This confirms the hanging issue in the data pipeline")

if __name__ == "__main__":
    debug_data_pipeline() 