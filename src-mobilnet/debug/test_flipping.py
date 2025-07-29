#!/usr/bin/env python3

import sys
import os
from pathlib import Path
sys.path.append('..')

from data_pipeline import TickBinaryDataset

def test_flipping():
    print("Testing flipping implementation for negative samples...")
    
    # Set up paths for chunk1
    data_dir = Path("../../data")
    chunk_dir = data_dir / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    # Get all image files (limit to first 20 for testing)
    image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')][:20]
    annotation_files = [ann_file] * len(image_files)
    
    print(f"Testing with {len(image_files)} images")
    
    # Create dataset
    dataset = TickBinaryDataset(
        image_paths=image_files,
        annotation_files=annotation_files,
        mode='train'
    )
    
    print(f"Total samples: {len(dataset)}")
    
    # Find negative samples and check if they have flipping flags
    negative_samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['label'] == 0:  # negative sample
            negative_samples.append(i)
            if len(negative_samples) >= 10:  # Check first 10 negative samples
                break
    
    print(f"\nFound {len(negative_samples)} negative samples")
    
    # Check the flipping flags in the original sample data
    for i, sample_idx in enumerate(negative_samples):
        original_sample = dataset.samples[sample_idx]
        has_horizontal_flip = original_sample.get('flip_horizontal', False)
        has_vertical_flip = original_sample.get('flip_vertical', False)
        
        print(f"Negative sample {i+1}: horizontal_flip={has_horizontal_flip}, vertical_flip={has_vertical_flip}")
        
        if has_horizontal_flip or has_vertical_flip:
            print(f"  ✓ This sample will be flipped to avoid edge bias")
        else:
            print(f"  ✗ This sample has no flipping flags (unexpected)")

if __name__ == "__main__":
    test_flipping() 