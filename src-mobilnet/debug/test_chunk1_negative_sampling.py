#!/usr/bin/env python3

import sys
import os
from pathlib import Path
sys.path.append('..')

from data_pipeline import TickBinaryDataset

def test_chunk1_negative_sampling():
    print("Testing edge-based negative sampling on all images in chunk1...")
    
    # Set up paths for chunk1
    data_dir = Path("../../data")
    chunk_dir = data_dir / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    if not chunk_dir.exists():
        print(f"Chunk directory not found: {chunk_dir}")
        return
    
    # Get all image files
    image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')]
    annotation_files = [ann_file] * len(image_files)
    
    print(f"Found {len(image_files)} images in chunk1")
    
    # Create dataset for chunk1
    dataset = TickBinaryDataset(
        image_paths=image_files,
        annotation_files=annotation_files,
        mode='train'
    )
    
    print(f"Total samples created: {len(dataset)}")
    
    # Count positive and negative samples
    positive_count = 0
    negative_count = 0
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['label'] == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    print(f"\nSample distribution:")
    print(f"Positive samples: {positive_count}")
    print(f"Negative samples: {negative_count}")
    print(f"Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        positive_ratio = (positive_count / len(dataset)) * 100
        print(f"Positive ratio: {positive_ratio:.1f}%")
        print(f"Negative ratio: {100 - positive_ratio:.1f}%")
    
    # Show some sample details
    print(f"\nFirst 5 samples:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}: label={sample['label']}, crop={sample['crop_coords']}, path={sample['image_path']}")

if __name__ == "__main__":
    test_chunk1_negative_sampling() 