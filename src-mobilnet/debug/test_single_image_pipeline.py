#!/usr/bin/env python3
"""
Test Single Image Pipeline
Run the actual data pipeline on a single image to debug the issue
"""

import json
from pathlib import Path
from data_pipeline import TickBinaryDataset, get_transforms
from config import DATA_DIR, DATA_CONFIG

def test_single_image_pipeline():
    """Test the data pipeline on a single image"""
    
    # Test with the first image from chunk_001
    chunk_dir = DATA_DIR / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    # Get the first image
    image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')]
    if not image_files:
        print("No images found")
        return
    
    test_image = image_files[0]
    print(f"Testing with image: {test_image.name}")
    
    # Create a single-image dataset
    test_image_paths = [test_image]
    test_annotation_paths = [ann_file]
    
    # Create dataset
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
            print(f"Sample {i}: label={sample['label']}, crop={sample['crop_coords']}")
    
    print(f"\nSample distribution:")
    print(f"Positive samples: {positive_count}")
    print(f"Negative samples: {negative_count}")
    print(f"Total samples: {positive_count + negative_count}")
    if positive_count + negative_count > 0:
        print(f"Positive ratio: {positive_count/(positive_count + negative_count)*100:.1f}%")

if __name__ == "__main__":
    test_single_image_pipeline() 