#!/usr/bin/env python3
"""
Test Sample Counting
Count how many samples are actually being created per image
"""

import json
from pathlib import Path
from config import DATA_DIR, DATA_CONFIG

def test_sample_counting():
    """Test how many samples are being created per image"""
    
    print("Testing sample counting...")
    
    # Find chunk directories
    chunk_dirs = []
    max_chunk = DATA_CONFIG.get('max_chunk', 7)
    for chunk_num in range(1, max_chunk + 1):
        chunk_dir = DATA_DIR / f"chunk_{chunk_num:03d}"
        if chunk_dir.exists():
            chunk_dirs.append(chunk_dir)
    
    print(f"Using chunks: {[d.name for d in chunk_dirs]}")
    
    total_images = 0
    total_positive_samples = 0
    total_negative_samples = 0
    images_with_ticks = 0
    images_without_ticks = 0
    
    for chunk_dir in chunk_dirs:
        images_dir = chunk_dir / "images"
        ann_file = chunk_dir / "annotations.json"
        
        if not images_dir.exists() or not ann_file.exists():
            continue
            
        # Load annotations
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Count images
        image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')]
        total_images += len(image_files)
        
        # Count annotations per image
        image_annotation_counts = {}
        for ann in annotations.get('annotations', []):
            if ann['category_id'] == 1:  # tick category
                image_id = ann['image_id']
                if image_id not in image_annotation_counts:
                    image_annotation_counts[image_id] = 0
                image_annotation_counts[image_id] += 1
        
        # Calculate samples per image
        for img_info in annotations.get('images', []):
            image_id = img_info['id']
            tick_count = image_annotation_counts.get(image_id, 0)
            
            if tick_count > 0:
                images_with_ticks += 1
                # Positive samples: one per bounding box
                positive_samples = tick_count
                # Negative samples: trying to create negative_crops_per_image
                negative_samples = DATA_CONFIG['negative_crops_per_image']
                total_positive_samples += positive_samples
                total_negative_samples += negative_samples
            else:
                images_without_ticks += 1
                # Images without ticks: only negative samples
                negative_samples = DATA_CONFIG['negative_crops_per_image']
                total_negative_samples += negative_samples
    
    print(f"\nSample counting results:")
    print(f"Total images: {total_images}")
    print(f"Images with ticks: {images_with_ticks}")
    print(f"Images without ticks: {images_without_ticks}")
    print(f"Total positive samples (theoretical): {total_positive_samples}")
    print(f"Total negative samples (theoretical): {total_negative_samples}")
    print(f"Total samples (theoretical): {total_positive_samples + total_negative_samples}")
    
    # Calculate actual vs theoretical
    if total_images > 0:
        avg_positive_per_image = total_positive_samples / total_images
        avg_negative_per_image = total_negative_samples / total_images
        print(f"Average positive samples per image: {avg_positive_per_image:.2f}")
        print(f"Average negative samples per image: {avg_negative_per_image:.2f}")

if __name__ == "__main__":
    test_sample_counting() 