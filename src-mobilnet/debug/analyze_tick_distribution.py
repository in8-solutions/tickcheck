#!/usr/bin/env python3
"""
Analyze Tick Distribution
Look at the actual annotations in chunks 1-7 to understand tick distribution
"""

import json
from pathlib import Path
from collections import Counter
from config import DATA_DIR

def analyze_tick_distribution():
    """Analyze the distribution of ticks per image in chunks 1-7"""
    
    print("Analyzing tick distribution in chunks 1-7...")
    
    # Find chunk directories
    chunk_dirs = []
    for chunk_num in range(1, 8):  # chunks 1-7
        chunk_dir = DATA_DIR / f"chunk_{chunk_num:03d}"
        if chunk_dir.exists():
            chunk_dirs.append(chunk_dir)
    
    print(f"Found chunks: {[d.name for d in chunk_dirs]}")
    
    total_images = 0
    total_ticks = 0
    tick_counts_per_image = []
    images_with_ticks = 0
    images_without_ticks = 0
    
    for chunk_dir in chunk_dirs:
        print(f"\nAnalyzing {chunk_dir.name}...")
        
        ann_file = chunk_dir / "annotations.json"
        if not ann_file.exists():
            print(f"  No annotations file found")
            continue
            
        # Load annotations
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Count images
        image_files = list((chunk_dir / "images").glob("*.jpg"))
        chunk_images = len(image_files)
        total_images += chunk_images
        print(f"  Images in chunk: {chunk_images}")
        
        # Count ticks per image
        image_tick_counts = {}
        chunk_ticks = 0
        
        for ann in annotations.get('annotations', []):
            if ann['category_id'] == 1:  # tick category
                image_id = ann['image_id']
                if image_id not in image_tick_counts:
                    image_tick_counts[image_id] = 0
                image_tick_counts[image_id] += 1
                chunk_ticks += 1
        
        total_ticks += chunk_ticks
        print(f"  Total ticks in chunk: {chunk_ticks}")
        
        # Analyze distribution
        tick_counts = list(image_tick_counts.values())
        tick_counts_per_image.extend(tick_counts)
        
        if tick_counts:
            print(f"  Ticks per image: min={min(tick_counts)}, max={max(tick_counts)}, avg={sum(tick_counts)/len(tick_counts):.2f}")
        
        # Count images with/without ticks
        images_with_ticks_in_chunk = len(image_tick_counts)
        images_without_ticks_in_chunk = chunk_images - images_with_ticks_in_chunk
        
        images_with_ticks += images_with_ticks_in_chunk
        images_without_ticks += images_without_ticks_in_chunk
        
        print(f"  Images with ticks: {images_with_ticks_in_chunk}")
        print(f"  Images without ticks: {images_without_ticks_in_chunk}")
    
    # Overall statistics
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total images: {total_images}")
    print(f"Total ticks: {total_ticks}")
    print(f"Images with ticks: {images_with_ticks}")
    print(f"Images without ticks: {images_without_ticks}")
    
    if tick_counts_per_image:
        print(f"\nTick distribution:")
        tick_counter = Counter(tick_counts_per_image)
        for tick_count in sorted(tick_counter.keys()):
            count = tick_counter[tick_count]
            percentage = (count / len(tick_counts_per_image)) * 100
            print(f"  {tick_count} tick(s): {count} images ({percentage:.1f}%)")
        
        print(f"\nSummary:")
        print(f"  Average ticks per image: {sum(tick_counts_per_image)/len(tick_counts_per_image):.2f}")
        print(f"  Median ticks per image: {sorted(tick_counts_per_image)[len(tick_counts_per_image)//2]}")
        print(f"  Most common: {tick_counter.most_common(1)[0][0]} tick(s) per image")

if __name__ == "__main__":
    analyze_tick_distribution() 