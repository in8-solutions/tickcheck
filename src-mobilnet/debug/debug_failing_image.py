#!/usr/bin/env python3
"""
Debug Failing Image
Examine one of the images that's failing negative sampling
"""

import json
import random
from pathlib import Path
from PIL import Image
from config import DATA_DIR, DATA_CONFIG

def debug_failing_image():
    """Debug negative sampling for a failing image"""
    
    # Find the first failing image
    chunk_dir = DATA_DIR / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Find tick_2176.jpg (one of the failing images)
    target_image = images_dir / "tick_2176.jpg"
    
    if not target_image.exists():
        print(f"Image {target_image} not found")
        return
    
    print(f"Debugging failing image: {target_image.name}")
    
    # Load image
    image = Image.open(target_image)
    img_width, img_height = image.size
    print(f"Image size: {img_width}x{img_height}")
    
    # Find image_id for this image
    image_id = None
    for img_info in annotations.get('images', []):
        if img_info['file_name'] == target_image.name:
            image_id = img_info['id']
            break
    
    if image_id is None:
        print("Image not found in annotations")
        return
    
    # Get all annotations for this image
    image_annotations = []
    for ann in annotations.get('annotations', []):
        if ann['image_id'] == image_id and ann['category_id'] == 1:
            image_annotations.append(ann)
    
    print(f"Number of tick annotations: {len(image_annotations)}")
    
    # Show bounding boxes
    total_coverage = 0
    for i, ann in enumerate(image_annotations):
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        area = w * h
        coverage = (area / (img_width * img_height)) * 100
        total_coverage += coverage
        print(f"  Bbox {i}: ({x}, {y}, {w}, {h}) - area: {area}, coverage: {coverage:.1f}%")
    
    print(f"Total coverage: {total_coverage:.1f}%")
    print(f"Available space: {100 - total_coverage:.1f}%")
    
    # Test negative sampling with the exact same logic as data_pipeline.py
    negative_crops_per_image = DATA_CONFIG['negative_crops_per_image']
    print(f"\nTrying to create {negative_crops_per_image} negative samples...")
    
    negative_samples_created = 0
    max_attempts = 200
    attempts = 0
    
    while negative_samples_created < negative_crops_per_image and attempts < max_attempts:
        attempts += 1
        
        # Use the exact same crop size logic as data_pipeline.py
        crop_size = min(200, img_width // 2, img_height // 2)
        if crop_size < 50:
            crop_size = min(50, img_width, img_height)
        
        x1 = random.randint(0, max(0, img_width - crop_size))
        y1 = random.randint(0, max(0, img_height - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        print(f"  Attempt {attempts}: crop ({x1}, {y1}, {x2}, {y2}) size={crop_size}")
        
        # Check if crop overlaps with any tick bounding box
        overlap = False
        for ann in image_annotations:
            tx, ty, tw, th = ann['bbox']
            # Check for intersection
            if (x1 < tx + tw and x2 > tx and 
                y1 < ty + th and y2 > ty):
                overlap = True
                print(f"    OVERLAPS with bbox ({tx}, {ty}, {tw}, {th})")
                break
        
        if not overlap:
            print(f"    SUCCESS - no overlap")
            negative_samples_created += 1
        else:
            print(f"    FAILED - has overlap")
    
    print(f"\nResults:")
    print(f"  Negative samples created: {negative_samples_created}/{negative_crops_per_image}")
    print(f"  Attempts made: {attempts}")
    if attempts > 0:
        print(f"  Success rate: {negative_samples_created/attempts*100:.1f}%")

if __name__ == "__main__":
    debug_failing_image() 