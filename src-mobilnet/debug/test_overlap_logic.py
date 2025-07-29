#!/usr/bin/env python3
"""
Test Overlap Logic
Verify the overlap detection logic is working correctly
"""

import json
import random
from pathlib import Path
from PIL import Image
from config import DATA_DIR

def test_overlap_logic():
    """Test the overlap detection logic"""
    
    # Find the first image with ticks
    chunk_dir = DATA_DIR / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Find first image with tick annotations
    image_annotations = []
    target_image = None
    
    for ann in annotations.get('annotations', []):
        if ann['category_id'] == 1:  # tick category
            image_id = ann['image_id']
            # Find image filename
            for img_info in annotations.get('images', []):
                if img_info['id'] == image_id:
                    target_image = images_dir / img_info['file_name']
                    break
            if target_image:
                break
    
    if not target_image:
        print("No image with ticks found")
        return
    
    print(f"Testing overlap logic on: {target_image.name}")
    
    # Load image
    image = Image.open(target_image)
    img_width, img_height = image.size
    print(f"Image size: {img_width}x{img_height}")
    
    # Get all annotations for this image
    image_annotations = []
    for ann in annotations.get('annotations', []):
        if ann['image_id'] == image_id and ann['category_id'] == 1:
            image_annotations.append(ann)
    
    print(f"Number of tick annotations: {len(image_annotations)}")
    
    # Show bounding boxes
    for i, ann in enumerate(image_annotations):
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        print(f"  Bbox {i}: ({x}, {y}, {w}, {h})")
    
    # Test overlap detection with known non-overlapping regions
    print(f"\nTesting overlap detection:")
    
    # Test 1: Crop completely outside all bounding boxes
    test_crop = (0, 0, 50, 50)  # Top-left corner
    x1, y1, x2, y2 = test_crop
    
    overlap = False
    for ann in image_annotations:
        tx, ty, tw, th = ann['bbox']
        if (x1 < tx + tw and x2 > tx and 
            y1 < ty + th and y2 > ty):
            overlap = True
            print(f"  Test 1 (0,0,50,50): OVERLAPS with bbox ({tx}, {ty}, {tw}, {th})")
            break
    
    if not overlap:
        print(f"  Test 1 (0,0,50,50): NO OVERLAP ✓")
    
    # Test 2: Crop that should overlap
    test_crop = (100, 100, 150, 150)  # Middle area
    x1, y1, x2, y2 = test_crop
    
    overlap = False
    for ann in image_annotations:
        tx, ty, tw, th = ann['bbox']
        if (x1 < tx + tw and x2 > tx and 
            y1 < ty + th and y2 > ty):
            overlap = True
            print(f"  Test 2 (100,100,150,150): OVERLAPS with bbox ({tx}, {ty}, {tw}, {th})")
            break
    
    if not overlap:
        print(f"  Test 2 (100,100,150,150): NO OVERLAP")
    
    # Test 3: Try 10 random small crops and see how many overlap
    print(f"\nTesting 10 random 32x32 crops:")
    overlap_count = 0
    
    for i in range(10):
        x1 = random.randint(0, img_width - 32)
        y1 = random.randint(0, img_height - 32)
        x2 = x1 + 32
        y2 = y1 + 32
        
        overlap = False
        for ann in image_annotations:
            tx, ty, tw, th = ann['bbox']
            if (x1 < tx + tw and x2 > tx and 
                y1 < ty + th and y2 > ty):
                overlap = True
                break
        
        if overlap:
            overlap_count += 1
            print(f"  Crop {i+1} ({x1},{y1},{x2},{y2}): OVERLAPS")
        else:
            print(f"  Crop {i+1} ({x1},{y1},{x2},{y2}): NO OVERLAP")
    
    print(f"\nResults: {overlap_count}/10 crops overlapped ({overlap_count*10}%)")

if __name__ == "__main__":
    test_overlap_logic() 