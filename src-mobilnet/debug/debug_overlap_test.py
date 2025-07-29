#!/usr/bin/env python3
"""
Debug Overlap Test
Test the overlap detection logic on a simple case
"""

import json
from pathlib import Path
from config import DATA_DIR

def debug_overlap_test():
    """Test the overlap detection logic"""
    
    # Test with a simple case first
    chunk_dir = DATA_DIR / "chunk_001"
    ann_file = chunk_dir / "annotations.json"
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Find first image with ticks
    image_annotations = []
    target_image = None
    
    for ann in annotations.get('annotations', []):
        if ann['category_id'] == 1:  # tick category
            image_id = ann['image_id']
            # Find the image info
            for img_info in annotations.get('images', []):
                if img_info['id'] == image_id:
                    target_image = img_info['file_name']
                    break
            if target_image:
                break
    
    if not target_image:
        print("No image with ticks found")
        return
    
    print(f"Testing with image: {target_image}")
    
    # Get image dimensions (approximate)
    img_width, img_height = 500, 375  # Typical size
    
    # Get all tick annotations for this image
    for ann in annotations.get('annotations', []):
        if ann['category_id'] == 1 and ann['image_id'] == image_id:
            image_annotations.append(ann)
    
    print(f"Found {len(image_annotations)} tick annotations")
    
    # Test the overlap detection logic
    print("\nTesting overlap detection:")
    
    # Test a few grid cells
    grid_size = 16
    grid_cols = max(1, img_width // grid_size)
    grid_rows = max(1, img_height // grid_size)
    
    print(f"Grid: {grid_cols} x {grid_rows} = {grid_cols * grid_rows} total cells")
    
    negative_cells = 0
    total_cells = 0
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            total_cells += 1
            
            # Calculate crop coordinates for this grid cell
            x1 = col * grid_size
            y1 = row * grid_size
            x2 = min(x1 + grid_size, img_width)
            y2 = min(y1 + grid_size, img_height)
            
            # Skip if crop is too small
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            
            # Check overlap percentage
            total_overlap_area = 0
            crop_area = (x2 - x1) * (y2 - y1)
            
            for ann in image_annotations:
                tx, ty, tw, th = ann['bbox']
                # Calculate intersection area
                ix1 = max(x1, tx)
                iy1 = max(y1, ty)
                ix2 = min(x2, tx + tw)
                iy2 = min(y2, ty + th)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    total_overlap_area += intersection_area
            
            # Allow up to 10% overlap
            overlap_percentage = total_overlap_area / crop_area if crop_area > 0 else 0
            
            if overlap_percentage < 0.1:  # Less than 10% overlap
                negative_cells += 1
                
                # Show first few negative cells
                if negative_cells <= 5:
                    print(f"  Negative cell {negative_cells}: ({x1},{y1}) to ({x2},{y2}), overlap: {overlap_percentage:.3f}")
    
    print(f"\nResults:")
    print(f"Total cells: {total_cells}")
    print(f"Negative cells: {negative_cells}")
    print(f"Negative percentage: {negative_cells/total_cells*100:.1f}%")
    
    # Also show the tick bounding boxes
    print(f"\nTick bounding boxes:")
    for i, ann in enumerate(image_annotations):
        tx, ty, tw, th = ann['bbox']
        print(f"  Tick {i+1}: ({tx},{ty}) to ({tx+tw},{ty+th}), size: {tw}x{th}")

if __name__ == "__main__":
    debug_overlap_test() 