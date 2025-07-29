#!/usr/bin/env python3
"""
Debug Image Dimensions
Check the actual image dimensions and see why negative sampling is failing
"""

import json
from pathlib import Path
from PIL import Image
from config import DATA_DIR

def debug_image_dimensions():
    """Debug image dimensions and negative sampling"""
    
    # Test with the same image that's failing
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
    
    # Load the actual image to get dimensions
    image = Image.open(test_image)
    img_width, img_height = image.size
    print(f"Actual image dimensions: {img_width} x {img_height}")
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Find the image ID
    image_id = None
    for img_info in annotations.get('images', []):
        if img_info['file_name'] == test_image.name:
            image_id = img_info['id']
            break
    
    if not image_id:
        print("Image not found in annotations")
        return
    
    # Get tick annotations for this image
    image_annotations = []
    for ann in annotations.get('annotations', []):
        if ann['category_id'] == 1 and ann['image_id'] == image_id:
            image_annotations.append(ann)
    
    print(f"Found {len(image_annotations)} tick annotations")
    
    # Show tick bounding boxes
    for i, ann in enumerate(image_annotations):
        tx, ty, tw, th = ann['bbox']
        print(f"  Tick {i+1}: ({tx},{ty}) to ({tx+tw},{ty+th}), size: {tw}x{th}")
    
    # Test the grid approach with actual dimensions
    print(f"\nTesting grid with actual dimensions:")
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

if __name__ == "__main__":
    debug_image_dimensions() 