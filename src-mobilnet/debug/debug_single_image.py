#!/usr/bin/env python3
"""
Debug Single Image - Check why new approach fails
"""

import json
from pathlib import Path
from config import DATA_DIR

def debug_single_image():
    """Debug why new approach fails for tick_2176.jpg"""
    
    chunk_dir = DATA_DIR / "chunk_001"
    ann_file = chunk_dir / "annotations.json"
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Find tick_2176.jpg
    target_image = None
    for img_info in annotations.get('images', []):
        if img_info['file_name'] == 'tick_2176.jpg':
            target_image = img_info
            break
    
    if not target_image:
        print("Image not found")
        return
    
    # Get tick annotations for this image
    image_id = target_image['id']
    image_annotations = []
    for ann in annotations.get('annotations', []):
        if ann['category_id'] == 1 and ann['image_id'] == image_id:
            image_annotations.append(ann)
    
    print(f"Image: {target_image['file_name']}")
    print(f"Size: {target_image['width']} x {target_image['height']}")
    print(f"Ticks: {len(image_annotations)}")
    
    for i, ann in enumerate(image_annotations):
        print(f"  Tick {i+1}: {ann['bbox']}")
    
    # Test the new logic
    if len(image_annotations) == 1:
        print("\nTesting new edge-based logic:")
        bbox = image_annotations[0]['bbox']
        tx, ty, tw, th = bbox
        img_width, img_height = target_image['width'], target_image['height']
        target_size = 200
        
        print(f"Bbox: ({tx}, {ty}, {tw}, {th})")
        print(f"Image: {img_width} x {img_height}")
        print(f"Target size: {target_size}")
        
        # Test each edge
        print(f"\nLeft edge: tx >= target_size? {tx} >= {target_size} = {tx >= target_size}")
        print(f"Right edge: tx + tw + target_size <= img_width? {tx + tw + target_size} <= {img_width} = {tx + tw + target_size <= img_width}")
        print(f"Top edge: ty >= target_size? {ty} >= {target_size} = {ty >= target_size}")
        print(f"Bottom edge: ty + th + target_size <= img_height? {ty + th + target_size} <= {img_height} = {ty + th + target_size <= img_height}")
        
    else:
        print(f"\nImage has {len(image_annotations)} ticks, not 1")

if __name__ == "__main__":
    debug_single_image() 