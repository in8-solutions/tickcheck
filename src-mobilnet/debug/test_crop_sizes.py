#!/usr/bin/env python3
"""
Test Crop Sizes
Understand the relationship between image sizes and crop sizes
"""

import json
from pathlib import Path
from PIL import Image
from config import DATA_DIR

def test_crop_sizes():
    """Test different crop sizes to understand the issue"""
    
    # Find a few images to test
    chunk_dir = DATA_DIR / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    # Test first 5 images
    test_count = 0
    for img_info in annotations.get('images', []):
        if test_count >= 5:
            break
            
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            continue
            
        # Load image
        image = Image.open(img_path)
        img_width, img_height = image.size
        
        print(f"\nImage: {img_path.name}")
        print(f"  Size: {img_width}x{img_height}")
        
        # Test different crop size strategies
        strategies = [
            ("Original (200, w//2, h//2)", min(200, img_width // 2, img_height // 2)),
            ("Smaller (100, w//4, h//4)", min(100, img_width // 4, img_height // 4)),
            ("Very small (50, w//8, h//8)", min(50, img_width // 8, img_height // 8)),
            ("Fixed small (64)", 64),
            ("Fixed tiny (32)", 32),
        ]
        
        for name, crop_size in strategies:
            if crop_size < 32:
                crop_size = min(32, img_width, img_height)
            
            # Calculate how many possible crops
            possible_crops = (img_width - crop_size + 1) * (img_height - crop_size + 1)
            print(f"  {name}: {crop_size}x{crop_size} ({possible_crops} possible positions)")
        
        test_count += 1

if __name__ == "__main__":
    test_crop_sizes() 