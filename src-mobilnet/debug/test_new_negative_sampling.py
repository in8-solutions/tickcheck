#!/usr/bin/env python3
"""
Test New Negative Sampling - Test the new edge-based approach
"""

from data_pipeline import TickBinaryDataset, get_transforms
from config import DATA_DIR, DATA_CONFIG
from pathlib import Path

def test_new_approach():
    """Test the new edge-based negative sampling approach"""
    
    print("Testing new edge-based negative sampling...")
    
    # Use only chunk_001 for testing
    chunk_dir = DATA_DIR / "chunk_001"
    images_dir = chunk_dir / "images"
    ann_file = chunk_dir / "annotations.json"
    
    if not images_dir.exists() or not ann_file.exists():
        print("Chunk directory not found")
        return
    
    # Get just the first 5 images
    image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')][:5]
    annotation_files = [ann_file] * len(image_files)
    
    print(f"Testing with {len(image_files)} images")
    
    # Create dataset
    try:
        dataset = TickBinaryDataset(
            image_paths=image_files,
            annotation_files=annotation_files,
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
                print(f"Sample {i}: label={sample['label']}, crop={sample['crop_coords']}, path={sample['image_path'].name}")
        
        print(f"\nSample distribution:")
        print(f"Positive samples: {positive_count}")
        print(f"Negative samples: {negative_count}")
        print(f"Total samples: {positive_count + negative_count}")
        if positive_count + negative_count > 0:
            print(f"Positive ratio: {positive_count/(positive_count + negative_count)*100:.1f}%")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_approach() 