import os
import shutil
import argparse
from pathlib import Path

def import_images(source_dir, target_dir="data", chunks=None):
    """
    Import images from external storage to the data directory.
    
    Args:
        source_dir (str): Path to the external storage directory containing chunks
        target_dir (str): Path to the data directory (default: "data")
        chunks (list): List of chunk numbers to import (e.g., ["001", "002"]). If None, imports all chunks.
    """
    # Convert chunks to list of chunk directories
    if chunks:
        chunk_dirs = [f"chunk_{chunk:03d}" for chunk in chunks]
    else:
        # Get all chunk directories from source
        chunk_dirs = [d for d in os.listdir(source_dir) if d.startswith("chunk_")]
        chunk_dirs.sort()

    print(f"Found {len(chunk_dirs)} chunks to process")
    
    for chunk_dir in chunk_dirs:
        source_chunk = os.path.join(source_dir, chunk_dir)
        target_chunk = os.path.join(target_dir, chunk_dir)
        
        if not os.path.exists(source_chunk):
            print(f"Warning: Source chunk {chunk_dir} not found, skipping...")
            continue
            
        source_images = os.path.join(source_chunk, "images")
        target_images = os.path.join(target_chunk, "images")
        
        if not os.path.exists(source_images):
            print(f"Warning: No images directory in {chunk_dir}, skipping...")
            continue
            
        # Create target directory if it doesn't exist
        os.makedirs(target_images, exist_ok=True)
        
        # Count images before copying
        image_count = len([f for f in os.listdir(source_images) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\nProcessing {chunk_dir}...")
        print(f"  Found {image_count} images to copy")
        
        # Copy images
        for img in os.listdir(source_images):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(source_images, img)
                dst = os.path.join(target_images, img)
                shutil.copy2(src, dst)
        
        print(f"  Copied {image_count} images to {target_images}")

def main():
    parser = argparse.ArgumentParser(description="Import images from external storage to data directory")
    parser.add_argument("source_dir", help="Path to external storage directory containing chunks")
    parser.add_argument("--target-dir", default="data", help="Path to data directory (default: data)")
    parser.add_argument("--chunks", nargs="+", help="List of chunk numbers to import (e.g., 1 2 3)")
    
    args = parser.parse_args()
    
    # Convert chunk numbers to list if provided
    chunks = None
    if args.chunks:
        chunks = [int(c) for c in args.chunks]
    
    import_images(args.source_dir, args.target_dir, chunks)

if __name__ == "__main__":
    main() 