#!/usr/bin/env python3

import argparse
import os
import yaml
from utils import split_coco_annotations

def parse_args():
    parser = argparse.ArgumentParser(description='Manage COCO annotation chunks')
    parser.add_argument('--action', choices=['split', 'list', 'clean'], required=True,
                       help='Action to perform: split annotations, list chunks, or clean chunks')
    parser.add_argument('--annotation-file', type=str, default='data/annotations.json',
                       help='Path to the original COCO annotation file')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Number of images per chunk')
    parser.add_argument('--output-dir', type=str, default='data/chunks',
                       help='Directory to save chunk directories')
    return parser.parse_args()

def update_config_for_chunk(config_path: str, chunk_dir: str, chunk_idx: int):
    """Update the config file for a specific chunk."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths
    config['data']['train_path'] = os.path.join(chunk_dir, 'images')
    config['data']['train_annotations'] = os.path.join(chunk_dir, 'annotations.json')
    
    # Update output directories to be chunk-specific
    config['training']['checkpoint_dir'] = os.path.join(config['training']['checkpoint_dir'], f'chunk_{chunk_idx}')
    config['training']['output_dir'] = os.path.join(config['training']['output_dir'], f'chunk_{chunk_idx}')
    
    # Save updated config
    chunk_config_path = f'config_chunk_{chunk_idx}.yaml'
    with open(chunk_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return chunk_config_path

def list_chunks(output_dir: str):
    """List all chunks and their statistics."""
    import json
    
    if not os.path.exists(output_dir):
        print(f"No chunks found in {output_dir}")
        return
    
    chunks = []
    for chunk_dir in sorted(os.listdir(output_dir)):
        chunk_path = os.path.join(output_dir, chunk_dir)
        if not os.path.isdir(chunk_path):
            continue
            
        annotations_file = os.path.join(chunk_path, 'annotations.json')
        images_dir = os.path.join(chunk_path, 'images')
        
        if not os.path.exists(annotations_file):
            print(f"Warning: No annotations.json found in {chunk_dir}")
            continue
            
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        num_physical_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg'))])
        
        chunks.append({
            'dir': chunk_dir,
            'images': num_images,
            'annotations': num_annotations,
            'physical_images': num_physical_images
        })
    
    if not chunks:
        print(f"No valid chunks found in {output_dir}")
        return
    
    print("\nChunk Statistics:")
    print("-" * 100)
    print(f"{'Directory':<20} {'Images':>10} {'Annotations':>15} {'Physical Images':>20}")
    print("-" * 100)
    
    total_images = 0
    total_annotations = 0
    total_physical = 0
    
    for chunk in chunks:
        print(f"{chunk['dir']:<20} {chunk['images']:>10} {chunk['annotations']:>15} {chunk['physical_images']:>20}")
        total_images += chunk['images']
        total_annotations += chunk['annotations']
        total_physical += chunk['physical_images']
    
    print("-" * 100)
    print(f"{'Total:':<20} {total_images:>10} {total_annotations:>15} {total_physical:>20}")

def clean_chunks(output_dir: str):
    """Remove all chunk directories."""
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist")
        return
    
    import shutil
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Cleaned all chunks from {output_dir}")

def main():
    args = parse_args()
    
    if args.action == 'split':
        # Create chunks
        print(f"\nSplitting annotations into chunks of {args.chunk_size} images...")
        chunk_dirs = split_coco_annotations(
            args.annotation_file,
            chunk_size=args.chunk_size,
            output_dir=args.output_dir
        )
        
        # Create chunk-specific configs
        print("\nCreating chunk-specific configurations...")
        for i, chunk_dir in enumerate(chunk_dirs, 1):
            config_path = update_config_for_chunk('config.yaml', chunk_dir, i)
            print(f"Created config for chunk {i}: {config_path}")
            
    elif args.action == 'list':
        list_chunks(args.output_dir)
        
    elif args.action == 'clean':
        clean_chunks(args.output_dir)

if __name__ == '__main__':
    main() 