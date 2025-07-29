#!/usr/bin/env python3
"""
Analyze bounding box distribution for MobileNet training data
Focuses on chunks 1-7 as specified
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_bbox_distribution(data_dir: Path, max_chunk: int = 7) -> Dict:
    """Analyze bounding box distribution from specified chunks"""
    
    bbox_data = {
        'widths': [],
        'heights': [],
        'areas': [],
        'aspect_ratios': [],
        'chunk_counts': {},
        'total_annotations': 0,
        'total_images': 0
    }
    
    # Analyze chunks 1 through max_chunk
    for chunk_num in range(1, max_chunk + 1):
        chunk_dir = data_dir / f"chunk_{chunk_num:03d}"
        ann_file = chunk_dir / "annotations.json"
        
        if not chunk_dir.exists():
            logger.warning(f"Chunk {chunk_num} directory not found: {chunk_dir}")
            continue
            
        if not ann_file.exists():
            logger.warning(f"Annotations file not found for chunk {chunk_num}: {ann_file}")
            continue
        
        logger.info(f"Analyzing chunk {chunk_num}...")
        
        # Load annotations
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Count images in this chunk
        images_dir = chunk_dir / "images"
        if images_dir.exists():
            image_count = len([f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')])
            bbox_data['total_images'] += image_count
            bbox_data['chunk_counts'][chunk_num] = image_count
        
        # Analyze tick annotations
        chunk_annotations = 0
        for ann in annotations.get('annotations', []):
            if ann['category_id'] == 1:  # tick category
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Store bbox statistics
                bbox_data['widths'].append(w)
                bbox_data['heights'].append(h)
                bbox_data['areas'].append(w * h)
                bbox_data['aspect_ratios'].append(w / h if h > 0 else 0)
                chunk_annotations += 1
        
        bbox_data['total_annotations'] += chunk_annotations
        logger.info(f"Chunk {chunk_num}: {chunk_annotations} tick annotations, {image_count} images")
    
    return bbox_data

def create_distribution_plots(bbox_data: Dict, output_dir: Path):
    """Create visualization plots for bounding box distribution"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy arrays for easier analysis
    widths = np.array(bbox_data['widths'])
    heights = np.array(bbox_data['heights'])
    areas = np.array(bbox_data['areas'])
    aspect_ratios = np.array(bbox_data['aspect_ratios'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Bounding Box Distribution Analysis (Chunks 1-7)\n'
                f'Total: {bbox_data["total_annotations"]} annotations, {bbox_data["total_images"]} images', 
                fontsize=16)
    
    # Width distribution
    axes[0, 0].hist(widths, bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.1f}')
    axes[0, 0].axvline(np.median(widths), color='green', linestyle='--', label=f'Median: {np.median(widths):.1f}')
    axes[0, 0].legend()
    
    # Height distribution
    axes[0, 1].hist(heights, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.1f}')
    axes[0, 1].axvline(np.median(heights), color='blue', linestyle='--', label=f'Median: {np.median(heights):.1f}')
    axes[0, 1].legend()
    
    # Area distribution
    axes[0, 2].hist(areas, bins=50, alpha=0.7, color='orange')
    axes[0, 2].set_title('Area Distribution')
    axes[0, 2].set_xlabel('Area (pixels²)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].axvline(np.mean(areas), color='red', linestyle='--', label=f'Mean: {np.mean(areas):.0f}')
    axes[0, 2].axvline(np.median(areas), color='blue', linestyle='--', label=f'Median: {np.median(areas):.0f}')
    axes[0, 2].legend()
    
    # Aspect ratio distribution
    axes[1, 0].hist(aspect_ratios, bins=50, alpha=0.7, color='purple')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].set_xlabel('Width/Height Ratio')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].axvline(np.mean(aspect_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(aspect_ratios):.2f}')
    axes[1, 0].axvline(np.median(aspect_ratios), color='blue', linestyle='--', label=f'Median: {np.median(aspect_ratios):.2f}')
    axes[1, 0].legend()
    
    # Width vs Height scatter
    axes[1, 1].scatter(widths, heights, alpha=0.5, s=10)
    axes[1, 1].set_title('Width vs Height Scatter')
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    
    # Chunk distribution
    chunk_nums = list(bbox_data['chunk_counts'].keys())
    chunk_counts = list(bbox_data['chunk_counts'].values())
    axes[1, 2].bar(chunk_nums, chunk_counts, alpha=0.7, color='teal')
    axes[1, 2].set_title('Images per Chunk')
    axes[1, 2].set_xlabel('Chunk Number')
    axes[1, 2].set_ylabel('Number of Images')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Distribution plots saved to {output_dir / 'bbox_distribution_analysis.png'}")

def print_statistics(bbox_data: Dict):
    """Print comprehensive statistics"""
    
    widths = np.array(bbox_data['widths'])
    heights = np.array(bbox_data['heights'])
    areas = np.array(bbox_data['areas'])
    aspect_ratios = np.array(bbox_data['aspect_ratios'])
    
    print("\n" + "="*60)
    print("BOUNDING BOX DISTRIBUTION ANALYSIS (Chunks 1-7)")
    print("="*60)
    
    print(f"\nOVERVIEW:")
    print(f"  Total annotations: {bbox_data['total_annotations']:,}")
    print(f"  Total images: {bbox_data['total_images']:,}")
    print(f"  Annotations per image: {bbox_data['total_annotations'] / bbox_data['total_images']:.2f}")
    
    print(f"\nCHUNK BREAKDOWN:")
    for chunk_num in sorted(bbox_data['chunk_counts'].keys()):
        print(f"  Chunk {chunk_num}: {bbox_data['chunk_counts'][chunk_num]:,} images")
    
    print(f"\nWIDTH STATISTICS (pixels):")
    print(f"  Mean: {np.mean(widths):.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    print(f"  Std Dev: {np.std(widths):.1f}")
    print(f"  Min: {np.min(widths):.1f}")
    print(f"  Max: {np.max(widths):.1f}")
    print(f"  25th percentile: {np.percentile(widths, 25):.1f}")
    print(f"  75th percentile: {np.percentile(widths, 75):.1f}")
    
    print(f"\nHEIGHT STATISTICS (pixels):")
    print(f"  Mean: {np.mean(heights):.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    print(f"  Std Dev: {np.std(heights):.1f}")
    print(f"  Min: {np.min(heights):.1f}")
    print(f"  Max: {np.max(heights):.1f}")
    print(f"  25th percentile: {np.percentile(heights, 25):.1f}")
    print(f"  75th percentile: {np.percentile(heights, 75):.1f}")
    
    print(f"\nAREA STATISTICS (pixels²):")
    print(f"  Mean: {np.mean(areas):.0f}")
    print(f"  Median: {np.median(areas):.0f}")
    print(f"  Std Dev: {np.std(areas):.0f}")
    print(f"  Min: {np.min(areas):.0f}")
    print(f"  Max: {np.max(areas):.0f}")
    print(f"  25th percentile: {np.percentile(areas, 25):.0f}")
    print(f"  75th percentile: {np.percentile(areas, 75):.0f}")
    
    print(f"\nASPECT RATIO STATISTICS (width/height):")
    print(f"  Mean: {np.mean(aspect_ratios):.2f}")
    print(f"  Median: {np.median(aspect_ratios):.2f}")
    print(f"  Std Dev: {np.std(aspect_ratios):.2f}")
    print(f"  Min: {np.min(aspect_ratios):.2f}")
    print(f"  Max: {np.max(aspect_ratios):.2f}")
    
    print(f"\nSLIDING WINDOW RECOMMENDATIONS:")
    print(f"  Based on 75th percentile width: {np.percentile(widths, 75):.0f}px")
    print(f"  Based on 75th percentile height: {np.percentile(heights, 75):.0f}px")
    print(f"  Recommended window size: {max(np.percentile(widths, 75), np.percentile(heights, 75)) * 2:.0f}x{max(np.percentile(widths, 75), np.percentile(heights, 75)) * 2:.0f}px")
    print(f"  This would capture ~75% of ticks with reasonable padding")
    
    print("="*60)

def main():
    """Main analysis function"""
    
    # Paths
    data_dir = Path("../data")
    output_dir = Path("../outputs/mobile/analysis")
    
    # Analyze bounding box distribution
    logger.info("Starting bounding box distribution analysis...")
    bbox_data = analyze_bbox_distribution(data_dir, max_chunk=7)
    
    # Print statistics
    print_statistics(bbox_data)
    
    # Create visualization plots
    logger.info("Creating distribution plots...")
    create_distribution_plots(bbox_data, output_dir)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 