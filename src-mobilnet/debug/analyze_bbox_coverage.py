#!/usr/bin/env python3
"""
Analyze Bounding Box Coverage
Determines what percentage of images have bounding boxes covering >75% of the image area.
This helps identify if we have sufficient tick-free zones for creating negative samples.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_bbox_coverage(data_dir: Path, max_chunk: int = 7) -> Dict:
    """
    Analyze bounding box coverage for all images in chunks 1-max_chunk
    
    Returns:
        Dict with coverage statistics and problematic images
    """
    coverage_data = []
    problematic_images = []
    
    for chunk_num in range(1, max_chunk + 1):
        chunk_dir = data_dir / f"chunk_{chunk_num:03d}"
        if not chunk_dir.exists():
            logger.warning(f"Chunk {chunk_num} directory not found: {chunk_dir}")
            continue
            
        ann_file = chunk_dir / "annotations.json"
        if not ann_file.exists():
            logger.warning(f"Annotations file not found: {ann_file}")
            continue
            
        # Load annotations
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Create image_id to annotations mapping
        img_to_anns = {}
        for ann in annotations.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)
        
        # Analyze each image
        for img_info in annotations.get('images', []):
            img_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']
            img_area = img_width * img_height
            
            tick_annotations = [a for a in img_to_anns.get(img_id, []) if a['category_id'] == 1]
            
            if not tick_annotations:
                # Image with no ticks - 0% coverage
                coverage_data.append({
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'chunk': chunk_num,
                    'coverage_percent': 0.0,
                    'bbox_count': 0,
                    'total_bbox_area': 0,
                    'image_area': img_area,
                    'img_width': img_width,
                    'img_height': img_height
                })
                continue
            
            # Calculate total bounding box area
            total_bbox_area = 0
            for ann in tick_annotations:
                x, y, w, h = ann['bbox']
                bbox_area = w * h
                total_bbox_area += bbox_area
            
            coverage_percent = (total_bbox_area / img_area) * 100
            
            coverage_data.append({
                'image_id': img_id,
                'filename': img_info['file_name'],
                'chunk': chunk_num,
                'coverage_percent': coverage_percent,
                'bbox_count': len(tick_annotations),
                'total_bbox_area': total_bbox_area,
                'image_area': img_area,
                'img_width': img_width,
                'img_height': img_height
            })
            
            # Check if problematic (>75% coverage)
            if coverage_percent > 75:
                problematic_images.append({
                    'image_id': img_id,
                    'filename': img_info['file_name'],
                    'chunk': chunk_num,
                    'coverage_percent': coverage_percent,
                    'bbox_count': len(tick_annotations)
                })
    
    return {
        'coverage_data': coverage_data,
        'problematic_images': problematic_images
    }

def print_statistics(analysis_results: Dict):
    """Print coverage statistics"""
    coverage_data = analysis_results['coverage_data']
    problematic_images = analysis_results['problematic_images']
    
    if not coverage_data:
        print("No data found!")
        return
    
    # Calculate statistics
    coverage_percents = [d['coverage_percent'] for d in coverage_data]
    bbox_counts = [d['bbox_count'] for d in coverage_data]
    
    print("=" * 60)
    print("BOUNDING BOX COVERAGE ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal images analyzed: {len(coverage_data)}")
    print(f"Images with ticks: {len([d for d in coverage_data if d['bbox_count'] > 0])}")
    print(f"Images without ticks: {len([d for d in coverage_data if d['bbox_count'] == 0])}")
    
    print(f"\nCoverage Statistics:")
    print(f"  Mean coverage: {np.mean(coverage_percents):.2f}%")
    print(f"  Median coverage: {np.median(coverage_percents):.2f}%")
    print(f"  Std coverage: {np.std(coverage_percents):.2f}%")
    print(f"  Min coverage: {np.min(coverage_percents):.2f}%")
    print(f"  Max coverage: {np.max(coverage_percents):.2f}%")
    
    print(f"\nBounding Box Count Statistics:")
    print(f"  Mean bbox count: {np.mean(bbox_counts):.2f}")
    print(f"  Median bbox count: {np.median(bbox_counts):.2f}")
    print(f"  Max bbox count: {np.max(bbox_counts)}")
    
    # Coverage thresholds
    thresholds = [25, 50, 75, 90, 95]
    print(f"\nCoverage Threshold Analysis:")
    for threshold in thresholds:
        count = len([d for d in coverage_data if d['coverage_percent'] > threshold])
        percent = (count / len(coverage_data)) * 100
        print(f"  >{threshold}% coverage: {count} images ({percent:.1f}%)")
    
    print(f"\nPROBLEMATIC IMAGES (>75% coverage): {len(problematic_images)}")
    if problematic_images:
        print("  These images may have insufficient tick-free zones for negative sampling:")
        for img in problematic_images[:10]:  # Show first 10
            print(f"    {img['filename']} (chunk_{img['chunk']}): {img['coverage_percent']:.1f}% ({img['bbox_count']} bboxes)")
        if len(problematic_images) > 10:
            print(f"    ... and {len(problematic_images) - 10} more")
    
    # Recommendations
    problematic_percent = (len(problematic_images) / len(coverage_data)) * 100
    print(f"\nRECOMMENDATIONS:")
    if problematic_percent > 50:
        print(f"  ⚠️  HIGH RISK: {problematic_percent:.1f}% of images have >75% coverage")
        print(f"     Consider: Using external negative samples or different sampling strategy")
    elif problematic_percent > 25:
        print(f"  ⚠️  MEDIUM RISK: {problematic_percent:.1f}% of images have >75% coverage")
        print(f"     Consider: Increasing negative_crops_per_image or using external negatives")
    else:
        print(f"  ✅ LOW RISK: {problematic_percent:.1f}% of images have >75% coverage")
        print(f"     Current negative sampling strategy should work well")

def create_coverage_plots(analysis_results: Dict, output_dir: Path):
    """Create visualization plots for coverage analysis"""
    coverage_data = analysis_results['coverage_data']
    
    if not coverage_data:
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bounding Box Coverage Analysis', fontsize=16, fontweight='bold')
    
    coverage_percents = [d['coverage_percent'] for d in coverage_data]
    bbox_counts = [d['bbox_count'] for d in coverage_data]
    
    # Plot 1: Coverage distribution histogram
    axes[0, 0].hist(coverage_percents, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=75, color='red', linestyle='--', label='75% threshold')
    axes[0, 0].set_xlabel('Coverage Percentage')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].set_title('Distribution of Bounding Box Coverage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Coverage vs bbox count scatter
    axes[0, 1].scatter(bbox_counts, coverage_percents, alpha=0.6, color='green')
    axes[0, 1].axhline(y=75, color='red', linestyle='--', label='75% threshold')
    axes[0, 1].set_xlabel('Number of Bounding Boxes')
    axes[0, 1].set_ylabel('Coverage Percentage')
    axes[0, 1].set_title('Coverage vs Number of Bounding Boxes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Coverage by chunk
    chunk_data = {}
    for d in coverage_data:
        chunk = d['chunk']
        if chunk not in chunk_data:
            chunk_data[chunk] = []
        chunk_data[chunk].append(d['coverage_percent'])
    
    chunks = sorted(chunk_data.keys())
    chunk_means = [np.mean(chunk_data[chunk]) for chunk in chunks]
    chunk_stds = [np.std(chunk_data[chunk]) for chunk in chunks]
    
    axes[1, 0].bar(chunks, chunk_means, yerr=chunk_stds, capsize=5, alpha=0.7, color='orange')
    axes[1, 0].axhline(y=75, color='red', linestyle='--', label='75% threshold')
    axes[1, 0].set_xlabel('Chunk Number')
    axes[1, 0].set_ylabel('Mean Coverage Percentage')
    axes[1, 0].set_title('Mean Coverage by Chunk')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative distribution
    sorted_coverage = sorted(coverage_percents)
    cumulative = np.arange(1, len(sorted_coverage) + 1) / len(sorted_coverage) * 100
    axes[1, 1].plot(sorted_coverage, cumulative, linewidth=2, color='purple')
    axes[1, 1].axvline(x=75, color='red', linestyle='--', label='75% threshold')
    axes[1, 1].set_xlabel('Coverage Percentage')
    axes[1, 1].set_ylabel('Cumulative Percentage of Images')
    axes[1, 1].set_title('Cumulative Distribution of Coverage')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bbox_coverage_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to: {output_dir / 'bbox_coverage_analysis.png'}")

def main():
    """Main analysis function"""
    from config import DATA_DIR, OUTPUT_DIR, DATA_CONFIG
    
    print("Analyzing bounding box coverage...")
    
    # Get max_chunk from config
    max_chunk = DATA_CONFIG.get('max_chunk', 7)
    print(f"Using chunks 1-{max_chunk} (as configured in config.py)")
    
    # Run analysis
    analysis_results = analyze_bbox_coverage(DATA_DIR, max_chunk=max_chunk)
    
    # Print statistics
    print_statistics(analysis_results)
    
    # Create plots
    plots_dir = OUTPUT_DIR / "analysis"
    create_coverage_plots(analysis_results, plots_dir)
    
    # Save detailed results
    results_file = plots_dir / "bbox_coverage_results.json"
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main() 