#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time
import json
from typing import List, Tuple, Dict, Any
import logging

from model import create_model
from config import MODEL_CONFIG, OUTPUT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WindowedEvaluator:
    def __init__(self, model_path: Path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        self.model = create_model(MODEL_CONFIG)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)  # Move model to device
        self.model.eval()
        
        # Sliding window parameters (from inference strategy)
        self.window_size = 500  # pixels
        self.stride = 100       # pixels
        self.model_input_size = 224  # pixels
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Window size: {self.window_size}x{self.window_size}")
        logger.info(f"Stride: {self.stride} pixels")
        logger.info(f"Model input size: {self.model_input_size}x{self.model_input_size}")
    
    def extract_windows(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Extract sliding windows from image with their positions"""
        height, width = image.shape[:2]
        windows = []
        positions = []
        
        # If image is smaller than window size, use the whole image as one window
        if height < self.window_size or width < self.window_size:
            # Pad the image to window size or use the whole image
            if height < self.window_size and width < self.window_size:
                # Both dimensions are small, use whole image
                windows.append(image)
                positions.append((0, 0))
            else:
                # One dimension is small, create windows along the larger dimension
                if height < self.window_size:
                    # Height is small, slide horizontally
                    for x in range(0, width - self.window_size + 1, self.stride):
                        window = image[:, x:x + self.window_size]
                        windows.append(window)
                        positions.append((x, 0))
                else:
                    # Width is small, slide vertically
                    for y in range(0, height - self.window_size + 1, self.stride):
                        window = image[y:y + self.window_size, :]
                        windows.append(window)
                        positions.append((0, y))
        else:
            # Normal sliding window extraction
            for y in range(0, height - self.window_size + 1, self.stride):
                for x in range(0, width - self.window_size + 1, self.stride):
                    # Extract window
                    window = image[y:y + self.window_size, x:x + self.window_size]
                    windows.append(window)
                    positions.append((x, y))
        
        logger.info(f"Extracted {len(windows)} windows from {width}x{height} image")
        return windows, positions
    
    def preprocess_window(self, window: np.ndarray) -> torch.Tensor:
        """Preprocess a window for model inference"""
        # Resize to model input size
        resized = cv2.resize(window, (self.model_input_size, self.model_input_size))
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict_window(self, window_tensor: torch.Tensor) -> float:
        """Run model inference on a single window"""
        with torch.no_grad():
            window_tensor = window_tensor.to(self.device)
            output = self.model(window_tensor)
            
            # Get probability of tick class
            probabilities = torch.softmax(output['logits'], dim=1)
            tick_probability = probabilities[0, 1].item()  # Class 1 = tick
            
            return tick_probability
    
    def create_heatmap(self, image: np.ndarray, windows: List[np.ndarray], 
                      positions: List[Tuple[int, int]], predictions: List[float]) -> np.ndarray:
        """Create a heatmap visualization of predictions"""
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate predictions for each pixel
        for window, (x, y), pred in zip(windows, positions, predictions):
            # Get actual window dimensions
            window_height, window_width = window.shape[:2]
            
            # Create a mask for this window
            mask = np.ones((window_height, window_width), dtype=np.float32)
            
            # Add prediction to heatmap
            heatmap[y:y + window_height, x:x + window_width] += pred * mask
            count_map[y:y + window_height, x:x + window_width] += mask
        
        # Average overlapping regions
        heatmap = np.divide(heatmap, count_map, out=np.zeros_like(heatmap), where=count_map != 0)
        
        return heatmap
    
    def visualize_results(self, image: np.ndarray, heatmap: np.ndarray, 
                         predictions: List[float], output_path: Path):
        """Create visualization of results"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im = axes[1].imshow(heatmap, cmap='RdYlGn_r', vmin=0, vmax=1)
        axes[1].set_title('Tick Probability Heatmap')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Prediction distribution
        axes[2].hist(predictions, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[2].set_title('Prediction Distribution')
        axes[2].set_xlabel('Tick Probability')
        axes[2].set_ylabel('Number of Windows')
        axes[2].axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
    
    def evaluate_image(self, image_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Evaluate a single image using sliding window approach"""
        logger.info(f"Processing {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Extract windows
        windows, positions = self.extract_windows(image)
        
        # Process each window
        predictions = []
        start_time = time.time()
        
        for i, window in enumerate(windows):
            # Preprocess window
            window_tensor = self.preprocess_window(window)
            
            # Run inference
            prediction = self.predict_window(window_tensor)
            predictions.append(prediction)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(windows)} windows")
        
        processing_time = time.time() - start_time
        
        # Create heatmap
        heatmap = self.create_heatmap(image, windows, positions, predictions)
        
        # Generate visualization
        viz_path = output_dir / f"{image_path.stem}_windowed_eval.png"
        self.visualize_results(image, heatmap, predictions, viz_path)
        
        # Calculate metrics
        predictions_array = np.array(predictions)
        metrics = {
            'image_name': image_path.name,
            'image_size': f"{image.shape[1]}x{image.shape[0]}",
            'num_windows': len(windows),
            'processing_time_seconds': processing_time,
            'windows_per_second': len(windows) / processing_time,
            'mean_prediction': float(np.mean(predictions_array)),
            'std_prediction': float(np.std(predictions_array)),
            'max_prediction': float(np.max(predictions_array)),
            'min_prediction': float(np.min(predictions_array)),
            'windows_above_threshold': int(np.sum(predictions_array > 0.5)),
            'windows_above_threshold_pct': float(np.mean(predictions_array > 0.5) * 100),
            'heatmap_path': str(viz_path)
        }
        
        logger.info(f"Results for {image_path.name}:")
        logger.info(f"  Processing time: {processing_time:.2f}s ({len(windows)/processing_time:.1f} windows/s)")
        logger.info(f"  Mean prediction: {metrics['mean_prediction']:.3f}")
        logger.info(f"  Windows above threshold: {metrics['windows_above_threshold']}/{len(windows)} ({metrics['windows_above_threshold_pct']:.1f}%)")
        
        return metrics
    
    def evaluate_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Evaluate all images in a directory with proper classification metrics"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if this is a structured evaluation directory
        with_ticks_dir = input_dir / "with_ticks"
        without_ticks_dir = input_dir / "without_ticks"
        
        if with_ticks_dir.exists() and without_ticks_dir.exists():
            logger.info("Found structured evaluation directory with 'with_ticks' and 'without_ticks' subdirectories")
            return self._evaluate_structured_directory(input_dir, output_dir)
        else:
            logger.info("Treating as single directory evaluation")
            return self._evaluate_single_directory(input_dir, output_dir)
    
    def _evaluate_structured_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Evaluate structured directory with separate positive/negative samples"""
        with_ticks_dir = input_dir / "with_ticks"
        without_ticks_dir = input_dir / "without_ticks"
        
        # Find all image files in both directories
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        positive_files = []
        negative_files = []
        
        for ext in image_extensions:
            positive_files.extend(with_ticks_dir.rglob(f'*{ext}'))
            positive_files.extend(with_ticks_dir.rglob(f'*{ext.upper()}'))
            negative_files.extend(without_ticks_dir.rglob(f'*{ext}'))
            negative_files.extend(without_ticks_dir.rglob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(positive_files)} positive samples (with ticks)")
        logger.info(f"Found {len(negative_files)} negative samples (without ticks)")
        
        # Evaluate positive samples
        positive_results = []
        positive_output_dir = output_dir / "with_ticks"
        positive_output_dir.mkdir(exist_ok=True)
        
        logger.info("Evaluating positive samples...")
        for i, image_path in enumerate(positive_files):
            try:
                metrics = self.evaluate_image(image_path, positive_output_dir)
                metrics['true_label'] = 1  # Positive
                positive_results.append(metrics)
                logger.info(f"Completed positive {i+1}/{len(positive_files)}: {image_path.name}")
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                continue
        
        # Evaluate negative samples
        negative_results = []
        negative_output_dir = output_dir / "without_ticks"
        negative_output_dir.mkdir(exist_ok=True)
        
        logger.info("Evaluating negative samples...")
        for i, image_path in enumerate(negative_files):
            try:
                metrics = self.evaluate_image(image_path, negative_output_dir)
                metrics['true_label'] = 0  # Negative
                negative_results.append(metrics)
                logger.info(f"Completed negative {i+1}/{len(negative_files)}: {image_path.name}")
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                continue
        
        # Calculate classification metrics
        all_results = positive_results + negative_results
        
        # For each image, determine if it's classified as positive (any window above threshold)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        for result in all_results:
            is_predicted_positive = result['windows_above_threshold'] > 0
            is_actually_positive = result['true_label'] == 1
            
            if is_predicted_positive and is_actually_positive:
                true_positives += 1
            elif is_predicted_positive and not is_actually_positive:
                false_positives += 1
            elif not is_predicted_positive and not is_actually_positive:
                true_negatives += 1
            else:  # not predicted positive but actually positive
                false_negatives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(all_results) if len(all_results) > 0 else 0
        
        # Summary statistics
        summary = {
            'total_images': len(all_results),
            'positive_samples': len(positive_results),
            'negative_samples': len(negative_results),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'avg_processing_time_per_image': np.mean([m['processing_time_seconds'] for m in all_results]),
            'avg_windows_per_second': np.mean([m['windows_per_second'] for m in all_results]),
            'avg_mean_prediction_positive': np.mean([m['mean_prediction'] for m in positive_results]) if positive_results else 0,
            'avg_mean_prediction_negative': np.mean([m['mean_prediction'] for m in negative_results]) if negative_results else 0
        }
        
        logger.info("\n" + "="*60)
        logger.info("CLASSIFICATION EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Total images: {summary['total_images']}")
        logger.info(f"Positive samples: {summary['positive_samples']}")
        logger.info(f"Negative samples: {summary['negative_samples']}")
        logger.info(f"True Positives: {true_positives}")
        logger.info(f"False Positives: {false_positives}")
        logger.info(f"True Negatives: {true_negatives}")
        logger.info(f"False Negatives: {false_negatives}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1-Score: {f1_score:.3f}")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Average processing time per image: {summary['avg_processing_time_per_image']:.2f}s")
        logger.info(f"Average windows per second: {summary['avg_windows_per_second']:.1f}")
        logger.info(f"Average prediction (positive): {summary['avg_mean_prediction_positive']:.3f}")
        logger.info(f"Average prediction (negative): {summary['avg_mean_prediction_negative']:.3f}")
        
        # Save detailed results
        results_path = output_dir / "classification_evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'summary': summary,
                'positive_results': positive_results,
                'negative_results': negative_results
            }, f, indent=2)
        
        logger.info(f"Detailed results saved to {results_path}")
        
        return summary
    
    def _evaluate_single_directory(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Evaluate single directory without structured labels"""
        # Find all image files (recursively)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.rglob(f'*{ext}'))
            image_files.extend(input_dir.rglob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {}
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        all_metrics = []
        total_start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            try:
                metrics = self.evaluate_image(image_path, output_dir)
                all_metrics.append(metrics)
                logger.info(f"Completed {i+1}/{len(image_files)}: {image_path.name}")
            except Exception as e:
                logger.error(f"Error processing {image_path.name}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Calculate summary statistics
        if all_metrics:
            summary = {
                'total_images': len(all_metrics),
                'total_processing_time': total_time,
                'avg_processing_time_per_image': total_time / len(all_metrics),
                'avg_windows_per_second': np.mean([m['windows_per_second'] for m in all_metrics]),
                'avg_mean_prediction': np.mean([m['mean_prediction'] for m in all_metrics]),
                'avg_windows_above_threshold_pct': np.mean([m['windows_above_threshold_pct'] for m in all_metrics])
            }
            
            logger.info("\n" + "="*50)
            logger.info("SUMMARY STATISTICS")
            logger.info("="*50)
            logger.info(f"Total images processed: {summary['total_images']}")
            logger.info(f"Total processing time: {summary['total_processing_time']:.2f}s")
            logger.info(f"Average time per image: {summary['avg_processing_time_per_image']:.2f}s")
            logger.info(f"Average windows per second: {summary['avg_windows_per_second']:.1f}")
            logger.info(f"Average mean prediction: {summary['avg_mean_prediction']:.3f}")
            logger.info(f"Average windows above threshold: {summary['avg_windows_above_threshold_pct']:.1f}%")
            
            # Save detailed results
            results_path = output_dir / "windowed_evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'detailed_results': all_metrics
                }, f, indent=2)
            
            logger.info(f"Detailed results saved to {results_path}")
        
        return summary


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate mobile model using sliding window approach')
    parser.add_argument('--model', type=str, default='../outputs/mobile/checkpoints/best_f1_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, default='../test_images',
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='../outputs/mobile/windowed_eval',
                       help='Path to output directory')
    args = parser.parse_args()
    
    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        return
    
    # Create evaluator
    evaluator = WindowedEvaluator(model_path)
    
    # Run evaluation
    if input_path.is_file():
        # Single image
        evaluator.evaluate_image(input_path, output_path)
    else:
        # Directory of images
        evaluator.evaluate_directory(input_path, output_path)


if __name__ == "__main__":
    main() 