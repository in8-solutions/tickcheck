"""
This script evaluates an object detection model on test images. It's designed to help understand:
1. How ML models perform on real-world data
2. The importance of comprehensive testing
3. The relationship between confidence thresholds and model predictions
4. How to visualize and interpret model outputs

Key ML Concepts Demonstrated:
- Model inference (forward pass without training)
- Confidence thresholds for predictions
- Post-processing of model outputs
- Evaluation metrics and visualization
- Handling real-world input data

Modern Deep Learning Evolution:
- While this example uses CNNs for spatial data (images), similar principles apply
  to temporal data (time series, network protocols, etc.)
- Traditional approaches like BDTs worked well for structured temporal features
  but struggled with raw sequential data
- Modern approaches for temporal data include:
  * Transformers: Great for capturing long-range dependencies in sequences
  * LSTMs/GRUs: Better than traditional RNNs for avoiding vanishing gradients
  * Temporal CNNs: 1D convolutions for sequence processing
  * Attention mechanisms: Help focus on relevant parts of sequences
- Key differences from image processing:
  * Temporal locality vs spatial locality
  * Variable sequence lengths vs fixed image sizes
  * Causal vs non-causal processing (future data availability)
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
from pathlib import Path
import argparse
from utils import load_config
from transforms import get_transform
from model import create_model
from torchvision.transforms import functional as F

class ModelEvaluator:
    """
    A class to evaluate an object detection model on images.
    
    This class demonstrates several important ML concepts:
    1. Model Loading: How to load a trained model from a checkpoint
    2. Inference: How to run the model on new images
    3. Post-processing: How to interpret and filter model outputs
    4. Visualization: How to display model predictions
    
    Note on Model Architecture Choices:
    - CNNs excel at spatial feature extraction (like our tick detection)
    - For temporal data (like network protocols), consider:
      * Sequence length requirements
      * Real-time vs batch processing needs
      * Feature correlation across time steps
      * Causality requirements (future data availability)
    """
    
    def __init__(self, model_path, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load configuration for transforms
        config_path = 'config.yaml'  # Updated to use top-level config
        self.config = load_config(config_path)
        
        # Create model with our configuration
        self.model = create_model(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transform(self.config, train=False, inference_only=True)
        
        # Initialize metrics storage
        self.all_predictions = []
        self.all_targets = []
    
    def predict_image(self, image_path):
        """Run prediction on a single image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)  # Convert PIL Image to numpy array
        
        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions - pass as a list of images
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        return predictions[0]  # Remove batch dimension
        
    def evaluate_directory(self, input_dir, output_dir):
        """Evaluate model on all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_detections = 0
        detection_counts = []
        confidence_scores = []
        
        # Process images with progress bar
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, f"pred_{image_file}")
            
            # Get predictions
            predictions = self.predict_image(image_path)
            boxes = predictions['boxes']
            scores = predictions['scores']
            labels = predictions['labels']
            
            # Update statistics
            total_detections += len(boxes)
            detection_counts.append(len(boxes))
            if isinstance(scores, torch.Tensor):
                confidence_scores.extend(scores.cpu().numpy().tolist())
            else:
                confidence_scores.extend(scores)
            
            # Save visualization
            self.visualize_predictions(image_path, boxes, scores, labels, output_path)
        
        # Generate and save report
        self.generate_report(
            total_images=len(image_files),
            total_detections=total_detections,
            detection_counts=detection_counts,
            confidence_scores=confidence_scores,
            output_dir=output_dir
        )

    def visualize_predictions(self, image_path, boxes, scores, labels, output_path):
        """Draw bounding boxes on image and save."""
        # Load image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Convert scores to numpy array if it's a tensor
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # Filter predictions by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask].cpu().numpy()
        scores = scores[mask]
        
        # Draw each box
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            draw.text((x1, y1-10), f'{score:.2f}', fill='red')
        
        # Save image
        image.save(output_path)

    def generate_report(self, total_images, total_detections, detection_counts, confidence_scores, output_dir):
        """Calculate and save evaluation metrics."""
        # Calculate metrics
        avg_detections = total_detections / total_images if total_images > 0 else 0
        
        # Calculate detection distribution
        detection_ranges = {
            '0-1': 0,
            '2-5': 0,
            '6-10': 0,
            '>10': 0
        }
        
        for count in detection_counts:
            if count <= 1:
                detection_ranges['0-1'] += 1
            elif count <= 5:
                detection_ranges['2-5'] += 1
            elif count <= 10:
                detection_ranges['6-10'] += 1
            else:
                detection_ranges['>10'] += 1
        
        # Calculate confidence distribution
        confidence_ranges = {
            '0.5-0.6': 0,
            '0.6-0.7': 0,
            '0.7-0.8': 0,
            '0.8-0.9': 0,
            '0.9-1.0': 0
        }
        
        for score in confidence_scores:
            if score >= 0.5:
                if score < 0.6:
                    confidence_ranges['0.5-0.6'] += 1
                elif score < 0.7:
                    confidence_ranges['0.6-0.7'] += 1
                elif score < 0.8:
                    confidence_ranges['0.7-0.8'] += 1
                elif score < 0.9:
                    confidence_ranges['0.8-0.9'] += 1
                else:
                    confidence_ranges['0.9-1.0'] += 1
        
        # Save metrics to JSON file
        metrics = {
            'confidence_threshold': self.confidence_threshold,
            'total_images': total_images,
            'total_detections': total_detections,
            'average_detections_per_image': avg_detections,
            'detection_distribution': detection_ranges,
            'confidence_distribution': confidence_ranges
        }
        
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print human-readable report
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(f"\nModel Evaluation Summary:")
        print(f"Confidence Threshold: {self.confidence_threshold:.2f}")
        print(f"Total Images Processed: {total_images}")
        print(f"Total Detections: {total_detections}")
        print(f"Average Detections per Image: {avg_detections:.2f}")
        
        print("\nDetection Distribution:")
        for range_name, count in detection_ranges.items():
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"  {range_name} detections: {count} images ({percentage:.1f}%)")
        
        print("\nConfidence Distribution:")
        for range_name, count in confidence_ranges.items():
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"  {range_name}: {count} detections ({percentage:.1f}%)")
        
        print("\n" + "="*50)
        print(f"Full report saved to: {report_path}")
        print("="*50 + "\n")

def evaluate_directory(model_path, input_dir, output_dir, confidence_threshold=0.5):
    """Evaluate a directory of images."""
    evaluator = ModelEvaluator(model_path, confidence_threshold=confidence_threshold)
    evaluator.evaluate_directory(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on images")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--input", required=True, help="Path to input directory containing images")
    parser.add_argument("--output", required=True, help="Path to output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    evaluate_directory(args.model, args.input, args.output, args.threshold) 