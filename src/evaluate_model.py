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
from torchvision import models
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
        config_path = os.path.join('config', 'train_config.yaml')
        self.config = load_config(config_path)
        
        # Load checkpoint first to get model structure
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Create model with matching architecture
        self.model = models.detection.retinanet_resnet50_fpn(
            num_classes=91,  # COCO classes
            pretrained=False
        )
        
        # Remove the extra 'model.' prefix from keys
        state_dict = {k.replace('model.model.', 'model.'): v for k, v in state_dict.items()}
        
        # Load state dict with strict=False to handle missing/extra keys
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transform(inference_only=True, config=self.config)
        
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
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Debug print
        print(f"\nPredictions for {image_path}:")
        print(f"Boxes shape: {predictions[0]['boxes'].shape}")
        print(f"Scores shape: {predictions[0]['scores'].shape}")
        print(f"Labels shape: {predictions[0]['labels'].shape}")
        print(f"Max score: {predictions[0]['scores'].max().item():.3f}")
        
        return predictions[0]  # Remove batch dimension
        
    def evaluate_directory(self, input_dir, output_dir):
        """Evaluate all images in a directory."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Process each image
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file)
            
            # Get predictions
            predictions = self.predict_image(image_path)
            
            # Store predictions for metrics
            self.all_predictions.append({
                'boxes': predictions['boxes'].cpu().numpy(),
                'scores': predictions['scores'].cpu().numpy(),
                'labels': predictions['labels'].cpu().numpy()
            })
            
            # Draw predictions on image
            self.visualize_predictions(image_path, predictions, output_path)
        
        # Calculate and save metrics
        self.save_evaluation_report(output_dir)

    def visualize_predictions(self, image_path, predictions, output_path):
        """Draw bounding boxes on image and save."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Filter predictions by confidence
        mask = predictions['scores'] >= self.confidence_threshold
        boxes = predictions['boxes'][mask].cpu().numpy()
        scores = predictions['scores'][mask].cpu().numpy()
        
        # Draw each box
        for box, score in zip(boxes, scores):
            # Convert box to integers
            box = [int(x) for x in box]
            
            # Draw rectangle
            draw.rectangle(box, outline='red', width=2)
            
            # Draw score
            draw.text((box[0], box[1] - 10), f'{score:.2f}', fill='red')
        
        # Save image
        image.save(output_path)

    def save_evaluation_report(self, output_dir):
        """Calculate and save evaluation metrics."""
        # Calculate metrics
        total_images = len(self.all_predictions)
        total_detections = sum(len(p['boxes']) for p in self.all_predictions)
        avg_detections = total_detections / total_images if total_images > 0 else 0
        
        # Calculate detection distribution
        detection_ranges = {
            '0-1': 0,
            '2-5': 0,
            '6-10': 0,
            '>10': 0
        }
        
        for pred in self.all_predictions:
            num_detections = len(pred['boxes'])
            if num_detections <= 1:
                detection_ranges['0-1'] += 1
            elif num_detections <= 5:
                detection_ranges['2-5'] += 1
            elif num_detections <= 10:
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
        
        for pred in self.all_predictions:
            for score in pred['scores']:
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