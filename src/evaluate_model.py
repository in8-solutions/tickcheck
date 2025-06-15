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
from dataset import DetectionDataset

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
        config_path = 'config.yaml'
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
        self.metrics = {
            'with_ticks': {
                'true_positives': 0,
                'false_negatives': 0,
                'false_positives': 0,
                'total_ground_truth': 0
            },
            'without_ticks': {
                'false_positives': 0,
                'total_images': 0
            }
        }
    
    def predict_image(self, image_path):
        """Run prediction on a single image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        img_height, img_width = image_np.shape[:2]
        
        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        # Filter predictions to only include tick class (class 1)
        pred = predictions[0]
        tick_mask = pred['labels'] == 1  # Only keep tick class
        filtered_pred = {
            'boxes': pred['boxes'][tick_mask],
            'scores': pred['scores'][tick_mask],
            'labels': pred['labels'][tick_mask]
        }
        
        # Scale box coordinates back to original image size
        scale_x = img_width / self.config['data']['input_size'][1]
        scale_y = img_height / self.config['data']['input_size'][0]
        
        filtered_pred['boxes'][:, 0] = filtered_pred['boxes'][:, 0] * scale_x
        filtered_pred['boxes'][:, 1] = filtered_pred['boxes'][:, 1] * scale_y
        filtered_pred['boxes'][:, 2] = filtered_pred['boxes'][:, 2] * scale_x
        filtered_pred['boxes'][:, 3] = filtered_pred['boxes'][:, 3] * scale_y
        
        return filtered_pred
    
    def get_image_files(self, directory):
        """Recursively get all image files from a directory and its subdirectories."""
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        return image_files
    
    def evaluate_directory(self, input_dir, output_dir, is_with_ticks=True):
        """Evaluate model on all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files recursively
        image_files = self.get_image_files(input_dir)
        
        for image_path in tqdm(image_files, desc=f"Processing {'with_ticks' if is_with_ticks else 'without_ticks'} images"):
            # Create output path preserving subdirectory structure
            rel_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(output_dir, f"pred_{rel_path}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get predictions
            predictions = self.predict_image(image_path)
            boxes = predictions['boxes']
            scores = predictions['scores']
            labels = predictions['labels']
            
            # Filter predictions by confidence
            mask = scores >= self.confidence_threshold
            boxes = boxes[mask].cpu().numpy()
            scores = scores[mask].cpu().numpy()
            
            # Update metrics
            if is_with_ticks:
                # For with_ticks images, we expect at least one detection
                self.metrics['with_ticks']['total_ground_truth'] += 1
                if len(boxes) > 0:
                    self.metrics['with_ticks']['true_positives'] += 1
                else:
                    self.metrics['with_ticks']['false_negatives'] += 1
            else:
                # For without_ticks images, any detection is a false positive
                self.metrics['without_ticks']['total_images'] += 1
                if len(boxes) > 0:
                    self.metrics['without_ticks']['false_positives'] += 1
            
            # Save visualization
            self.visualize_predictions(image_path, boxes, scores, output_path)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes."""
        # Convert to [x1, y1, x2, y2] format
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def visualize_predictions(self, image_path, boxes, scores, output_path):
        """Draw bounding boxes on image and save."""
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Convert boxes to integers and clamp to image boundaries
        boxes = boxes.astype(int)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_width)   # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_height)  # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_width)   # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_height)  # y2
        
        # Draw each box with a different color based on confidence
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            # Use different colors based on confidence
            if score > 0.7:
                color = 'red'
            elif score > 0.5:
                color = 'orange'
            else:
                color = 'yellow'
            
            # Draw box with thicker line
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            # Draw score with larger font
            draw.text((x1, y1-20), f'{score:.2f}', fill=color, font=None, font_size=24)
        
        # Save with high quality
        image.save(output_path, quality=95)
    
    def generate_report(self, output_dir):
        """Generate and save evaluation report."""
        # Calculate metrics
        with_ticks = self.metrics['with_ticks']
        without_ticks = self.metrics['without_ticks']
        
        # Calculate precision and recall for with_ticks
        precision = with_ticks['true_positives'] / (with_ticks['true_positives'] + with_ticks['false_positives']) if (with_ticks['true_positives'] + with_ticks['false_positives']) > 0 else 0
        recall = with_ticks['true_positives'] / with_ticks['total_ground_truth'] if with_ticks['total_ground_truth'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate false positive rate for without_ticks
        fpr = without_ticks['false_positives'] / without_ticks['total_images'] if without_ticks['total_images'] > 0 else 0
        
        # Generate report
        report = f"""
==================================================
EVALUATION REPORT
==================================================

Confidence Threshold: {self.confidence_threshold:.2f}

With Ticks Images:
Total Ground Truth Ticks: {with_ticks['total_ground_truth']}
True Positives: {with_ticks['true_positives']}
False Positives: {with_ticks['false_positives']}
False Negatives: {with_ticks['false_negatives']}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

Without Ticks Images:
Total Images: {without_ticks['total_images']}
False Positives: {without_ticks['false_positives']}
False Positive Rate: {fpr:.4f}
"""
        
        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Print report
        print(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, nargs='+', required=True, help='Input directories containing images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for detections')
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, args.threshold)
    
    # Process each input directory
    for input_dir in args.input:
        is_with_ticks = 'with_ticks' in input_dir
        output_dir = os.path.join(args.output, 'with_ticks' if is_with_ticks else 'without_ticks')
        evaluator.evaluate_directory(input_dir, output_dir, is_with_ticks)
    
    # Generate final report
    evaluator.generate_report(args.output)

if __name__ == '__main__':
    main() 