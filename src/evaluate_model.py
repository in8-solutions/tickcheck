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
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class ModelEvaluator:
    """
    A class to evaluate an object detection model on images.
    
    This class handles binary classification (tick vs background) and provides
    separate metrics for images with and without ticks.
    """
    
    def __init__(self, model_path, confidence_threshold=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration for transforms
        config_path = 'config.yaml'
        self.config = load_config(config_path)
        
        # Use model's box_score_thresh if no confidence threshold provided
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.config['model']['box_score_thresh']
        
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
                'total_images': 0,
                'total_detections': 0,
                'detection_counts': [],
                'confidence_scores': [],
                'true_positives': 0,
                'false_negatives': 0
            },
            'without_ticks': {
                'total_images': 0,
                'total_detections': 0,
                'detection_counts': [],
                'confidence_scores': [],
                'false_positives': 0,
                'true_negatives': 0
            }
        }
    
    def predict_image(self, image_path):
        """Run prediction on a single image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model([image_tensor])
        
        return predictions[0]
    
    def evaluate_directory(self, input_dir, output_dir):
        """Evaluate model on images in both 'with_ticks' and 'without_ticks' subdirectories."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each category
        for category in ['with_ticks', 'without_ticks']:
            category_dir = os.path.join(input_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: {category_dir} not found, skipping...")
                continue
                
            category_output_dir = os.path.join(output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)
            
            # Get all images in category
            image_files = []
            for root, _, files in os.walk(category_dir):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, f))
            
            print(f"\nProcessing {category} images...")
            for image_path in tqdm(image_files, desc=f"Processing {category}"):
                # Get relative path for output
                rel_path = os.path.relpath(image_path, category_dir)
                output_path = os.path.join(category_output_dir, f"pred_{rel_path}")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Get predictions
                predictions = self.predict_image(image_path)
                boxes = predictions['boxes']
                scores = predictions['scores']
                labels = predictions['labels']
                
                # Update category metrics
                self.metrics[category]['total_images'] += 1
                num_detections = len(boxes)
                self.metrics[category]['total_detections'] += num_detections
                self.metrics[category]['detection_counts'].append(num_detections)
                
                if isinstance(scores, torch.Tensor):
                    self.metrics[category]['confidence_scores'].extend(scores.cpu().numpy().tolist())
                else:
                    self.metrics[category]['confidence_scores'].extend(scores)
                
                # Update classification metrics
                if category == 'with_ticks':
                    if num_detections > 0:
                        self.metrics[category]['true_positives'] += 1
                    else:
                        self.metrics[category]['false_negatives'] += 1
                else:  # without_ticks
                    if num_detections > 0:
                        self.metrics[category]['false_positives'] += 1
                    else:
                        self.metrics[category]['true_negatives'] += 1
                
                # Save visualization
                self.visualize_predictions(image_path, boxes, scores, labels, output_path)
        
        # Generate and save comprehensive report
        self.generate_report(output_dir)
    
    def visualize_predictions(self, image_path, boxes, scores, labels, output_path):
        """Draw bounding boxes on image and save."""
        # Load original image to get dimensions
        image = Image.open(image_path)
        orig_width, orig_height = image.size
        
        # Calculate scaling factors
        scale_x = orig_width / self.config['data']['input_size'][1]
        scale_y = orig_height / self.config['data']['input_size'][0]
        
        draw = ImageDraw.Draw(image)
        
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # Filter predictions by model's minimum threshold
        min_threshold_mask = scores >= self.config['model']['box_score_thresh']
        boxes = boxes[min_threshold_mask].cpu().numpy()
        scores = scores[min_threshold_mask]
        
        # Scale boxes back to original image size
        boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
        
        # Draw each box
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            # Show yellow boxes for detections within 0.2 of threshold
            if score >= self.confidence_threshold:
                color = 'red'
            elif score >= (self.confidence_threshold - 0.2):
                color = 'yellow'
            else:
                continue  # Skip boxes below this range
                
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1-10), f'{score:.2f}', fill=color)
        
        image.save(output_path)
    
    def generate_report(self, output_dir):
        """Generate comprehensive evaluation report with binary classification metrics."""
        # Calculate overall metrics
        total_images = (self.metrics['with_ticks']['total_images'] + 
                       self.metrics['without_ticks']['total_images'])
        
        # Calculate precision, recall, and F1 score
        tp = self.metrics['with_ticks']['true_positives']
        fp = self.metrics['without_ticks']['false_positives']
        fn = self.metrics['with_ticks']['false_negatives']
        tn = self.metrics['without_ticks']['true_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate detection distributions
        for category in ['with_ticks', 'without_ticks']:
            detection_ranges = {
                '0-1': 0,
                '2-5': 0,
                '6-10': 0,
                '>10': 0
            }
            
            for count in self.metrics[category]['detection_counts']:
                if count <= 1:
                    detection_ranges['0-1'] += 1
                elif count <= 5:
                    detection_ranges['2-5'] += 1
                elif count <= 10:
                    detection_ranges['6-10'] += 1
                else:
                    detection_ranges['>10'] += 1
            
            self.metrics[category]['detection_distribution'] = detection_ranges
        
        # Calculate confidence distributions
        for category in ['with_ticks', 'without_ticks']:
            confidence_ranges = {
                '0.5-0.6': 0,
                '0.6-0.7': 0,
                '0.7-0.8': 0,
                '0.8-0.9': 0,
                '0.9-1.0': 0
            }
            
            for score in self.metrics[category]['confidence_scores']:
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
            
            self.metrics[category]['confidence_distribution'] = confidence_ranges
        
        # Prepare final report
        report = {
            'confidence_threshold': self.confidence_threshold,
            'binary_classification_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': {
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                }
            },
            'with_ticks': {
                'total_images': self.metrics['with_ticks']['total_images'],
                'total_detections': self.metrics['with_ticks']['total_detections'],
                'average_detections': (self.metrics['with_ticks']['total_detections'] / 
                                     self.metrics['with_ticks']['total_images'] 
                                     if self.metrics['with_ticks']['total_images'] > 0 else 0),
                'detection_distribution': self.metrics['with_ticks']['detection_distribution'],
                'confidence_distribution': self.metrics['with_ticks']['confidence_distribution']
            },
            'without_ticks': {
                'total_images': self.metrics['without_ticks']['total_images'],
                'total_detections': self.metrics['without_ticks']['total_detections'],
                'average_detections': (self.metrics['without_ticks']['total_detections'] / 
                                     self.metrics['without_ticks']['total_images']
                                     if self.metrics['without_ticks']['total_images'] > 0 else 0),
                'detection_distribution': self.metrics['without_ticks']['detection_distribution'],
                'confidence_distribution': self.metrics['without_ticks']['confidence_distribution']
            }
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print human-readable report
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        
        print("\nBinary Classification Metrics:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print("\nConfusion Matrix:")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives: {tn}")
        print(f"False Negatives: {fn}")
        
        for category in ['with_ticks', 'without_ticks']:
            print(f"\n{category.replace('_', ' ').title()} Metrics:")
            print(f"Total Images: {self.metrics[category]['total_images']}")
            print(f"Total Detections: {self.metrics[category]['total_detections']}")
            print(f"Average Detections per Image: {report[category]['average_detections']:.2f}")
            
            print("\nDetection Distribution:")
            for range_name, count in self.metrics[category]['detection_distribution'].items():
                percentage = (count / self.metrics[category]['total_images'] * 100 
                            if self.metrics[category]['total_images'] > 0 else 0)
                print(f"  {range_name} detections: {count} images ({percentage:.1f}%)")
            
            print("\nConfidence Distribution:")
            for range_name, count in self.metrics[category]['confidence_distribution'].items():
                percentage = (count / self.metrics[category]['total_detections'] * 100 
                            if self.metrics[category]['total_detections'] > 0 else 0)
                print(f"  {range_name}: {count} detections ({percentage:.1f}%)")
        
        print("\n" + "="*50)
        print(f"Full report saved to: {report_path}")
        print("="*50 + "\n")

def evaluate_directory(model_path, input_dir, output_dir, confidence_threshold=None):
    """Evaluate a directory of images with separate 'with_ticks' and 'without_ticks' subdirectories."""
    evaluator = ModelEvaluator(model_path, confidence_threshold=confidence_threshold)
    evaluator.evaluate_directory(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on images")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--input", required=True, help="Path to input directory containing 'with_ticks' and 'without_ticks' subdirectories")
    parser.add_argument("--output", required=True, help="Path to output directory for results")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold for detections")
    
    args = parser.parse_args()
    evaluate_directory(args.model, args.input, args.output, args.confidence) 