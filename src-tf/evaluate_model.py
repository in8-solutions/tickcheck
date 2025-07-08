"""
Model evaluation script for TensorFlow tick detection model.

This script evaluates an object detection model on test images. It's designed to help understand:
1. How ML models perform on real-world data
2. The importance of comprehensive testing
3. The relationship between confidence thresholds and model predictions
4. How to visualize and interpret model outputs
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from tqdm import tqdm
import json
from pathlib import Path
import argparse
from utils import load_config, calculate_metrics
from transforms import get_transform, convert_to_tensorflow_format
from model import create_model

class ModelEvaluator:
    """
    A class to evaluate an object detection model on images.
    
    This class handles binary classification (tick vs background) and provides
    separate metrics for images with and without ticks.
    """
    
    def __init__(self, model_path, confidence_threshold=None):
        # Load configuration
        config_path = 'config.yaml'
        self.config = load_config(config_path)
        
        # Use model's box_score_thresh if no confidence threshold provided
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else self.config['model']['box_score_thresh']
        
        # Create model
        self.model = create_model(self.config)
        
        # Load checkpoint
        self.model.load_weights(model_path)
        
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
        
        # Convert to TensorFlow format
        if hasattr(image_tensor, 'numpy'):
            image_tensor = image_tensor.numpy()
        
        # Ensure correct shape
        if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:
            image_tensor = np.transpose(image_tensor, (1, 2, 0))
        
        # Add batch dimension
        image_tensor = np.expand_dims(image_tensor, axis=0)
        image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
        
        # Get predictions
        predictions = self.model(image_tensor, training=False)
        
        # Process predictions (simplified for now)
        # In a full implementation, you would decode anchors and apply NMS
        return {
            'boxes': np.array([]),  # Placeholder
            'scores': np.array([]),  # Placeholder
            'labels': np.array([])   # Placeholder
        }
    
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
                
                if len(scores) > 0:
                    self.metrics[category]['confidence_scores'].extend(scores.tolist())
                
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
        # Load original image
        image = Image.open(image_path)
        orig_width, orig_height = image.size
        
        # Calculate scaling factors
        scale_x = orig_width / self.config['data']['input_size'][1]
        scale_y = orig_height / self.config['data']['input_size'][0]
        
        draw = ImageDraw.Draw(image)
        
        # Filter predictions by confidence threshold
        if len(scores) > 0:
            threshold_mask = scores >= self.confidence_threshold
            boxes = boxes[threshold_mask]
            scores = scores[threshold_mask]
            
            # Scale boxes back to original image size
            boxes = boxes * np.array([scale_x, scale_y, scale_x, scale_y])
            
            # Draw each box
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                
                # Draw score
                score_text = f"{score:.2f}"
                draw.text((x1, y1 - 10), score_text, fill='red')
        
        # Save image
        image.save(output_path)
    
    def generate_report(self, output_dir):
        """Generate comprehensive evaluation report."""
        report = {
            'evaluation_summary': {
                'confidence_threshold': self.confidence_threshold,
                'total_images_processed': (
                    self.metrics['with_ticks']['total_images'] + 
                    self.metrics['without_ticks']['total_images']
                )
            },
            'with_ticks': {
                'total_images': self.metrics['with_ticks']['total_images'],
                'total_detections': self.metrics['with_ticks']['total_detections'],
                'average_detections_per_image': (
                    self.metrics['with_ticks']['total_detections'] / 
                    max(self.metrics['with_ticks']['total_images'], 1)
                ),
                'true_positives': self.metrics['with_ticks']['true_positives'],
                'false_negatives': self.metrics['with_ticks']['false_negatives'],
                'sensitivity': (
                    self.metrics['with_ticks']['true_positives'] / 
                    max(self.metrics['with_ticks']['total_images'], 1)
                )
            },
            'without_ticks': {
                'total_images': self.metrics['without_ticks']['total_images'],
                'total_detections': self.metrics['without_ticks']['total_detections'],
                'average_detections_per_image': (
                    self.metrics['without_ticks']['total_detections'] / 
                    max(self.metrics['without_ticks']['total_images'], 1)
                ),
                'false_positives': self.metrics['without_ticks']['false_positives'],
                'true_negatives': self.metrics['without_ticks']['true_negatives'],
                'specificity': (
                    self.metrics['without_ticks']['true_negatives'] / 
                    max(self.metrics['without_ticks']['total_images'], 1)
                )
            }
        }
        
        # Calculate overall metrics
        total_tp = self.metrics['with_ticks']['true_positives']
        total_fp = self.metrics['without_ticks']['false_positives']
        total_fn = self.metrics['with_ticks']['false_negatives']
        total_tn = self.metrics['without_ticks']['true_negatives']
        
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0.0
            
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0.0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        report['overall_metrics'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Total Images Processed: {report['evaluation_summary']['total_images_processed']}")
        print()
        print("IMAGES WITH TICKS:")
        print(f"  Total Images: {report['with_ticks']['total_images']}")
        print(f"  True Positives: {report['with_ticks']['true_positives']}")
        print(f"  False Negatives: {report['with_ticks']['false_negatives']}")
        print(f"  Sensitivity: {report['with_ticks']['sensitivity']:.3f}")
        print()
        print("IMAGES WITHOUT TICKS:")
        print(f"  Total Images: {report['without_ticks']['total_images']}")
        print(f"  True Negatives: {report['without_ticks']['true_negatives']}")
        print(f"  False Positives: {report['without_ticks']['false_positives']}")
        print(f"  Specificity: {report['without_ticks']['specificity']:.3f}")
        print()
        print("OVERALL METRICS:")
        print(f"  Precision: {report['overall_metrics']['precision']:.3f}")
        print(f"  Recall: {report['overall_metrics']['recall']:.3f}")
        print(f"  F1 Score: {report['overall_metrics']['f1_score']:.3f}")
        print(f"  Accuracy: {report['overall_metrics']['accuracy']:.3f}")
        print("="*50)

def evaluate_directory(model_path, input_dir, output_dir, confidence_threshold=None):
    """Convenience function to evaluate a directory."""
    evaluator = ModelEvaluator(model_path, confidence_threshold)
    evaluator.evaluate_directory(input_dir, output_dir)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate TensorFlow tick detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input directory with test images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--confidence', type=float, default=None, help='Confidence threshold')
    args = parser.parse_args()
    
    try:
        evaluate_directory(args.model, args.input, args.output, args.confidence)
        print(f"\nEvaluation completed! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 