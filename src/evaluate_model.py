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
import cv2
import json
from pathlib import Path
from model import create_model
from utils import load_config
from dataset import get_transform
from PIL import Image
import numpy as np
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
    
    def __init__(self, model_path, config_path='config.yaml', confidence_threshold=0.5):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to model configuration
            confidence_threshold: Minimum confidence score for predictions
                                (higher = more confident but might miss detections,
                                 lower = more detections but more false positives)
                                
        Note on Thresholds:
        - In image detection, thresholds filter spatial confidence
        - In temporal analysis, thresholds might filter:
          * Sequence pattern confidence
          * Anomaly detection scores
          * State transition probabilities
        """
        self.config = load_config(config_path)
        # Choose GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Load the model - this is where we prepare for inference
        self.model = create_model(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to eval mode - important for inference!
        # This affects batch norm, dropout, and other training-specific behaviors
        # In temporal models, this might also affect:
        # - Stateful components (LSTM hidden states)
        # - Causal masking (preventing future data leakage)
        # - Sequence padding handling
        self.model.eval()
        self.model.to(self.device)
        
        # Get the same transform used during training
        # Consistency between training and testing transforms is crucial!
        # For temporal data, transforms might include:
        # - Sequence normalization
        # - Time window selection
        # - Feature scaling across time steps
        self.transform = get_transform(self.config, train=False, inference_only=True)
    
    def process_image(self, image_path):
        """
        Process a single image through the model.
        Returns detections with confidence scores above threshold.
        """
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(image)
        
        # Get boxes, scores, and labels
        boxes = predictions[0]['boxes'].cpu()
        scores = predictions[0]['scores'].cpu()
        labels = predictions[0]['labels'].cpu()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'image_size': image.shape[2:]  # Height, Width
        }
    
    def visualize_results(self, image_path, detections, output_path=None):
        """
        Draw detection boxes on the image.
        
        Visualization is crucial for:
        1. Debugging model behavior
        2. Understanding false positives/negatives
        3. Communicating results to users
        4. Identifying patterns in model mistakes
        
        For temporal data visualization:
        1. Time series plots with predictions
        2. State transition diagrams
        3. Attention weight heatmaps
        4. Feature importance over time
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Draw each detection with its confidence score
        for box, score in zip(detections['boxes'], detections['scores']):
            x1, y1, x2, y2 = map(int, box)
            
            # Green boxes for detections
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add confidence score - helps understand model certainty
            label = f"Tick: {score:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(str(output_path), image)
            print(f"Saved visualization to: {output_path}")
        
        return image

def evaluate_directory(model_path, input_dir, output_dir='outputs/evaluation', confidence_threshold=0.5):
    """
    Evaluate model on all images in a directory.
    Expects input_dir to have subdirectories:
    - with_ticks/    : Images known to contain ticks
    - without_ticks/ : Images known to be tick-free
    """
    evaluator = ModelEvaluator(model_path, confidence_threshold=confidence_threshold)
    
    # Create output directories
    vis_dir = Path(output_dir) / 'visualizations'
    results_dir = Path(output_dir) / 'results'
    vis_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results tracking
    results = {
        'with_ticks': {},
        'without_ticks': {},
        'summary': {
            'total_images': 0,
            'total_detections': 0,
            'true_positives': 0,    # Detections in with_ticks/
            'false_positives': 0,   # Detections in without_ticks/
            'false_negatives': 0,   # Images in with_ticks/ with no detections
            'true_negatives': 0,    # Images in without_ticks/ with no detections
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confidence_distribution': {
                '0.9-1.0': 0,
                '0.8-0.9': 0,
                '0.7-0.8': 0,
                '0.6-0.7': 0,
                '0.5-0.6': 0
            }
        }
    }
    
    input_dir = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Process images by category
    for category in ['with_ticks', 'without_ticks']:
        category_dir = input_dir / category
        if not category_dir.exists():
            print(f"Warning: {category_dir} not found!")
            continue
            
        for image_path in category_dir.glob('**/*'):
            if image_path.suffix.lower() not in image_extensions:
                continue
            
            print(f"\nProcessing: {image_path}")
            results['summary']['total_images'] += 1
            
            # Run detection
            detections = evaluator.process_image(image_path)
            if detections is None:
                continue
                
            num_detections = len(detections['boxes'])
            results['summary']['total_detections'] += num_detections
            
            # Update confidence distribution
            for score in detections['scores']:
                score = float(score)
                for range_str in results['summary']['confidence_distribution']:
                    low, high = map(float, range_str.split('-'))
                    if low <= score < high:
                        results['summary']['confidence_distribution'][range_str] += 1
            
            # Save detailed results as JSON
            result_file = results_dir / f"{image_path.stem}_detections.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'boxes': detections['boxes'].tolist(),
                    'scores': detections['scores'].tolist(),
                    'labels': detections['labels'].tolist(),
                    'image_size': detections['image_size']
                }, f, indent=4)
            
            # Create and save visualization
            vis_file = vis_dir / f"{image_path.stem}_detected.jpg"
            evaluator.visualize_results(image_path, detections, vis_file)
            
            # Store results and update metrics
            results[category][image_path.name] = {
                'num_detections': num_detections,
                'scores': [float(s) for s in detections['scores']],
                'detections_file': str(result_file),
                'visualization_file': str(vis_file)
            }
            
            # Update true/false positives/negatives
            if category == 'with_ticks':
                if num_detections > 0:
                    results['summary']['true_positives'] += 1
                else:
                    results['summary']['false_negatives'] += 1
            else:  # without_ticks
                if num_detections > 0:
                    results['summary']['false_positives'] += 1
                else:
                    results['summary']['true_negatives'] += 1
    
    # Calculate final metrics
    total_predicted_positive = results['summary']['true_positives'] + results['summary']['false_positives']
    total_actual_positive = results['summary']['true_positives'] + results['summary']['false_negatives']
    
    if total_predicted_positive > 0:
        results['summary']['precision'] = results['summary']['true_positives'] / total_predicted_positive
    
    if total_actual_positive > 0:
        results['summary']['recall'] = results['summary']['true_positives'] / total_actual_positive
    
    if results['summary']['precision'] + results['summary']['recall'] > 0:
        results['summary']['f1_score'] = 2 * (results['summary']['precision'] * results['summary']['recall']) / (results['summary']['precision'] + results['summary']['recall'])
    
    # Save detailed evaluation summary
    summary_file = results_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create one-page model assessment
    report_file = results_dir / 'model_assessment.txt'
    with open(report_file, 'w') as f:
        # Header
        f.write("Tick Detection Model Assessment\n")
        f.write("=============================\n\n")

        # Dataset Overview
        f.write("Dataset Overview\n")
        f.write("-----------------\n")
        f.write(f"Total Images Evaluated: {results['summary']['total_images']}\n")
        f.write(f"└── With Ticks:    {len(results['with_ticks']):4d} images\n")
        f.write(f"└── Without Ticks: {len(results['without_ticks']):4d} images\n\n")

        # Confusion Matrix
        f.write("Confusion Matrix\n")
        f.write("-----------------\n")
        f.write("                  Predicted        \n")
        f.write("                  Tick    No Tick  \n")
        f.write("Actual   Tick     {:4d}     {:4d}    \n".format(
            results['summary']['true_positives'],
            results['summary']['false_negatives']
        ))
        f.write("         No Tick  {:4d}     {:4d}    \n\n".format(
            results['summary']['false_positives'],
            results['summary']['true_negatives']
        ))

        # Key Metrics
        f.write("Performance Metrics\n")
        f.write("-----------------\n")
        precision = results['summary']['precision']
        recall = results['summary']['recall']
        f1 = results['summary']['f1_score']
        accuracy = (results['summary']['true_positives'] + results['summary']['true_negatives']) / results['summary']['total_images']
        
        f.write(f"Precision: {precision:.3f}  (correct predictions among all tick predictions)\n")
        f.write(f"Recall:    {recall:.3f}  (found ticks among all actual ticks)\n")
        f.write(f"F1 Score:  {f1:.3f}  (harmonic mean of precision and recall)\n")
        f.write(f"Accuracy:  {accuracy:.3f}  (correct predictions overall)\n\n")

        # Confidence Analysis
        f.write("Confidence Distribution\n")
        f.write("-----------------\n")
        f.write("Score Range    Count    % of Detections\n")
        total_detections = sum(results['summary']['confidence_distribution'].values())
        for range_str, count in sorted(results['summary']['confidence_distribution'].items(), reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            f.write(f"{range_str:12} {count:6d}    {percentage:6.1f}%\n")
        f.write("\n")

        # Error Analysis
        f.write("Error Analysis\n")
        f.write("-----------------\n")
        false_pos_images = sum(1 for data in results['without_ticks'].values() if data['num_detections'] > 0)
        missed_images = sum(1 for data in results['with_ticks'].values() if data['num_detections'] == 0)
        
        if false_pos_images > 0:
            f.write(f"False Positives: {false_pos_images} images with incorrect detections\n")
            # List top 3 worst false positives
            worst_fp = sorted(
                [(img, data) for img, data in results['without_ticks'].items() if data['num_detections'] > 0],
                key=lambda x: x[1]['num_detections'],
                reverse=True
            )[:3]
            for img, data in worst_fp:
                f.write(f"└── {img}: {data['num_detections']} detection(s), ")
                f.write(f"confidence {max(data['scores']):.2f}\n")
        
        if missed_images > 0:
            f.write(f"\nMissed Detections: {missed_images} images with ticks not detected\n")
            # List first 3 missed images
            missed = [(img, data) for img, data in results['with_ticks'].items() if data['num_detections'] == 0][:3]
            for img, data in missed:
                f.write(f"└── {img}\n")

    print(f"\nEvaluation complete!")
    print(f"- Model assessment saved to: {report_file}")
    print(f"- Detailed JSON results saved to: {summary_file}")
    print(f"\nQuick Summary:")
    print(f"- Precision: {precision:.3f}")
    print(f"- Recall: {recall:.3f}")
    print(f"- F1 Score: {f1:.3f}")
    print(f"- Accuracy: {accuracy:.3f}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    # Command-line interface for easy testing
    parser = argparse.ArgumentParser(description='''
    Evaluate a tick detection model on test images.
    
    This script is designed for testing the model on real-world scenarios:
    1. Create a test_images/ directory with your test images
    2. Run this script to get predictions
    3. Check the evaluation_results/ directory for:
       - Visualizations of detected ticks
       - Detailed JSON files with detection data
       - Summary of results across all images
    
    Suggested directory structure for test images:
    test_images/
        skin_only/           # Images of skin without ticks
        skin_with_marks/     # Images of skin with moles, freckles, etc.
        clothing/            # Images with clothing
        different_lighting/  # Various lighting conditions
        blurry/             # Slightly out-of-focus shots
    
    This helps organize testing and understand model performance
    in different scenarios.
    ''')
    
    parser.add_argument('--model', default='checkpoints/best_model.pth',
                      help='Path to trained model checkpoint')
    parser.add_argument('--input', default='test_images',
                      help='Directory containing test images')
    parser.add_argument('--output', default='evaluation_results',
                      help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Confidence threshold for detections (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    evaluate_directory(args.model, args.input, args.output, args.threshold) 