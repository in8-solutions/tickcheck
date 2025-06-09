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

def evaluate_directory(model_path, input_dir, output_dir, confidence_threshold=0.5):
    """
    Evaluate model on all images in a directory.
    
    This function demonstrates:
    1. Batch processing of test images
    2. Organizing evaluation results
    3. Creating summary statistics
    4. Saving results for later analysis
    
    For student projects, consider:
    - Adding different test image categories (skin_only/, with_ticks/, etc.)
    - Tracking false positives in tick-free images
    - Measuring detection rates on known tick images
    - Testing with different lighting conditions
    """
    evaluator = ModelEvaluator(model_path, confidence_threshold=confidence_threshold)
    
    # Create output directory structure
    vis_dir = Path(output_dir) / 'visualizations'
    results_dir = Path(output_dir) / 'results'
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Process all images
    results = {}
    input_dir = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for image_path in input_dir.glob('**/*'):
        if image_path.suffix.lower() not in image_extensions:
            continue
        
        print(f"\nProcessing: {image_path}")
        
        # Run detection
        detections = evaluator.process_image(image_path)
        
        # Save detailed results as JSON for later analysis
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
        
        # Store summary metrics
        results[image_path.name] = {
            'num_detections': len(detections['boxes']),
            'max_confidence': float(max(detections['scores'])) if len(detections['scores']) > 0 else 0,
            'detections_file': str(result_file),
            'visualization_file': str(vis_file)
        }
    
    # Save evaluation summary
    summary_file = Path(output_dir) / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation complete! Summary saved to {summary_file}")
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