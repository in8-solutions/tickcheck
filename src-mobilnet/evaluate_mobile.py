"""
Mobile Model Evaluation Script

This script evaluates a trained mobile tick detection model on test images.
It's designed to evaluate binary classification performance on separate sets of
images with and without ticks.

Key Features:
- Binary classification evaluation (tick present vs no tick)
- Separate evaluation for images with and without ticks
- Confidence threshold analysis
- Comprehensive metrics and visualization
- Support for custom test datasets

Usage:
    python evaluate_mobile.py --model path/to/model.pth --input path/to/test/dir --output path/to/results
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple, Optional

from config import DATA_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG
from model import create_model
from data_pipeline import get_transforms
from utils import load_checkpoint


class MobileModelEvaluator:
    """
    A class to evaluate a mobile binary classification model on images.
    
    This class handles binary classification (tick vs no tick) and provides
    comprehensive metrics for images with and without ticks.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        # Create model
        self.model = create_model(MODEL_CONFIG)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get transforms
        self.transform = get_transforms('val')
        
        # Initialize metrics storage
        self.metrics = {
            'with_ticks': {
                'total_images': 0,
                'predictions': [],  # List of (predicted_label, confidence, true_label)
                'true_positives': 0,
                'false_negatives': 0,
                'confidence_scores': []
            },
            'without_ticks': {
                'total_images': 0,
                'predictions': [],  # List of (predicted_label, confidence, true_label)
                'false_positives': 0,
                'true_negatives': 0,
                'confidence_scores': []
            }
        }
        
        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {confidence_threshold}")
    
    def predict_image(self, image_path: str) -> Tuple[int, float]:
        """Run prediction on a single image."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = outputs['probabilities']
            logits = outputs['logits']
        
        # Get predicted class and confidence
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence
    
    def evaluate_directory(self, input_dir: str, output_dir: str):
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
                predicted_class, confidence = self.predict_image(image_path)
                true_label = 1 if category == 'with_ticks' else 0
                
                # Store prediction
                self.metrics[category]['predictions'].append((predicted_class, confidence, true_label))
                self.metrics[category]['confidence_scores'].append(confidence)
                
                # Update metrics
                self.metrics[category]['total_images'] += 1
                
                if category == 'with_ticks':
                    if predicted_class == 1:
                        self.metrics[category]['true_positives'] += 1
                    else:
                        self.metrics[category]['false_negatives'] += 1
                else:  # without_ticks
                    if predicted_class == 1:
                        self.metrics[category]['false_positives'] += 1
                    else:
                        self.metrics[category]['true_negatives'] += 1
                
                # Save visualization
                self.visualize_prediction(image_path, predicted_class, confidence, output_path)
        
        # Generate and save comprehensive report
        self.generate_report(output_dir)
    
    def visualize_prediction(self, image_path: str, predicted_class: int, confidence: float, output_path: str):
        """Create visualization of prediction on image."""
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Resize for display if too large
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create drawing object
        draw = ImageDraw.Draw(image)
        
        # Define colors and text
        if predicted_class == 1:  # Tick detected
            color = (255, 0, 0) if confidence >= self.confidence_threshold else (255, 165, 0)  # Red or Orange
            text = f"TICK DETECTED ({confidence:.3f})"
        else:  # No tick
            color = (0, 255, 0) if confidence >= self.confidence_threshold else (255, 255, 0)  # Green or Yellow
            text = f"NO TICK ({confidence:.3f})"
        
        # Draw border around image
        border_width = 5
        draw.rectangle([(0, 0), (image.width-1, image.height-1)], outline=color, width=border_width)
        
        # Add text label
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
        
        # Calculate text position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Position text at top-left with background
        text_x = border_width + 5
        text_y = border_width + 5
        
        # Draw text background
        draw.rectangle([(text_x-2, text_y-2), (text_x+text_width+2, text_y+text_height+2)], 
                      fill=(0, 0, 0, 128))
        
        # Draw text
        draw.text((text_x, text_y), text, fill=color, font=font)
        
        # Save image
        image.save(output_path)
    
    def generate_report(self, output_dir: str):
        """Generate comprehensive evaluation report with binary classification metrics."""
        # Calculate overall metrics
        tp = self.metrics['with_ticks']['true_positives']
        fp = self.metrics['without_ticks']['false_positives']
        fn = self.metrics['with_ticks']['false_negatives']
        tn = self.metrics['without_ticks']['true_negatives']
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Calculate confidence distributions
        for category in ['with_ticks', 'without_ticks']:
            confidence_ranges = {
                '0.0-0.2': 0,
                '0.2-0.4': 0,
                '0.4-0.6': 0,
                '0.6-0.8': 0,
                '0.8-1.0': 0
            }
            
            for score in self.metrics[category]['confidence_scores']:
                if score < 0.2:
                    confidence_ranges['0.0-0.2'] += 1
                elif score < 0.4:
                    confidence_ranges['0.2-0.4'] += 1
                elif score < 0.6:
                    confidence_ranges['0.4-0.6'] += 1
                elif score < 0.8:
                    confidence_ranges['0.6-0.8'] += 1
                else:
                    confidence_ranges['0.8-1.0'] += 1
            
            self.metrics[category]['confidence_distribution'] = confidence_ranges
        
        # Prepare final report
        report = {
            'confidence_threshold': self.confidence_threshold,
            'binary_classification_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'confusion_matrix': {
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn
                }
            },
            'with_ticks': {
                'total_images': self.metrics['with_ticks']['total_images'],
                'true_positives': tp,
                'false_negatives': fn,
                'average_confidence': np.mean(self.metrics['with_ticks']['confidence_scores']) if self.metrics['with_ticks']['confidence_scores'] else 0,
                'confidence_distribution': self.metrics['with_ticks']['confidence_distribution']
            },
            'without_ticks': {
                'total_images': self.metrics['without_ticks']['total_images'],
                'false_positives': fp,
                'true_negatives': tn,
                'average_confidence': np.mean(self.metrics['without_ticks']['confidence_scores']) if self.metrics['without_ticks']['confidence_scores'] else 0,
                'confidence_distribution': self.metrics['without_ticks']['confidence_distribution']
            }
        }
        
        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create visualizations
        self.create_visualizations(output_dir)
        
        # Print human-readable report
        print("\n" + "="*60)
        print("MOBILE MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\nModel Configuration:")
        print(f"Architecture: {MODEL_CONFIG['architecture']}")
        print(f"Input Size: {DATA_CONFIG['input_size']}")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        
        print("\nBinary Classification Metrics:")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        
        print("\nConfusion Matrix:")
        print(f"True Positives:  {tp}")
        print(f"False Positives: {fp}")
        print(f"True Negatives:  {tn}")
        print(f"False Negatives: {fn}")
        
        for category in ['with_ticks', 'without_ticks']:
            print(f"\n{category.replace('_', ' ').title()} Metrics:")
            print(f"Total Images: {self.metrics[category]['total_images']}")
            print(f"Average Confidence: {report[category]['average_confidence']:.3f}")
            
            print("\nConfidence Distribution:")
            for range_name, count in self.metrics[category]['confidence_distribution'].items():
                percentage = (count / self.metrics[category]['total_images'] * 100 
                            if self.metrics[category]['total_images'] > 0 else 0)
                print(f"  {range_name}: {count} images ({percentage:.1f}%)")
        
        print("\n" + "="*60)
        print(f"Full report saved to: {report_path}")
        print(f"Visualizations saved to: {output_dir}")
        print("="*60 + "\n")
    
    def create_visualizations(self, output_dir: str):
        """Create visualization plots for the evaluation results."""
        # Create confidence distribution plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        categories = ['with_ticks', 'without_ticks']
        colors = ['red', 'blue']
        
        for i, (category, color) in enumerate(zip(categories, colors)):
            if self.metrics[category]['total_images'] > 0:
                conf_dist = self.metrics[category]['confidence_distribution']
                ranges = list(conf_dist.keys())
                counts = list(conf_dist.values())
                
                axes[i].bar(ranges, counts, color=color, alpha=0.7)
                axes[i].set_title(f'Confidence Distribution - {category.replace("_", " ").title()}')
                axes[i].set_xlabel('Confidence Range')
                axes[i].set_ylabel('Number of Images')
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                for j, count in enumerate(counts):
                    percentage = (count / self.metrics[category]['total_images'] * 100)
                    axes[i].text(j, count + max(counts) * 0.01, f'{percentage:.1f}%', 
                               ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrix visualization
        tp = self.metrics['with_ticks']['true_positives']
        fp = self.metrics['without_ticks']['false_positives']
        fn = self.metrics['with_ticks']['false_negatives']
        tn = self.metrics['without_ticks']['true_negatives']
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        classes = ['No Tick', 'Tick']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text in cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_directory(model_path: str, input_dir: str, output_dir: str, confidence_threshold: float = 0.5):
    """Evaluate a directory of images with separate 'with_ticks' and 'without_ticks' subdirectories."""
    evaluator = MobileModelEvaluator(model_path, confidence_threshold=confidence_threshold)
    evaluator.evaluate_directory(input_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained mobile model on images")
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--input", required=True, help="Path to input directory containing 'with_ticks' and 'without_ticks' subdirectories")
    parser.add_argument("--output", required=True, help="Path to output directory for results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for predictions (default: 0.5)")
    
    args = parser.parse_args()
    evaluate_directory(args.model, args.input, args.output, args.confidence) 