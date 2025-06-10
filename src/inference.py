"""
Inference script for tick detection model.

This module handles model inference and result visualization. It provides
functionality to:
- Load trained models
- Process images for inference
- Run detection
- Visualize and save results

Key Features:
- Support for various image formats
- Configurable confidence thresholds
- Result visualization with bounding boxes
- Batch processing of test images
"""

import os
import torch
from PIL import Image
import torchvision.transforms as T
from utils import load_config, visualize_prediction
from model import create_model

def load_trained_model(checkpoint_path, config):
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint file
        config (dict): Model configuration dictionary
    
    Returns:
        nn.Module: Loaded model with trained weights
    """
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def prepare_image(image_path, config):
    """Prepare an image for model inference.
    
    This function:
    1. Loads and converts image to RGB
    2. Applies necessary transformations:
       - Resizing to model input size
       - Conversion to tensor
       - Normalization
    
    Args:
        image_path (str): Path to the input image
        config (dict): Configuration containing preprocessing parameters
    
    Returns:
        torch.Tensor: Preprocessed image tensor ready for inference
    """
    # Load and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Create transform
    transform = T.Compose([
        T.Resize(config['data']['input_size']),
        T.ToTensor(),
        T.Normalize(
            mean=config['data']['mean'],
            std=config['data']['std']
        )
    ])
    
    # Apply transform
    image_tensor = transform(image)
    
    return image_tensor

def run_inference(model, image_tensor, confidence_threshold=0.5):
    """Run object detection inference on a single image.
    
    This function:
    1. Moves data to appropriate device
    2. Runs model inference
    3. Filters predictions by confidence threshold
    4. Returns detection results
    
    Args:
        model (nn.Module): The trained detection model
        image_tensor (torch.Tensor): Preprocessed input image
        confidence_threshold (float): Minimum confidence for detections
    
    Returns:
        tuple: (boxes, scores, labels) for detections above threshold
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Get predictions above threshold
    pred = predictions[0]
    mask = pred['scores'] >= confidence_threshold
    
    filtered_boxes = pred['boxes'][mask]
    filtered_scores = pred['scores'][mask]
    filtered_labels = pred['labels'][mask]
    
    return filtered_boxes, filtered_scores, filtered_labels

def main():
    """Main inference function.
    
    This function orchestrates the inference process:
    1. Loads model configuration and weights
    2. Processes all images in test directory
    3. Runs inference on each image
    4. Visualizes and saves results
    
    The function handles:
    - Multiple image formats (PNG, JPG, JPEG)
    - Proper directory creation
    - Progress reporting
    - Result visualization and saving
    """
    # Load configuration
    config = load_config('config.yaml')
    
    # Load model
    model = load_trained_model(
        os.path.join(config['paths']['checkpoint_dir'], 'best_model.pth'),
        config
    )
    
    # Get class names from config
    class_names = config['classes']
    
    # Process images in the test directory
    test_dir = 'data/test_images'
    output_dir = config['paths']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    for image_name in os.listdir(test_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(test_dir, image_name)
        print(f'Processing {image_path}...')
        
        # Prepare image
        image_tensor = prepare_image(image_path, config)
        
        # Run inference
        boxes, scores, labels = run_inference(model, image_tensor)
        
        # Visualize results
        fig = visualize_prediction(
            image_tensor,
            boxes,
            scores,
            labels,
            class_names
        )
        
        # Save visualization
        output_path = os.path.join(output_dir, f'pred_{image_name}')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f'Saved prediction to {output_path}')

if __name__ == '__main__':
    main() 