"""
TensorFlow Lite conversion script for tick detection model.

This script converts a trained TensorFlow model to TensorFlow Lite format
for deployment on mobile and edge devices.
"""

import os
import argparse
import tensorflow as tf
from model import create_model
from utils import load_config

def convert_to_tflite(model_path, config_path, output_path, quantize=False, optimize=True):
    """
    Convert a trained TensorFlow model to TensorFlow Lite format.
    
    Args:
        model_path: Path to the trained model weights (.h5 file)
        config_path: Path to the configuration file
        output_path: Path to save the TFLite model
        quantize: Whether to apply quantization
        optimize: Whether to apply optimizations
    """
    
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    print(f"Creating model with configuration...")
    model = create_model(config)
    
    print(f"Loading model weights from {model_path}")
    model.load_weights(model_path)
    
    print("Converting to TensorFlow Lite...")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    if optimize:
        print("Applying optimizations...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply quantization if requested
    if quantize:
        print("Applying quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.representative_dataset = None  # You can add a representative dataset here
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    print(f"Saving TFLite model to {output_path}")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Print model size
    model_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"Model size: {model_size:.2f} MB")
    
    print("Conversion completed successfully!")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Convert TensorFlow model to TensorFlow Lite')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5 file)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', type=str, default='tick_detector.tflite', help='Output path for TFLite model')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--no-optimize', action='store_true', help='Disable optimizations')
    
    args = parser.parse_args()
    
    try:
        convert_to_tflite(
            model_path=args.model,
            config_path=args.config,
            output_path=args.output,
            quantize=args.quantize,
            optimize=not args.no_optimize
        )
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 