#!/usr/bin/env python3
"""
Mobile inference script for tick detection TFLite model.

Usage:
    python mobile_inference.py <image_path>
"""

import sys
import numpy as np
import tensorflow as tf
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(1024, 1024)):
    """Load and preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch

def run_inference(tflite_path, image_path):
    """Run inference on an image."""
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load and preprocess image
    input_data = load_and_preprocess_image(image_path)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get outputs
    outputs = {}
    for output in output_details:
        output_data = interpreter.get_tensor(output['index'])
        outputs[output['name']] = output_data
    
    return outputs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mobile_inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    tflite_path = "tick_detector.tflite"
    
    try:
        results = run_inference(tflite_path, image_path)
        print("✅ Inference completed!")
        print("📊 Results:")
        for key, value in results.items():
            print(f"  {key}: {value.shape}")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        sys.exit(1)
