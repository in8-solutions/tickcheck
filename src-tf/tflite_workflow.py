#!/usr/bin/env python3
"""
Complete TensorFlow → TFLite Workflow for Tick Detection

This script provides a complete pipeline from training to TFLite conversion
for mobile deployment of the tick detection model.
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
import time

def check_prerequisites():
    """Check if all prerequisites are met for TFLite conversion."""
    print("🔍 Checking prerequisites...")
    
    # Check TensorFlow version
    tf_version = tf.__version__
    print(f"✅ TensorFlow version: {tf_version}")
    
    # Check TFLite converter availability
    try:
        converter = tf.lite.TFLiteConverter
        print("✅ TFLite converter available")
    except Exception as e:
        print(f"❌ TFLite converter not available: {e}")
        return False
    
    # Check MPS support
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU devices available: {len(gpus)}")
    else:
        print("⚠️  No GPU devices found (will use CPU)")
    
    return True

def train_model(config_path="config-mac.yaml", quick_test=False):
    """Train the model using the training script."""
    print(f"\n🚀 Starting model training...")
    print(f"Config: {config_path}")
    print(f"Quick test: {quick_test}")
    
    # Check if training script exists
    if not os.path.exists("train.py"):
        print("❌ train.py not found!")
        return False
    
    # Build training command
    cmd = f"python train.py --config {config_path}"
    if quick_test:
        cmd += " --quick-test"
    
    print(f"Running: {cmd}")
    
    # For now, just return success (you can run this manually)
    print("📝 Note: Run the training command manually:")
    print(f"   {cmd}")
    print("\nAfter training completes, you'll have a .h5 model file.")
    
    return True

def convert_to_tflite(model_path, config_path="config-mac.yaml", 
                      output_path="tick_detector.tflite", 
                      quantize=False, optimize=True):
    """Convert trained model to TFLite format."""
    print(f"\n🔄 Converting model to TFLite...")
    print(f"Input model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {quantize}")
    print(f"Optimization: {optimize}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Import conversion function
        from convert_to_tflite import convert_to_tflite
        
        # Perform conversion
        convert_to_tflite(
            model_path=model_path,
            config_path=config_path,
            output_path=output_path,
            quantize=quantize,
            optimize=optimize
        )
        
        # Check file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ TFLite model created: {output_path}")
        print(f"📏 Model size: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tflite_model(tflite_path, test_image_size=(1024, 1024)):
    """Test the TFLite model with dummy data."""
    print(f"\n🧪 Testing TFLite model...")
    print(f"Model: {tflite_path}")
    print(f"Test input size: {test_image_size}")
    
    if not os.path.exists(tflite_path):
        print(f"❌ TFLite model not found: {tflite_path}")
        return False
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"✅ Model loaded successfully")
        print(f"📥 Input details: {input_details}")
        print(f"📤 Output details: {output_details}")
        
        # Create dummy input
        height, width = test_image_size
        dummy_input = np.random.random((1, height, width, 3)).astype(np.float32)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        
        # Run inference
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get outputs
        outputs = []
        for output in output_details:
            output_data = interpreter.get_tensor(output['index'])
            outputs.append(output_data)
        
        print(f"✅ Inference successful")
        print(f"⏱️  Inference time: {inference_time:.4f} seconds")
        print(f"📊 Output shapes: {[out.shape for out in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"❌ TFLite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_mobile_inference_script(tflite_path="tick_detector.tflite"):
    """Create a simple mobile inference script."""
    print(f"\n📱 Creating mobile inference script...")
    
    script_content = f'''#!/usr/bin/env python3
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
    outputs = {{}}
    for output in output_details:
        output_data = interpreter.get_tensor(output['index'])
        outputs[output['name']] = output_data
    
    return outputs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mobile_inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    tflite_path = "{tflite_path}"
    
    try:
        results = run_inference(tflite_path, image_path)
        print("✅ Inference completed!")
        print("📊 Results:")
        for key, value in results.items():
            print(f"  {{key}}: {{value.shape}}")
    except Exception as e:
        print(f"❌ Inference failed: {{e}}")
        sys.exit(1)
'''
    
    with open("mobile_inference.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created mobile_inference.py")
    print("📝 Usage: python mobile_inference.py <image_path>")

def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(description='TensorFlow → TFLite Workflow')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--convert', type=str, help='Convert model to TFLite (provide .h5 model path)')
    parser.add_argument('--test', type=str, help='Test TFLite model (provide .tflite model path)')
    parser.add_argument('--config', default='config-mac.yaml', help='Configuration file')
    parser.add_argument('--output', default='tick_detector.tflite', help='Output TFLite model path')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    parser.add_argument('--quick-test', action='store_true', help='Quick training test')
    parser.add_argument('--create-mobile-script', action='store_true', help='Create mobile inference script')
    
    args = parser.parse_args()
    
    print("🎯 TensorFlow → TFLite Workflow")
    print("=" * 50)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met!")
        return 1
    
    # Training step
    if args.train:
        if not train_model(args.config, args.quick_test):
            return 1
    
    # Conversion step
    if args.convert:
        if not convert_to_tflite(args.convert, args.config, args.output, args.quantize):
            return 1
    
    # Testing step
    if args.test:
        if not test_tflite_model(args.test):
            return 1
    
    # Create mobile script
    if args.create_mobile_script:
        create_mobile_inference_script(args.output)
    
    # If no specific action, show usage
    if not any([args.train, args.convert, args.test, args.create_mobile_script]):
        print("\n📋 Usage Examples:")
        print("1. Train model: python tflite_workflow.py --train")
        print("2. Convert model: python tflite_workflow.py --convert model.h5")
        print("3. Test TFLite: python tflite_workflow.py --test tick_detector.tflite")
        print("4. Full pipeline: python tflite_workflow.py --train --convert model.h5 --test tick_detector.tflite")
        print("5. Create mobile script: python tflite_workflow.py --create-mobile-script")
    
    print("\n✅ Workflow completed!")
    return 0

if __name__ == "__main__":
    exit(main()) 