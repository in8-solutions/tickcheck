"""
Test script to verify TensorFlow setup on Mac M3.

This script tests:
1. TensorFlow installation
2. MPS (Metal Performance Shaders) availability
3. Basic model creation
4. Data loading
5. Simple training step
"""

import os
import sys
import tensorflow as tf
import numpy as np
from pathlib import Path

def test_tensorflow_installation():
    """Test basic TensorFlow installation."""
    print("="*50)
    print("TESTING TENSORFLOW INSTALLATION")
    print("="*50)
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check available devices
    print("\nAvailable devices:")
    print(f"CPU devices: {tf.config.list_physical_devices('CPU')}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    
    # Test basic operations
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    c = tf.matmul(a, b)
    print(f"\nBasic matrix multiplication test: {c.numpy()}")
    
    return True

def test_mps_support():
    """Test MPS (Metal Performance Shaders) support."""
    print("\n" + "="*50)
    print("TESTING MPS SUPPORT")
    print("="*50)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU device(s)")
        
        # Enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ MPS memory growth enabled")
        except RuntimeError as e:
            print(f"✗ Memory growth setting failed: {e}")
        
        # Test GPU computation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
                b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
                c = tf.matmul(a, b)
                print(f"✓ GPU computation test: {c.numpy()}")
                return True
        except Exception as e:
            print(f"✗ GPU computation failed: {e}")
            return False
    else:
        print("No GPU devices found - MPS not available")
        return False

def test_model_creation():
    """Test basic model creation."""
    print("\n" + "="*50)
    print("TESTING MODEL CREATION")
    print("="*50)
    
    try:
        # Import our modules
        from utils import load_config
        from model import create_model
        
        # Load config
        config_path = 'config-mac.yaml'
        if not os.path.exists(config_path):
            print(f"Config file {config_path} not found, using default config")
            config = {
                'model': {
                    'num_classes': 2,
                    'pretrained': False,  # Don't download weights for test
                    'freeze_backbone': False,
                    'anchor_sizes': [16, 32, 64, 128, 256],
                    'anchor_ratios': [0.7, 1.0, 1.3]
                },
                'data': {
                    'input_size': [1024, 1024]  # Match model expectations
                },
                'training': {
                    'learning_rate': 0.0001,
                    'weight_decay': 0.001
                }
            }
        else:
            config = load_config(config_path)
        
        # Create model
        model = create_model(config)
        print(f"✓ Model created successfully")
        
        # Build the model before counting parameters
        test_input = tf.random.normal([1, 1024, 1024, 3])
        _ = model(test_input, training=False)  # This builds the model
        print(f"  Total parameters: {model.count_params():,}")
        
        # Test forward pass
        output = model(test_input, training=False)
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n" + "="*50)
    print("TESTING DATA LOADING")
    print("="*50)
    
    try:
        from dataset import DetectionDataset
        from transforms import get_transform
        
        # Check if data exists (relative to src-tf directory)
        data_paths = [
            "../data/chunk_001/images",
            "../data/chunk_001/annotations.json"
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                print(f"✓ Found: {path}")
            else:
                print(f"✗ Missing: {path}")
        
        # Test transform creation
        config = {
            'data': {
                'input_size': [512, 512],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'augmentation': {
                'train': {
                    'horizontal_flip': True,
                    'vertical_flip': False,
                    'rotate': {'enabled': True, 'limit': 45},
                    'scale': {'enabled': True, 'min': 0.8, 'max': 1.2},
                    'brightness': {'enabled': True, 'limit': 0.2},
                    'contrast': {'enabled': True, 'limit': 0.2},
                    'blur': {'enabled': True, 'limit': 3},
                    'noise': {'enabled': True, 'limit': 0.05}
                }
            }
        }
        
        transform = get_transform(config, train=True)
        print("✓ Transform pipeline created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a simple training step."""
    print("\n" + "="*50)
    print("TESTING TRAINING STEP")
    print("="*50)
    
    try:
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(512, 512, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create dummy data
        batch_size = 2
        x = tf.random.normal([batch_size, 512, 512, 3])
        y = tf.random.uniform([batch_size], maxval=2, dtype=tf.int32)
        
        # Test training step
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            loss, accuracy = model.train_on_batch(x, y)
            print(f"✓ Training step successful")
            print(f"  Loss: {loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("TensorFlow Mac M3 Setup Test")
    print("="*60)
    
    tests = [
        ("TensorFlow Installation", test_tensorflow_installation),
        ("MPS Support", test_mps_support),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Mac M3 setup is ready for training.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python train.py --config config-mac.yaml")
        print("3. For quick test: python train.py --config config-mac.yaml --quick-test")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 