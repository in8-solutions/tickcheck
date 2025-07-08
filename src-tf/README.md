# TensorFlow Tick Detection Model (Independent Tooling)

This directory contains a **standalone TensorFlow/Keras implementation** of the tick detection system, designed to be compatible with TensorFlow Lite for mobile deployment.

**Note**: This is an independent tooling directory for exploring TFLite options while the main model training/analysis happens in the top-level `src/` directory with PyTorch/CUDA.

## Overview

The TensorFlow version provides the same functionality as the PyTorch version but with the following advantages:

- **TensorFlow Lite Support**: Easy conversion to TFLite models for mobile deployment
- **Keras API**: Clean, high-level API for model development
- **TensorFlow Ecosystem**: Integration with TensorFlow's extensive tooling
- **Mixed Precision Training**: Built-in support for faster training with mixed precision
- **GPU Optimization**: Optimized for TensorFlow's GPU acceleration

## Architecture

The model is based on RetinaNet with the following components:

- **Backbone**: ResNet50 with ImageNet pretrained weights
- **Feature Pyramid Network (FPN)**: Multi-scale feature extraction
- **Classification Head**: Focal Loss for handling class imbalance
- **Regression Head**: Smooth L1 Loss for bounding box regression
- **Anchor Generation**: Multi-scale anchor boxes for detection

## Files Structure

```
src-tf/
├── requirements.txt          # Python dependencies
├── utils.py                  # Utility functions
├── transforms.py             # Data augmentation and preprocessing
├── dataset.py               # Dataset loading and management
├── model.py                 # RetinaNet model implementation
├── train.py                 # Training script
├── evaluate_model.py        # Model evaluation script
├── convert_to_tflite.py     # TFLite conversion script
├── tflite_workflow.py       # Complete TFLite workflow pipeline
├── mobile_inference.py      # Mobile inference reference script
├── test_mac_setup.py        # Mac MPS validation script
└── README.md               # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the same data structure as the PyTorch version:
```
data/
├── chunk_001/
│   ├── annotations.json
│   └── images/
├── chunk_002/
│   ├── annotations.json
│   └── images/
└── ...
```

## Usage

### Training

To train the model:

```bash
cd src-tf
python train.py --config ../config.yaml
```

Optional arguments:
- `--resume /path/to/checkpoint.h5`: Resume training from a checkpoint
- `--quick-test`: Run a quick test with a small dataset

### Evaluation

To evaluate a trained model:

```bash
python evaluate_model.py \
    --model /path/to/model.h5 \
    --input /path/to/test/images \
    --output /path/to/results \
    --confidence 0.5
```

### TensorFlow Lite Conversion

#### Option 1: Manual Conversion
To convert the trained model to TensorFlow Lite:

```python
import tensorflow as tf
from model import create_model
from utils import load_config

# Load configuration and create model
config = load_config('config.yaml')
model = create_model(config)

# Load trained weights
model.load_weights('path/to/model.h5')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('tick_detector.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### Option 2: Automated Workflow (Recommended)
Use the comprehensive TFLite workflow script:

```bash
# Check prerequisites
python tflite_workflow.py

# Convert trained model to TFLite
python tflite_workflow.py --convert model.h5

# Test TFLite model
python tflite_workflow.py --test tick_detector.tflite

# Create mobile inference reference
python tflite_workflow.py --create-mobile-script
```

#### Mobile Inference Reference
The `mobile_inference.py` script provides a reference implementation for using the TFLite model:

```bash
python mobile_inference.py <image_path>
```

## Key Features

### 1. Mixed Precision Training
The training script supports automatic mixed precision (AMP) for faster training and reduced memory usage:

```python
# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### 2. Data Augmentation
Comprehensive data augmentation using Albumentations:
- Horizontal and vertical flips
- Rotation and scaling
- Brightness and contrast adjustments
- Gaussian blur and noise

### 3. Loss Functions
- **Focal Loss**: Handles class imbalance in object detection
- **Smooth L1 Loss**: Robust regression loss for bounding boxes

### 4. Model Architecture
- **ResNet50 Backbone**: Feature extraction with pretrained weights
- **Feature Pyramid Network**: Multi-scale feature representation
- **RetinaNet Heads**: Classification and regression heads

## Configuration

The model uses the same `config.yaml` file as the PyTorch version. Key configuration sections:

```yaml
model:
  name: "retinanet"
  backbone: "resnet50"
  num_classes: 2
  pretrained: true
  freeze_backbone: false
  anchor_sizes: [16, 32, 64, 128, 256]
  anchor_ratios: [0.7, 1.0, 1.3]

training:
  device: "cuda"
  num_epochs: 100
  batch_size: 7
  learning_rate: 0.0001
  use_amp: true
```

## Performance Optimization

### 1. Memory Optimization
- Gradient checkpointing for large models
- Dynamic batch sizing
- Memory-efficient data loading

### 2. Training Speed
- Mixed precision training
- GPU memory optimization
- Efficient data pipeline with `tf.data`

### 3. Model Optimization
- Model pruning for smaller models
- Quantization for TFLite deployment
- Knowledge distillation for model compression

## Mobile Deployment

### TensorFlow Lite Conversion
```python
# Convert to TFLite with optimizations
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
```

### Android Integration
```java
// Load TFLite model in Android
Interpreter tflite = new Interpreter(loadModelFile(context, "tick_detector.tflite"));
```

### iOS Integration
```swift
// Load TFLite model in iOS
let interpreter = try Interpreter(modelPath: modelPath)
```

## Comparison with PyTorch Version

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| Model Architecture | RetinaNet | RetinaNet |
| Backbone | ResNet50 | ResNet50 |
| Loss Functions | Focal + Smooth L1 | Focal + Smooth L1 |
| Data Augmentation | Albumentations | Albumentations |
| Mixed Precision | PyTorch AMP | TensorFlow AMP |
| Mobile Deployment | TorchScript | TensorFlow Lite |
| Training Speed | Fast | Fast |
| Memory Usage | Optimized | Optimized |

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**
   - Reduce batch size in config
   - Enable mixed precision training
   - Use gradient checkpointing

2. **Data Loading Issues**
   - Check data paths in config
   - Verify annotation format
   - Ensure images are accessible

3. **Model Conversion Issues**
   - Ensure model is saved in correct format
   - Check for unsupported operations
   - Verify input/output shapes

### Performance Tips

1. **Training**
   - Use mixed precision for faster training
   - Optimize data pipeline with prefetching
   - Use appropriate batch size for your GPU

2. **Inference**
   - Use TensorFlow Lite for mobile deployment
   - Enable quantization for smaller models
   - Optimize input preprocessing

## Contributing

When contributing to the TensorFlow version:

1. Maintain compatibility with the PyTorch version
2. Follow TensorFlow best practices
3. Add comprehensive tests
4. Update documentation
5. Ensure TFLite compatibility

## License

Same license as the main project. 