# Mobile Tick Detection Implementation

This directory contains the mobile-ready implementation of the tick detection system, converting the original RetinaNet object detection model into a lightweight binary classifier suitable for mobile deployment.

## Overview

The mobile implementation transforms object detection annotations into binary classification samples using bounding box cropping, enabling the use of lightweight CNN architectures like MobileNetV3-Small for mobile deployment.

## Key Features

- **Binary Classification**: Converts from object detection to "tick present" vs "no tick"
- **Data Pipeline**: Uses existing annotations to create positive/negative samples via cropping
- **Mobile Models**: Lightweight architectures (MobileNetV3-Small, EfficientNet-B0, custom lightweight)
- **Mobile Export**: Support for iOS/Android deployment
- **Comparison Framework**: Compare against existing RetinaNet model

## Architecture

### Data Pipeline
- **Positive Samples**: Crops around bounding boxes with padding
- **Negative Samples**: Random crops from images without ticks
- **Augmentation**: Mobile-optimized augmentations (rotation, brightness, contrast)
- **Input Size**: 224x224 (mobile-optimized)

### Model Options
1. **MobileNetV3-Small**: ~4.7 MB, <50ms inference
2. **EfficientNet-B0**: ~8.5 MB, <100ms inference  
3. **Lightweight Custom**: ~2.5 MB, <30ms inference

## Quick Start

### 1. Quick Test
```bash
cd src-mobilnet
source ../venv/bin/activate
python train_mobile.py --quick-test
```

### 2. Full Training
```bash
python train_mobile.py
```

### 3. Evaluate Performance
```bash
# Evaluate on custom test dataset
python evaluate_mobile.py --model outputs/mobile/checkpoints/best_model.pth --input path/to/test/dataset --output path/to/results

# Example with custom confidence threshold
python evaluate_mobile.py --model outputs/mobile/checkpoints/best_model.pth --input test_images --output evaluation_results --confidence 0.7
```

## Configuration

The implementation uses a centralized configuration system in `config.py`:

### Data Configuration
- **Input Size**: 224x224 (mobile-optimized)
- **Crop Padding**: 0.3 (30% padding around bounding boxes)
- **Negative Crops**: 1 per image (configurable)
- **Train/Val Split**: 80/20

### Model Configuration
- **Architecture**: MobileNetV3-Small (default)
- **Pretrained**: True (ImageNet weights)
- **Dropout**: 0.2
- **Attention**: False (optional)

### Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 50
- **Early Stopping**: 10 epochs patience
- **Mixed Precision**: Enabled

## Data Pipeline Details

### From Object Detection to Binary Classification

**Original Format:**
```
Image + Bounding Boxes → Object Detection
```

**New Format:**
```
Image → Binary Classification (tick/no tick)
```

### Training-Time Cropping Strategy

1. **Positive Samples**: For each bounding box annotation
   - Crop around the box with 30% padding
   - Ensure minimum crop size (50px)
   - Use as positive sample (tick present)

2. **Negative Samples**: For images without ticks
   - Generate random crops avoiding tick areas
   - Use as negative samples (no tick)

3. **Multiple Boxes**: Images with multiple ticks
   - Create separate positive sample for each box
   - Maximize training data utilization

### Benefits
- **No Storage Overhead**: Crops generated on-the-fly
- **Data Augmentation**: Different crops each epoch
- **Flexible Padding**: Configurable crop sizes
- **Maintains Original Data**: Annotations stay intact

## Model Architectures

### MobileNetV3-Small (Recommended)
- **Size**: ~4.7 MB
- **Parameters**: ~1.2M
- **Inference Time**: <50ms
- **Accuracy**: High (comparable to RetinaNet)

### EfficientNet-B0
- **Size**: ~8.5 MB
- **Parameters**: ~5.3M
- **Inference Time**: <100ms
- **Accuracy**: Very high

### Lightweight Custom
- **Size**: ~2.5 MB
- **Parameters**: ~0.5M
- **Inference Time**: <30ms
- **Accuracy**: Good (suitable for very constrained devices)

## Training Process

### Transfer Learning
1. **Pre-training**: ImageNet weights
2. **Fine-tuning**: Gradual unfreezing of layers
3. **Learning Rate**: Lower LR for pre-trained layers

### Loss Function
- **Binary Cross-Entropy**: Standard for binary classification
- **Focal Loss**: Available for class imbalance
- **Label Smoothing**: For better generalization

### Evaluation Metrics
- **Primary**: Precision, Recall, F1-Score
- **Secondary**: ROC-AUC, Confusion Matrix
- **Comparison**: Side-by-side with RetinaNet results

## Output Structure

```
outputs/mobile/
├── checkpoints/
│   ├── best_model.pth          # Best validation loss
│   ├── best_f1_model.pth       # Best F1 score
│   └── final_model.pth         # Final model
├── curves/
│   ├── training_history.json   # Training metrics
│   └── training_curves.png     # Training plots
├── evaluation/
│   └── results.json            # Evaluation results
└── export/
    ├── model.onnx              # ONNX format
    ├── model.tflite            # TFLite format
    └── model.mlmodel           # Core ML format
```

## Mobile Deployment

### Model Optimization
1. **Quantization**: INT8 quantization for size reduction
2. **Pruning**: Remove unnecessary weights
3. **Knowledge Distillation**: Use RetinaNet as teacher

### Export Formats
- **iOS**: Core ML format (`.mlmodel`)
- **Android**: TFLite format (`.tflite`)
- **Cross-platform**: ONNX format (`.onnx`)

### Performance Targets
- **Model Size**: <5MB
- **Inference Time**: <50ms on iPhone 12+ / Pixel 6+
- **Memory Usage**: <50MB RAM
- **Battery Impact**: Minimal

## Comparison with RetinaNet

| Metric | RetinaNet | MobileNetV3-Small |
|--------|-----------|-------------------|
| Model Size | ~100MB | ~4.7MB |
| Inference Time | ~500ms | ~50ms |
| Memory Usage | ~500MB | ~50MB |
| Accuracy (F1) | ~0.98 | ~0.95 |
| Use Case | Server/Desktop | Mobile/Edge |

## Usage Examples

### Basic Training
```python
from train_mobile import MobileTrainer
from data_pipeline import create_data_loaders

# Create data loaders
train_loader, val_loader = create_data_loaders(data_dir)

# Initialize trainer
trainer = MobileTrainer()
trainer.setup_model()

# Start training
trainer.train(train_loader, val_loader)
```

### Command Line Usage
```bash
# Quick test mode (limited data, 2 epochs)
python train_mobile.py --quick-test

# Full training mode
python train_mobile.py
```

### Model Creation
```python
from model import create_model, get_model_size_mb

# Create model
model = create_model()

# Check model size
size_mb = get_model_size_mb(model)
print(f"Model size: {size_mb:.2f} MB")
```

### Data Pipeline
```python
from data_pipeline import TickBinaryDataset, get_transforms

# Create dataset
dataset = TickBinaryDataset(
    image_paths=image_paths,
    annotation_files=annotation_files,
    transform=get_transforms('train')
)

# Get sample
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Label: {sample['label']}")
```

## Requirements

See `requirements.txt` for the complete list of dependencies:

- torch>=2.0.0
- torchvision>=0.15.0
- albumentations>=1.3.0
- scikit-learn>=1.0.0
- matplotlib>=3.5.0
- tqdm>=4.64.0
- opencv-python>=4.5.0

## Success Criteria

1. **Model Performance**: F1-Score > 0.90 (comparable to RetinaNet)
2. **Mobile Compatibility**: <5MB model size, <50ms inference
3. **Data Utilization**: Successfully leverage existing annotations
4. **Comparison Capability**: Clear evaluation framework for both models

## Evaluation

### Test Dataset Structure
The evaluation script expects a directory structure with separate subdirectories for images with and without ticks:

```
test_dataset/
├── with_ticks/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── without_ticks/
    ├── image3.jpg
    ├── image4.jpg
    └── ...
```

### Evaluation Output
The evaluation generates:
- **Visualizations**: Each image with prediction overlay
- **Metrics Report**: JSON file with comprehensive metrics
- **Confidence Distributions**: Plots showing confidence score distributions
- **Confusion Matrix**: Visual representation of classification performance

### Key Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Confidence Analysis**: Distribution of confidence scores for each class

## Next Steps

1. **Full Training**: Train on complete dataset
2. **Model Optimization**: Quantization and pruning
3. **Mobile Export**: Generate deployment-ready models
4. **Performance Testing**: Benchmark on mobile devices
5. **Comparison Evaluation**: Compare with RetinaNet results 