# Mobile Tick Detection Design Document

## Overview
Transform the existing RetinaNet-based tick detection system into a lightweight binary classifier suitable for mobile deployment on iOS and Android devices, while preserving the ability to compare against the original RetinaNet model.

## Key Requirements

### 1. Dataset Preservation
- **Valuable Annotations**: 10,000+ manually reviewed images with bounding box annotations
- **Binary Classification**: Convert from object detection to "tick present" vs "no tick" classification
- **Data Utilization**: Use bounding box annotations to create positive/negative training samples

### 2. Model Comparison
- **RetinaNet Baseline**: Maintain ability to compare new mobile model against existing RetinaNet
- **Evaluation Pipeline**: Shared evaluation framework for both models
- **Performance Metrics**: Consistent metrics across both approaches

### 3. Mobile Deployment
- **Target Platforms**: Recent iOS and Android devices
- **Model Size**: <10MB for app store compliance
- **Inference Speed**: <100ms per image
- **Accuracy**: Maintain high precision/recall comparable to RetinaNet

## Technical Approach

### 1. Data Pipeline Transformation

#### From Object Detection to Binary Classification
```
Original: Image + Bounding Boxes → Object Detection
New: Image → Binary Classification (tick/no tick)
```

#### Training-Time Cropping Strategy

**Annotation Format Analysis:**
- COCO format: `[left, top, width, height]`
- Single category: "tick" (id=1)
- Variable image sizes: 375x500, 500x375, 500x500, etc.
- Variable tick sizes: from small (97x86) to large (307x250)

**Positive Samples (Images with ticks):**
```python
# For each image with bounding box annotations
for bbox in annotations:
    # Crop around bounding box with padding
    crop = crop_bbox_with_padding(image, bbox, padding=0.3)
    # Use as positive sample
```

**Negative Samples (Images without ticks):**
```python
# For images without annotations
# Generate multiple random crops
for _ in range(num_negative_crops):
    crop = random_crop(image, size=224)
    # Use as negative sample
```

**Multiple Bounding Boxes Handling:**
- **Images with 0 boxes**: Generate multiple random crops as negative samples
- **Images with 1 box**: Crop around that single box (with padding)
- **Images with multiple boxes**: Crop around each box separately
  - One image → multiple positive samples
  - Each crop gets the same label (tick present)
  - Maximizes training data utilization

**Benefits of Training-Time Cropping:**
- **No storage overhead**: Crops generated on-the-fly
- **Data augmentation**: Different crops each epoch
- **Flexible padding**: Can experiment with different crop sizes
- **Maintains original data**: Annotations stay intact
- **Handles small objects**: Focuses training on relevant regions

**Data Augmentation Strategy:**
- Maintain existing Albumentations pipeline
- Focus on mobile-relevant augmentations (rotation, brightness, contrast)
- Avoid heavy augmentations that might not occur in mobile usage
- Apply augmentations after cropping for better effectiveness

### 2. Model Architecture Options

#### Option A: Lightweight CNN (Recommended)
- **Backbone**: MobileNetV3-Small or EfficientNet-B0
- **Head**: Simple global average pooling + 2-class classifier
- **Size**: ~2-5MB
- **Speed**: <50ms inference

#### Option B: Vision Transformer (ViT-Tiny)
- **Architecture**: ViT-Tiny with patch size 16
- **Head**: Simple classification head
- **Size**: ~5-8MB
- **Speed**: <100ms inference

#### Option C: Hybrid Approach
- **Feature Extraction**: MobileNetV3-Small
- **Classification**: Lightweight transformer or MLP
- **Size**: ~3-6MB
- **Speed**: <75ms inference

### 3. Training Strategy

#### Transfer Learning
1. **Pre-training**: ImageNet weights
2. **Fine-tuning**: Gradual unfreezing of layers
3. **Learning Rate**: Lower LR for pre-trained layers

#### Loss Function
- **Binary Cross-Entropy**: Standard for binary classification
- **Focal Loss**: If class imbalance issues arise
- **Label Smoothing**: For better generalization

#### Evaluation Metrics
- **Primary**: Precision, Recall, F1-Score
- **Secondary**: ROC-AUC, Confusion Matrix
- **Comparison**: Side-by-side with RetinaNet results

### 4. Mobile Deployment Pipeline

#### Model Optimization
1. **Quantization**: INT8 quantization for size reduction
2. **Pruning**: Remove unnecessary weights
3. **Knowledge Distillation**: Use RetinaNet as teacher

#### Export Formats
- **iOS**: Core ML format
- **Android**: TFLite format
- **Cross-platform**: ONNX format

#### Performance Targets
- **Model Size**: <5MB
- **Inference Time**: <50ms on iPhone 12+ / Pixel 6+
- **Memory Usage**: <50MB RAM
- **Battery Impact**: Minimal

## Implementation Plan

### Phase 1: Data Pipeline
1. Create binary classification dataset from existing annotations
2. Implement data loading and augmentation pipeline
3. Validate data distribution and quality

### Phase 2: Model Development
1. Implement lightweight CNN architecture
2. Set up training pipeline with transfer learning
3. Train initial model and establish baseline

### Phase 3: Optimization
1. Model quantization and pruning
2. Performance optimization
3. Mobile export and testing

### Phase 4: Evaluation & Comparison
1. Comprehensive evaluation against RetinaNet
2. Mobile device testing
3. Performance benchmarking

## File Structure

```
.
├── data/                     # Shared data directory (annotations + images)
│   ├── chunk_001/
│   ├── chunk_002/
│   └── ...                   # All chunks with annotations
├── src/                      # Original RetinaNet implementation
│   ├── model.py              # RetinaNet model
│   ├── train.py              # RetinaNet training
│   ├── evaluate_model.py     # RetinaNet evaluation
│   └── ...                   # Other RetinaNet files
├── src-mobilnet/             # Mobile implementation
│   ├── data_pipeline.py      # Binary classification data loading
│   ├── model.py              # Lightweight mobile models
│   ├── train_mobile.py       # Mobile model training
│   ├── evaluate_mobile.py    # Mobile model evaluation
│   ├── compare_models.py     # RetinaNet vs Mobile comparison
│   ├── export_mobile.py      # Mobile model export
│   ├── utils.py              # Mobile-specific utilities
│   └── config.py             # Mobile model configuration
└── config/
    └── mobile_config.yaml    # Mobile model configuration
```

## Success Criteria

1. **Model Performance**: F1-Score > 0.90 (comparable to RetinaNet)
2. **Mobile Compatibility**: <5MB model size, <50ms inference
3. **Data Utilization**: Successfully leverage existing annotations
4. **Comparison Capability**: Clear evaluation framework for both models

## Next Steps

1. Implement binary classification data pipeline
2. Design and implement lightweight model architecture
3. Set up training and evaluation framework
4. Begin model training and optimization 