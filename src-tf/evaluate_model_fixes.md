# Evaluate Model Fixes - Backward Compatible Changes

## Overview
The `evaluate_model.py` script needs fixes to work with the current TensorFlow RetinaNet implementation. These changes are designed to be backward-compatible and work with both the current image-level classification model and future anchor-level detection models.

## Current Issues

### 1. **Broken Prediction Decoding**
**Problem:** The `predict_image()` method returns empty arrays instead of actual predictions.

**Current Code:**
```python
# Process predictions (simplified for now)
# In a full implementation, you would decode anchors and apply NMS
return {
    'boxes': np.array([]),  # Placeholder
    'scores': np.array([]),  # Placeholder
    'labels': np.array([])   # Placeholder
}
```

**Fix Needed:** Implement proper prediction decoding that works with both model types.

### 2. **Wrong Config Path**
**Problem:** Hardcoded to `'config.yaml'` instead of `'config-mac.yaml'`.

**Current Code:**
```python
config_path = 'config.yaml'
```

**Fix Needed:** Use the correct config file path.

### 3. **Missing Model Type Detection**
**Problem:** No way to distinguish between image-level and anchor-level models.

**Fix Needed:** Add detection logic to handle both model types appropriately.

## Model Architecture Analysis

### Current Model Output
- **Classification Head:** `(batch_size, total_anchors, num_classes)` - per-anchor class predictions
- **Regression Head:** `(batch_size, total_anchors, 4)` - per-anchor bounding box coordinates
- **Loss Function:** Aggregates to image-level "has tick or not" classification

### Model Types to Support

#### Type 1: Image-Level Classification (Current)
- **Purpose:** "Does this image contain ticks?"
- **Output:** Single confidence score per image
- **Bounding Boxes:** Single box covering whole image if tick detected
- **Use Case:** Quick screening, binary classification

#### Type 2: Anchor-Level Detection (Future)
- **Purpose:** "Where are the ticks in this image?"
- **Output:** Multiple bounding boxes with confidence scores
- **Bounding Boxes:** Precise locations of individual ticks
- **Use Case:** Detailed analysis, multiple tick detection

## Proposed Backward-Compatible Fixes

### 1. **Fix Config Path**
```python
def __init__(self, model_path, confidence_threshold=None):
    # Load configuration
    config_path = 'config-mac.yaml'  # Fixed path
    self.config = load_config(config_path)
```

### 2. **Add Model Type Detection**
```python
def is_image_level_model(self):
    """Detect if model is trained for image-level classification."""
    # Check model architecture or training history
    # For now, assume current model is image-level
    return True
```

### 3. **Implement Dual Prediction Strategy**
```python
def predict_image(self, image_path):
    """Run prediction on a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Apply transforms
    transformed = self.transform(image=image_np)
    image_tensor = transformed['image']
    
    # Convert to TensorFlow format
    if hasattr(image_tensor, 'numpy'):
        image_tensor = image_tensor.numpy()
    
    # Ensure correct shape
    if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:
        image_tensor = np.transpose(image_tensor, (1, 2, 0))
    
    # Add batch dimension
    image_tensor = np.expand_dims(image_tensor, axis=0)
    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    
    # Get predictions
    predictions = self.model(image_tensor, training=False)
    
    # Handle different model types
    if self.is_image_level_model():
        return self.extract_image_level_prediction(predictions, image_np.shape)
    else:
        return self.decode_anchor_predictions(predictions, image_np.shape)
```

### 4. **Image-Level Prediction Extraction**
```python
def extract_image_level_prediction(self, predictions, image_shape):
    """Extract image-level prediction from current model."""
    # Get classification outputs
    cls_outputs = predictions['classification']  # (1, total_anchors, num_classes)
    
    # Aggregate to image-level prediction
    # Take maximum probability across all anchors for each class
    image_predictions = tf.reduce_max(cls_outputs, axis=1)  # (1, num_classes)
    
    # Get confidence for tick class (class 1)
    tick_confidence = image_predictions[0, 1].numpy()
    
    # Determine if image has ticks
    has_tick = tick_confidence >= self.confidence_threshold
    
    if has_tick:
        # Create bounding box covering whole image
        height, width = image_shape[:2]
        boxes = np.array([[0, 0, width, height]])  # [x1, y1, x2, y2]
        scores = np.array([tick_confidence])
        labels = np.array([1])  # tick class
    else:
        # No ticks detected
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])
    
    return {
        'boxes': boxes,
        'scores': scores,
        'labels': labels
    }
```

### 5. **Future: Anchor-Level Prediction Decoding**
```python
def decode_anchor_predictions(self, predictions, image_shape):
    """Decode anchor-based predictions (for future models)."""
    # This will be implemented when switching to anchor-level training
    # For now, return empty arrays to maintain compatibility
    return {
        'boxes': np.array([]),
        'scores': np.array([]),
        'labels': np.array([])
    }
```

## Implementation Phases

### Phase 1: Backward Compatibility (Immediate)
1. **Fix config path** - use `config-mac.yaml`
2. **Add model type detection** - identify current vs future models
3. **Implement image-level prediction extraction** - work with current model
4. **Convert to bounding box format** - single box for whole image if tick detected

### Phase 2: Future Enhancement (Later)
1. **Implement anchor decoding** - for when you switch to anchor-level training
2. **Add Non-Maximum Suppression (NMS)** - remove duplicate detections
3. **Support multiple detections** - handle multiple ticks per image

## Benefits

### Backward Compatibility
- **Works with current model** - no retraining needed
- **Same evaluation interface** - scripts don't need to change
- **Preserves all metrics** - sensitivity, specificity, etc.

### Future-Proof
- **Ready for improvements** - easy to extend
- **Supports both model types** - seamless transition
- **Maintains evaluation pipeline** - consistent interface

## Testing Strategy

### Test with Current Model
1. **Run evaluation** on test images
2. **Verify metrics** - sensitivity, specificity, etc.
3. **Check visualizations** - bounding boxes on images
4. **Validate reports** - JSON and summary outputs

### Test with Future Model
1. **Implement anchor decoding**
2. **Add NMS functionality**
3. **Test multiple detection scenarios**
4. **Compare performance metrics**

## Files to Modify

1. **`evaluate_model.py`** - Main evaluation script
2. **`model.py`** - May need prediction helper methods
3. **`utils.py`** - May need NMS implementation

## Notes

- **Current model** uses image-level classification for simplicity
- **Future model** will use full anchor-level detection for precision
- **Evaluation script** should handle both seamlessly
- **No breaking changes** to existing interfaces 