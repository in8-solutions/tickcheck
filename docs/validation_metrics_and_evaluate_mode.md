# Validation Metrics and Evaluate Mode Challenges

## The Train vs Evaluate Mode Problem

### Why Cursor Keeps Breaking the Code

The fundamental issue is that **RetinaNet behaves differently in train vs eval mode**, and this creates a cascade of problems:

1. **Different Output Formats**: 
   - **Train mode**: Returns loss dictionaries with classification and regression losses
   - **Eval mode**: Returns prediction dictionaries with boxes, scores, and labels

2. **Data Type Inconsistencies**:
   - Train mode outputs are tensors optimized for backpropagation
   - Eval mode outputs are processed for inference (often converted to numpy/cpu)

3. **Model State Dependencies**:
   - Batch normalization layers behave differently
   - Dropout layers are disabled in eval mode
   - Some layers may have different forward passes

### The Cascade Effect

When you switch `model.eval()`, the following breaks:
- Loss computation (no losses in eval mode)
- Metric calculations expecting loss dictionaries
- Training loops that expect specific tensor formats
- Validation functions that try to compute losses on eval outputs

## Better Validation Metrics for Binary Tick Detection

### 1. Image-Level Classification Metrics

Since this is fundamentally a binary classifier (tick vs no tick), focus on image-level metrics:

```python
# Image-level binary classification metrics
def calculate_image_level_metrics(predictions, ground_truth):
    """
    Calculate metrics treating each image as a binary classification problem.
    
    Args:
        predictions: List of prediction dicts from model
        ground_truth: List of ground truth dicts
    
    Returns:
        dict: Image-level metrics
    """
    image_predictions = []
    image_ground_truth = []
    
    for pred, gt in zip(predictions, ground_truth):
        # Image has tick if any detection above threshold
        has_tick_pred = len(pred['boxes']) > 0
        has_tick_gt = len(gt['boxes']) > 0
        
        image_predictions.append(has_tick_pred)
        image_ground_truth.append(has_tick_gt)
    
    # Calculate binary classification metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        image_ground_truth, image_predictions, average='binary'
    )
    accuracy = accuracy_score(image_ground_truth, image_predictions)
    
    return {
        'image_precision': precision,
        'image_recall': recall,
        'image_f1': f1,
        'image_accuracy': accuracy
    }
```

### 2. Detection-Level Metrics (Alternative to IoU)

Instead of IoU, consider these metrics:

#### A. Confidence-Based Metrics
```python
def confidence_analysis(predictions, ground_truth, confidence_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """Analyze performance across different confidence thresholds."""
    results = {}
    
    for threshold in confidence_thresholds:
        filtered_predictions = []
        for pred in predictions:
            # Filter by confidence threshold
            mask = pred['scores'] >= threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'scores': pred['scores'][mask],
                'labels': pred['labels'][mask]
            }
            filtered_predictions.append(filtered_pred)
        
        # Calculate metrics at this threshold
        metrics = calculate_image_level_metrics(filtered_predictions, ground_truth)
        results[f'threshold_{threshold}'] = metrics
    
    return results
```

#### B. Distance-Based Metrics
```python
def calculate_center_distance_metrics(predictions, ground_truth, max_distance=50):
    """
    Calculate metrics based on center point distances rather than IoU.
    More suitable for small objects like ticks.
    """
    def get_center(box):
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    
    def calculate_distance(center1, center2):
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, gt in zip(predictions, ground_truth):
        pred_centers = [get_center(box) for box in pred['boxes']]
        gt_centers = [get_center(box) for box in gt['boxes']]
        
        # Match predictions to ground truth based on distance
        matched_gt = set()
        for pred_center in pred_centers:
            best_match = None
            best_distance = float('inf')
            
            for i, gt_center in enumerate(gt_centers):
                if i not in matched_gt:
                    distance = calculate_distance(pred_center, gt_center)
                    if distance < best_distance and distance <= max_distance:
                        best_distance = distance
                        best_match = i
            
            if best_match is not None:
                matched_gt.add(best_match)
                true_positives += 1
            else:
                false_positives += 1
        
        false_negatives += len(gt_centers) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'distance_precision': precision,
        'distance_recall': recall,
        'distance_f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
```

### 3. Tick-Specific Metrics

#### A. Size-Aware Metrics
```python
def calculate_size_aware_metrics(predictions, ground_truth):
    """
    Calculate metrics considering tick size variations.
    Small ticks are harder to detect, so weight them more.
    """
    def get_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    small_tick_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    large_tick_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    # Define size thresholds (in pixels)
    small_threshold = 100  # Small tick area
    large_threshold = 500  # Large tick area
    
    for pred, gt in zip(predictions, ground_truth):
        # Process ground truth by size
        for gt_box in gt['boxes']:
            area = get_area(gt_box)
            if area <= small_threshold:
                small_tick_metrics['fn'] += 1
            elif area <= large_threshold:
                large_tick_metrics['fn'] += 1
        
        # Process predictions by size
        for pred_box in pred['boxes']:
            area = get_area(pred_box)
            # Simplified: assume prediction is correct if area matches
            if area <= small_threshold:
                small_tick_metrics['tp'] += 1
            elif area <= large_threshold:
                large_tick_metrics['tp'] += 1
    
    # Calculate weighted metrics
    small_precision = small_tick_metrics['tp'] / (small_tick_metrics['tp'] + small_tick_metrics['fp']) if (small_tick_metrics['tp'] + small_tick_metrics['fp']) > 0 else 0
    large_precision = large_tick_metrics['tp'] / (large_tick_metrics['tp'] + large_tick_metrics['fp']) if (large_tick_metrics['tp'] + large_tick_metrics['fp']) > 0 else 0
    
    return {
        'small_tick_precision': small_precision,
        'large_tick_precision': large_precision,
        'weighted_precision': (small_precision * 0.7 + large_precision * 0.3)  # Weight small ticks more
    }
```

## Proper Validation Implementation Strategy

### 1. Separate Validation Function

Create a dedicated validation function that doesn't try to compute losses:

```python
def validate_model(model, data_loader, device, confidence_threshold=0.5):
    """
    Proper validation function that handles eval mode correctly.
    """
    model.eval()  # Set to evaluation mode
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc='Validating'):
            images = [image.to(device) for image in images]
            
            # Get predictions (not losses)
            predictions = model(images)
            
            # Process predictions for metric calculation
            for pred, target in zip(predictions, targets):
                # Filter by confidence threshold
                mask = pred['scores'] >= confidence_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][mask].cpu().numpy(),
                    'scores': pred['scores'][mask].cpu().numpy(),
                    'labels': pred['labels'][mask].cpu().numpy()
                }
                
                all_predictions.append(filtered_pred)
                all_ground_truth.append({
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                })
    
    # Calculate various metrics
    metrics = {}
    
    # Image-level metrics
    image_metrics = calculate_image_level_metrics(all_predictions, all_ground_truth)
    metrics.update(image_metrics)
    
    # Distance-based metrics
    distance_metrics = calculate_center_distance_metrics(all_predictions, all_ground_truth)
    metrics.update(distance_metrics)
    
    # Confidence analysis
    confidence_metrics = confidence_analysis(all_predictions, all_ground_truth)
    metrics.update(confidence_metrics)
    
    return metrics
```

### 2. Training Loop Integration

Modify your training loop to use the proper validation:

```python
def train_with_proper_validation(model, train_loader, val_loader, optimizer, device, epochs):
    """
    Training loop with proper validation that doesn't break.
    """
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        
        # Validation phase - use separate function
        val_metrics = validate_model(model, val_loader, device)
        
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Image F1: {val_metrics['image_f1']:.4f}")
        print(f"  Val Distance F1: {val_metrics['distance_f1']:.4f}")
```

## Key Principles to Avoid Breaking Code

1. **Never compute losses in eval mode** - Use separate validation functions
2. **Handle data type conversions explicitly** - Convert tensors to numpy when needed
3. **Use different metrics for different purposes** - Image-level vs detection-level
4. **Test validation functions independently** - Don't integrate until they work standalone
5. **Keep train and eval logic separate** - Don't try to reuse training code for validation

## Recommended Validation Workflow

1. **Start with image-level binary classification metrics** (simplest)
2. **Add distance-based detection metrics** (more sophisticated)
3. **Implement confidence threshold analysis** (for model tuning)
4. **Add tick-specific metrics** (domain knowledge)

This approach avoids the train/eval mode confusion and provides meaningful metrics for your binary tick detection problem. 