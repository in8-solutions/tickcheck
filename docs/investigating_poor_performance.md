# Investigating Poor Model Performance

This document outlines strategies for investigating and improving model performance without modifying the core training code.

## Current Symptoms
- Model trains with decreasing loss
- Validation loss shows 0.0
- No detections during evaluation
- Box score threshold: 0.1
- Box NMS threshold: 0.3
- Box detections per image: 1

## Safe Investigation Steps

### 1. Data Analysis
- Use `verify_annotations.py` to visually inspect training data
- Check distribution of box sizes and positions
- Verify annotation format matches model expectations
- Look for potential class imbalance

### 2. Model Output Analysis
- Run evaluation with `--debug` flag if available
- Save raw model outputs before post-processing
- Check confidence scores distribution
- Verify box coordinates are within image bounds

### 3. Configuration Testing
- Try different confidence thresholds (0.05, 0.1, 0.2)
- Test different NMS thresholds (0.3, 0.5, 0.7)
- Experiment with box_detections_per_img (1, 3, 5)
- Keep original config.yaml as backup

### 4. Training Monitoring
- Review training_history.json for loss patterns
- Check if loss components (classification vs regression) are balanced
- Monitor learning rate changes
- Look for signs of overfitting

### 5. Evaluation Pipeline
- Verify transform pipeline matches training
- Check image preprocessing steps
- Confirm box coordinate transformations
- Validate label mapping

## Safe Testing Approach
1. Create a new config file for testing
2. Keep original files untouched
3. Use quick-test mode for rapid iteration
4. Document all changes and results

## Potential Areas to Investigate
1. Anchor box sizes vs actual tick sizes
2. Learning rate impact on detection confidence
3. Data augmentation effects
4. Model capacity vs dataset size
5. Loss function behavior

## Next Steps
1. Start with data verification
2. Move to configuration testing
3. Analyze model outputs
4. Review training metrics
5. Test evaluation pipeline

Remember: Document all findings and keep original code unchanged until root cause is identified.

## Model Architecture Details

### Backbone and Classification Head
- **ResNet50 Backbone**:
  - Pre-trained feature extractor
  - Outputs class-agnostic feature maps
  - Contains general visual feature representations
  - Number of classes doesn't affect backbone compatibility
  - Currently frozen to prevent overfitting

- **RetinaNet Head**:
  - Takes backbone's feature maps as input
  - Contains:
    - Feature Pyramid Network (FPN) for multi-scale processing
    - Classification subnet for object detection
    - Regression subnet for bounding box refinement
  - Handles the 2-class classification (tick vs no tick)
  - This is the part being trained while backbone is frozen

### Training Strategy Benefits
1. Prevents overfitting by keeping backbone's general-purpose features stable
2. Reduces number of trainable parameters significantly
3. Allows model to focus on learning tick-specific detection patterns
4. Maintains backbone's ability to extract meaningful features
5. Compatible with 2-class initialization despite backbone being pre-trained on many classes 

## Model Architecture and Training

### Head Weights and Their Importance
In object detection models like RetinaNet, the "head" refers to the final layers of the network that make the actual predictions. There are typically two heads:
1. **Classification Head**: Predicts what class an object is (e.g., tick or background)
2. **Regression Head**: Predicts the bounding box coordinates (x, y, width, height)

The head weights are crucial because:
- They are the last layers that process features from the backbone
- They need to be properly initialized to make meaningful predictions
- In our case, we're using pretrained weights for the backbone but training the heads from scratch
- The classification head is particularly important as it determines the model's confidence in its predictions

### Validation Metrics in Object Detection
Object detection models use several metrics to evaluate performance:

1. **IoU (Intersection over Union)**:
   - Measures overlap between predicted and ground truth boxes
   - Range: 0 (no overlap) to 1 (perfect overlap)
   - Formula: Area of Intersection / Area of Union
   - Common threshold: 0.5 (boxes with IoU > 0.5 are considered correct)
   - Advantages:
     - Scale-invariant (works for any box size)
     - Intuitive interpretation
     - Standard in object detection

2. **Precision**:
   - Ratio of correct predictions to total predictions
   - Formula: True Positives / (True Positives + False Positives)
   - Measures how many of the model's predictions are correct
   - Range: 0 to 1 (higher is better)

3. **Recall**:
   - Ratio of correctly detected objects to total ground truth objects
   - Formula: True Positives / (True Positives + False Negatives)
   - Measures how many of the actual objects the model found
   - Range: 0 to 1 (higher is better)

4. **Mean Average Precision (mAP)**:
   - More comprehensive metric that considers precision at different recall levels
   - Commonly used in object detection benchmarks
   - Takes into account both precision and recall
   - More stable than individual precision/recall values

### Analysis of Current Validation Output

Looking at the detailed validation output, we can observe several concerning patterns:

1. **Prediction Confidence vs Localization**:
   - The model is making predictions with relatively high confidence scores (0.62-0.72)
   - However, the IoU values are very low (mostly < 0.3, many at 0.0)
   - This suggests the model is confident but wrong in its predictions

2. **Box Prediction Patterns**:
   - The model consistently predicts one box per image
   - The predicted boxes are often in completely different regions than ground truth
   - Example from Batch 1, Image 3:
     - Prediction: [24.95, 497.84, 510.45, 890.19]
     - Ground Truth: [487.42, 364.16, 698.36, 793.23]
     - IoU: 0.0248 (very poor overlap)

3. **Systematic Issues**:
   - All predictions have confidence > 0.6, suggesting the model is overconfident
   - No predictions meet the IoU threshold of 0.5
   - The model is making predictions, but they're consistently misaligned

4. **Potential Causes**:
   a. **Anchor Box Issues**:
      - Current anchor sizes [16, 32, 64] might be too small
      - Ground truth boxes are much larger (often 200-300 pixels)
      - Need to analyze ground truth box sizes to set appropriate anchors

   b. **Regression Head Problems**:
      - The regression head might not be learning proper box transformations
      - Could be due to improper initialization or learning rate
      - Might need to adjust the regression loss weights

   c. **Feature Extraction**:
      - The backbone might not be extracting relevant features
      - Could be due to improper fine-tuning or frozen layers
      - Consider unfreezing more layers or using a different backbone

5. **Immediate Actions Needed**:
   1. Analyze ground truth box sizes to set appropriate anchor sizes
   2. Consider increasing the number of detections per image
   3. Adjust the learning rate for the regression head
   4. Implement proper box regression loss weighting
   5. Add visualization of predicted vs ground truth boxes

### Current Issues and Potential Solutions

1. **Model Configuration**:
   - Current anchor sizes: [16, 32, 64]
   - Current anchor ratios: [0.5, 1.0, 2.0]
   - Box score threshold: 0.05
   - Box NMS threshold: 0.5
   - Box detections per image: 1

2. **Training Process**:
   - Using pretrained ResNet50 backbone
   - Training heads from scratch
   - Learning rate: 0.000001
   - Batch size: 4
   - Number of epochs: 2 (in quick test mode)

3. **Performance Issues**:
   - High training loss (1532.7442)
   - Zero validation metrics (IoU, Precision, Recall)
   - Model making predictions but with poor localization

4. **Potential Solutions**:
   a. **Anchor Configuration**:
      - Adjust anchor sizes to better match our object sizes
      - Consider adding more anchor ratios
      - Analyze ground truth box sizes to inform anchor design

   b. **Training Strategy**:
      - Increase number of detections per image
      - Adjust learning rate schedule
      - Consider using a different optimizer
      - Implement learning rate warmup

   c. **Model Architecture**:
      - Consider using a different backbone
      - Add additional data augmentation
      - Implement feature pyramid network improvements

   d. **Data Quality**:
      - Verify annotation quality
      - Check for class imbalance
      - Analyze distribution of object sizes

5. **Next Steps**:
   1. Analyze ground truth box sizes to inform anchor configuration
   2. Implement proper learning rate scheduling
   3. Add more comprehensive data augmentation
   4. Consider using a different backbone architecture
   5. Implement proper validation metrics tracking

6. **Monitoring and Debugging**:
   - Track training and validation metrics
   - Visualize predictions during training
   - Monitor gradient flow
   - Check for numerical stability issues

7. **Code Improvements**:
   - Add proper logging
   - Implement checkpointing
   - Add visualization tools
   - Improve error handling

8. **Documentation**:
   - Document model architecture
   - Document training process
   - Document validation metrics
   - Document debugging process

## Conclusion
The model is currently underperforming, but there are several potential solutions to explore. The next steps should focus on:
1. Analyzing the data to inform model configuration
2. Implementing proper training strategies
3. Improving the model architecture
4. Adding comprehensive monitoring and debugging tools

## References
1. RetinaNet paper: https://arxiv.org/abs/1708.02002
2. PyTorch documentation: https://pytorch.org/docs/stable/index.html
3. COCO dataset: https://cocodataset.org/
4. Object detection metrics: https://github.com/rafaelpadilla/Object-Detection-Metrics 