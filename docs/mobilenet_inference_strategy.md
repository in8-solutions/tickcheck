# MobileNet Inference Strategy

## Overview

Unlike RetinaNet which performs object detection directly on full images, our MobileNet model is trained on cropped tick samples and requires a sliding window approach for inference. This document outlines the strategy, implementation approach, and key considerations.

## Architecture Comparison

### RetinaNet (Original)
- **Input**: Full image (e.g., 2048x2048)
- **Output**: Bounding boxes with confidence scores
- **Inference**: Single forward pass per image
- **Memory**: High (large model, full image processing)

### MobileNet (Current)
- **Input**: Cropped windows (224x224)
- **Output**: Binary classification (tick/no tick)
- **Inference**: Multiple forward passes per image (sliding window)
- **Memory**: Low (small model, small windows)

## Proposed Implementation

### Phase 1: Basic Sliding Window

**Parameters:**
- **Input Image**: 2048x2048 pixels
- **Window Size**: 500x500 pixels
- **Stride**: 100 pixels
- **Model Input**: 224x224 (resized from 500x500)
- **Windows per Image**: ~200 (16x16 grid with overlap)

**Processing Flow:**
1. Load full image
2. Extract overlapping 500x500 windows
3. Resize each window to 224x224
4. Run MobileNet inference on each window
5. Generate confidence scores
6. Create heatmap visualization

**Performance Estimate:**
- **Processing Time**: ~20 seconds per image
- **Memory Usage**: Low (processes small windows)
- **Device Compatibility**: Phones <5 years old

### Visualization Strategy

**Heatmap Generation:**
- Color-code areas based on tick probability
- Green: Low probability (0-0.3)
- Yellow: Medium probability (0.3-0.7)
- Red: High probability (0.7-1.0)
- Overlay heatmap on original image

**User Interface:**
- Show original image with color-coded overlay
- Display confidence scores for selected areas
- Allow threshold adjustment for sensitivity

## Key Advantages

1. **Mobile-Friendly**: Small memory footprint, efficient processing
2. **Visual Feedback**: Heatmap provides intuitive tick location indication
3. **Flexible Detection**: Adjustable confidence thresholds
4. **Scalable**: Can optimize parameters based on performance requirements
5. **Battery Efficient**: Smaller model, targeted processing

## Identified Risks and Gaps

### Technical Risks

**1. Performance Bottlenecks**
- **Risk**: 20 seconds per image may be too slow for user acceptance
- **Impact**: High - could make app unusable
- **Mitigation**: Optimize window size, stride, and batch processing

**2. False Positive/Negative Balance**
- **Risk**: Sliding window approach may miss ticks at window boundaries
- **Impact**: Medium - affects detection accuracy
- **Mitigation**: Overlapping windows, multi-scale detection

**3. Memory Management**
- **Risk**: Processing many windows could cause memory issues on older devices
- **Impact**: Medium - app crashes on low-end devices
- **Mitigation**: Batch processing, memory monitoring

**4. Model Generalization**
- **Risk**: Model trained on cropped ticks may not generalize to full images
- **Impact**: High - poor real-world performance
- **Mitigation**: Test with full images, consider fine-tuning

### Implementation Gaps

**1. Window Boundary Issues**
- **Gap**: Ticks spanning multiple windows may be missed or double-counted
- **Solution**: Implement overlap detection and merging logic

**2. Scale Sensitivity**
- **Gap**: Fixed 500x500 windows may miss very small or large ticks
- **Solution**: Multi-scale window approach (300x300, 500x500, 700x700)

**3. Edge Cases**
- **Gap**: Ticks at image edges, unusual orientations, partial visibility
- **Solution**: Comprehensive testing with edge case images

**4. Performance Optimization**
- **Gap**: No GPU acceleration strategy for mobile inference
- **Solution**: Implement batch processing, GPU utilization

### User Experience Gaps

**1. Processing Feedback**
- **Gap**: Users need to wait 20 seconds without clear progress indication
- **Solution**: Progress bar, estimated time remaining

**2. Result Interpretation**
- **Gap**: Heatmap may be confusing for non-technical users
- **Solution**: Clear legend, tutorial, confidence explanations

**3. False Alarm Handling**
- **Gap**: High false positive rate could reduce user trust
- **Solution**: Adjustable sensitivity, user feedback collection

## Optimization Strategies

### Phase 2: Advanced Optimizations

**1. Smart Tiling**
- Use low-resolution preview to identify regions of interest
- Focus high-resolution processing on promising areas
- Reduce total processing time by 50-70%

**2. Multi-Scale Detection**
- Process windows at multiple scales (300x300, 500x500, 700x700)
- Combine results for better detection accuracy
- Handle varying tick sizes more effectively

**3. Batch Processing**
- Group multiple windows for parallel processing
- Utilize GPU more efficiently
- Reduce per-window overhead

**4. Early Exit Strategies**
- Stop processing if high-confidence tick is found
- Implement confidence thresholds for early termination
- Balance speed vs. thoroughness

### Phase 3: Advanced Features

**1. Real-time Processing**
- Process video streams for live tick detection
- Implement frame skipping and temporal consistency
- Target <5 second processing time

**2. Offline Mode**
- Pre-download model for offline use
- Cache processed results
- Reduce dependency on network connectivity

**3. User Feedback Integration**
- Collect user corrections to improve model
- Implement active learning for model updates
- Build user-specific confidence thresholds

## Success Metrics

### Technical Metrics
- **Processing Time**: Target <10 seconds per image
- **Accuracy**: >90% detection rate, <10% false positive rate
- **Memory Usage**: <500MB peak memory
- **Battery Impact**: <5% battery per image processed

### User Experience Metrics
- **User Satisfaction**: >4.0/5.0 app rating
- **Retention Rate**: >70% weekly active users
- **Processing Completion**: >90% of users complete full image processing
- **False Alarm Rate**: <15% user-reported false positives

## Implementation Roadmap

### Phase 1 (Current)
- [ ] Basic sliding window implementation
- [ ] Heatmap visualization
- [ ] Performance baseline measurement
- [ ] User testing with basic functionality

### Phase 2 (Next 2-3 months)
- [ ] Smart tiling optimization
- [ ] Multi-scale detection
- [ ] Batch processing implementation
- [ ] Performance optimization

### Phase 3 (3-6 months)
- [ ] Real-time processing capabilities
- [ ] Advanced user interface features
- [ ] User feedback integration
- [ ] Model fine-tuning based on real-world data

## Conclusion

The MobileNet sliding window approach offers a viable path to mobile tick detection, balancing performance, accuracy, and user experience. While there are significant technical challenges to overcome, the modular approach allows for iterative improvement and optimization based on real-world usage patterns.

The key to success will be managing user expectations around processing time while delivering accurate, actionable results that provide clear value to users in tick-prone environments. 