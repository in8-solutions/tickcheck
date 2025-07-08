# Mac M3 Setup Guide for TensorFlow Tick Detection

This guide will help you set up and run the TensorFlow tick detection model on your Mac M3.

## Prerequisites

1. **macOS 12.3 or later** (required for MPS support)
2. **Python 3.8-3.11** (TensorFlow 2.13+ supports Python 3.8-3.11)
3. **Apple Silicon Mac** (M1, M2, M3, M3 Pro, M3 Max)

## Installation Steps

### 1. Install Python Dependencies

```bash
cd src-tf
pip install -r requirements.txt
```

### 2. Verify TensorFlow Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}'); print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')"
```

You should see output like:
```
TensorFlow 2.13.0
GPU devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 3. Test Your Setup

Run the comprehensive test script:

```bash
python test_mac_setup.py
```

This will test:
- TensorFlow installation
- MPS (Metal Performance Shaders) support
- Model creation
- Data loading
- Training step

## Configuration

The Mac-specific configuration is in `config-mac.yaml`. Key differences from CUDA:

```yaml
training:
  device: "mps"  # Use Metal Performance Shaders
  batch_size: 4  # Reduced for Mac memory
  num_workers: 2 # Reduced for Mac
  use_amp: true  # Mixed precision works well on M3
```

## Running Training

### Quick Test (Recommended First)
```bash
python train.py --config config-mac.yaml --quick-test
```

### Full Training
```bash
python train.py --config config-mac.yaml
```

### Resume Training
```bash
python train.py --config config-mac.yaml --resume outputs/training/checkpoints/best_model.h5
```

## Performance Tips for Mac M3

### 1. Memory Management
- Start with `batch_size: 4` and increase if memory allows
- Use mixed precision training (`use_amp: true`)
- Enable memory growth for MPS

### 2. Data Loading
- Reduce `num_workers` to 2-4 for Mac
- Disable `pin_memory` (not needed on Mac)
- Use `persistent_workers: false`

### 3. Model Optimization
- Use smaller input sizes for testing (512x512 instead of 1024x1024)
- Consider freezing backbone initially
- Use gradient checkpointing for large models

## Troubleshooting

### Common Issues

1. **"No module named 'tensorflow'"**
   ```bash
   pip install tensorflow>=2.13.0
   ```

2. **MPS not available**
   - Ensure macOS 12.3+
   - Check TensorFlow version (2.13+ required)
   - Verify Apple Silicon Mac

3. **Memory issues**
   - Reduce batch size
   - Use smaller input images
   - Enable mixed precision

4. **Slow training**
   - Ensure MPS is being used (check device output)
   - Use mixed precision training
   - Optimize data pipeline

### Performance Monitoring

Check if MPS is being used:
```python
import tensorflow as tf
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
print(f"Current device: {tf.config.get_visible_devices()}")
```

## Expected Performance

On Mac M3:
- **Training speed**: ~2-4x faster than CPU
- **Memory usage**: ~8-16GB depending on batch size
- **Model size**: ~100-200MB for RetinaNet

## Converting to TensorFlow Lite

After training, convert to TFLite for mobile deployment:

```bash
python convert_to_tflite.py \
    --model outputs/training/checkpoints/best_model.h5 \
    --config config-mac.yaml \
    --output tick_detector.tflite \
    --quantize
```

## Evaluation

Test your trained model:

```bash
python evaluate_model.py \
    --model outputs/training/checkpoints/best_model.h5 \
    --input ../test_images \
    --output evaluation_results \
    --confidence 0.5
```

## Advanced Configuration

### Custom Batch Size
Edit `config-mac.yaml`:
```yaml
training:
  batch_size: 8  # Increase if memory allows
```

### Custom Input Size
```yaml
data:
  input_size: [768, 768]  # Smaller for faster training
```

### Disable Mixed Precision
```yaml
training:
  use_amp: false  # If you encounter issues
```

## Support

If you encounter issues:
1. Run `python test_mac_setup.py` to diagnose
2. Check TensorFlow version compatibility
3. Verify macOS version
4. Check available memory

## Performance Comparison

| Device | Training Speed | Memory Usage | Batch Size |
|--------|---------------|--------------|------------|
| Mac M3 CPU | 1x | Low | 2-4 |
| Mac M3 MPS | 2-4x | Medium | 4-8 |
| CUDA GPU | 4-8x | High | 8-16 |

The M3 provides excellent performance for TensorFlow training, especially with the MPS backend! 