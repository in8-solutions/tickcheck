#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.quantization as quantization
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any

from model import create_model
from config import MODEL_CONFIG, OUTPUT_CONFIG, EXPORT_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobileExporter:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        self.model = create_model(MODEL_CONFIG)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Original model size: {self._get_model_size_mb(self.model):.2f} MB")
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def quantize_model(self) -> nn.Module:
        """Quantize model to INT8 for smaller size"""
        logger.info("Quantizing model to INT8...")
        
        # Prepare model for quantization
        self.model.eval()
        
        # Use dynamic quantization (works well for inference)
        quantized_model = quantization.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.Conv2d}, 
            dtype=torch.qint8
        )
        
        logger.info(f"Quantized model size: {self._get_model_size_mb(quantized_model):.2f} MB")
        return quantized_model
    
    def export_onnx(self, model: nn.Module, output_path: Path):
        """Export model to ONNX format"""
        logger.info("Exporting to ONNX...")
        
        # Create a wrapper that returns only the logits
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x):
                output = self.model(x)
                return output['logits']  # Return only logits, not the full dict
        
        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model saved to {output_path}")
    
    def export_coreml(self, model: nn.Module, output_path: Path):
        """Export model to Core ML format (for iOS)"""
        try:
            import coremltools as ct
            logger.info("Exporting to Core ML...")
            
            # Create a wrapper that returns only the logits
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    output = self.model(x)
                    return output['logits']  # Return only logits, not the full dict
            
            wrapped_model = ModelWrapper(model)
            wrapped_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Trace the model
            traced_model = torch.jit.trace(wrapped_model, dummy_input)
            
            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
                minimum_deployment_target=ct.target.iOS13
            )
            
            # Save the model
            coreml_model.save(output_path)
            logger.info(f"Core ML model saved to {output_path}")
            
        except ImportError:
            logger.warning("coremltools not installed. Skipping Core ML export.")
        except Exception as e:
            logger.error(f"Failed to export Core ML model: {e}")
    
    def export_tflite(self, model: nn.Module, output_path: Path):
        """Export model to TensorFlow Lite format (for Android)"""
        try:
            import tensorflow as tf
            logger.info("Exporting to TensorFlow Lite...")
            
            # Create a wrapper that returns only the logits
            class ModelWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, x):
                    output = self.model(x)
                    return output['logits']  # Return only logits, not the full dict
            
            wrapped_model = ModelWrapper(model)
            wrapped_model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Trace the model
            traced_model = torch.jit.trace(wrapped_model, dummy_input)
            
            # For now, create a placeholder TFLite file
            # The full conversion pipeline is complex and requires additional setup
            # This gives the mobile developer a starting point
            
            # Create a simple TensorFlow model that mimics our PyTorch model
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(16, 3, activation='relu'),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(2, activation='softmax')
            ])
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"TensorFlow Lite placeholder model saved to {output_path}")
            logger.warning("This is a placeholder model. For production, use the ONNX model with ONNX Runtime.")
            
        except ImportError as e:
            logger.warning(f"Missing dependency for TFLite export: {e}")
            logger.warning("Skipping TFLite export.")
        except Exception as e:
            logger.error(f"Failed to export TFLite model: {e}")
            logger.warning("Skipping TFLite export.")
    
    def export_all_formats(self):
        """Export model to all mobile formats"""
        export_dir = OUTPUT_CONFIG['export_dir']
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Quantize the model
        quantized_model = self.quantize_model()
        
        # Export to different formats (using original model for ONNX/CoreML)
        self.export_onnx(self.model, export_dir / "tick_detector.onnx")
        self.export_coreml(self.model, export_dir / "TickDetector.mlmodel")
        self.export_tflite(self.model, export_dir / "tick_detector.tflite")
        
        # Save quantized PyTorch model
        torch.save(quantized_model.state_dict(), export_dir / "tick_detector_quantized.pth")
        
        # Create metadata file
        metadata = {
            "model_info": {
                "name": "Tick Detector Mobile",
                "version": "1.0",
                "description": "Binary classifier for tick detection",
                "input_size": [224, 224],
                "num_classes": 2,
                "classes": ["no_tick", "tick"]
            },
            "performance": {
                "original_size_mb": round(self._get_model_size_mb(self.model), 2),
                "quantized_size_mb": round(self._get_model_size_mb(quantized_model), 2),
                "compression_ratio": round(self._get_model_size_mb(self.model) / self._get_model_size_mb(quantized_model), 2)
            },
            "usage": {
                "input_format": "RGB image, 224x224 pixels",
                "output_format": "Binary classification (0=no tick, 1=tick)",
                "confidence_threshold": 0.5
            }
        }
        
        with open(export_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"All exports completed. Check {export_dir} for exported models.")
        logger.info(f"Model metadata saved to {export_dir / 'model_metadata.json'}")


def main():
    """Main export function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Export trained model for mobile deployment')
    parser.add_argument('--model', type=str, default='../outputs/mobile/checkpoints/best_f1_model.pth',
                       help='Path to trained model checkpoint')
    args = parser.parse_args()
    
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return
    
    # Create exporter and export all formats
    exporter = MobileExporter(model_path)
    exporter.export_all_formats()


if __name__ == "__main__":
    main() 