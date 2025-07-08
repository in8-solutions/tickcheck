"""
RetinaNet model implementation for TensorFlow tick detection.

This module implements a custom RetinaNet model using TensorFlow/Keras.
The model uses a ResNet50-FPN backbone and is designed for efficient single-stage
object detection with balanced handling of class imbalance through Focal Loss.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any, List, Tuple

class FPNBlock(layers.Layer):
    """Feature Pyramid Network block."""
    
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv = layers.Conv2D(filters, 1, padding='same')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
    
    def call(self, x):
        return self.relu(self.bn(self.conv(x)))

class RetinaNetClassificationHead(layers.Layer):
    """Classification head for RetinaNet."""
    
    def __init__(self, num_classes: int, num_anchors: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification layers
        self.conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(num_classes * num_anchors, 3, padding='same')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Reshape to (batch, height * width * num_anchors, num_classes)
        batch_size = tf.shape(x)[0]
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [batch_size, height * width * self.num_anchors, self.num_classes])
        
        return x

class RetinaNetRegressionHead(layers.Layer):
    """Regression head for RetinaNet."""
    
    def __init__(self, num_anchors: int, **kwargs):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors
        
        # Regression layers
        self.conv1 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv5 = layers.Conv2D(4 * num_anchors, 3, padding='same')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Reshape to (batch, height * width * num_anchors, 4)
        batch_size = tf.shape(x)[0]
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        x = tf.reshape(x, [batch_size, height * width * self.num_anchors, 4])
        
        return x

class FocalLoss(keras.losses.Loss):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Convert to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate focal loss
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Calculate loss
        loss = -alpha_weight * focal_weight * tf.math.log(pt)
        
        return tf.reduce_mean(loss)

class SmoothL1Loss(keras.losses.Loss):
    """Smooth L1 Loss for bounding box regression."""
    
    def __init__(self, beta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
    
    def call(self, y_true, y_pred):
        # Convert to float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calculate smooth L1 loss
        diff = tf.abs(y_true - y_pred)
        quadratic = tf.minimum(diff, self.beta)
        linear = diff - quadratic
        
        loss = 0.5 * quadratic ** 2 + self.beta * linear
        
        return tf.reduce_mean(loss)

class RetinaNet(keras.Model):
    """RetinaNet model for object detection."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Model parameters
        self.num_classes = config['model']['num_classes']
        self.anchor_sizes = config['model']['anchor_sizes']
        self.anchor_ratios = config['model']['anchor_ratios']
        self.num_anchors = len(self.anchor_sizes) * len(self.anchor_ratios)
        
        # Backbone (ResNet50)
        self.backbone = keras.applications.ResNet50(
            include_top=False,
            weights='imagenet' if config['model']['pretrained'] else None,
            input_shape=(config['data']['input_size'][0], config['data']['input_size'][1], 3)
        )
        # Create a model to extract intermediate outputs
        self.backbone_feature_extractor = keras.Model(
            inputs=self.backbone.input,
            outputs=[
                self.backbone.get_layer('conv3_block4_out').output,
                self.backbone.get_layer('conv4_block6_out').output,
                self.backbone.get_layer('conv5_block3_out').output,
            ]
        )
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            self.backbone.trainable = False
            self.backbone_feature_extractor.trainable = False
        
        # FPN layers
        self.fpn_layers = []
        for i in range(5):  # P3, P4, P5, P6, P7
            self.fpn_layers.append(FPNBlock(256))
        
        # Classification and regression heads
        self.classification_head = RetinaNetClassificationHead(self.num_classes, self.num_anchors)
        self.regression_head = RetinaNetRegressionHead(self.num_anchors)
        
        # Loss functions
        self.focal_loss = FocalLoss()
        self.smooth_l1_loss = SmoothL1Loss()
    
    def build_fpn(self, c3, c4, c5):
        """Build Feature Pyramid Network."""
        # Lateral connections
        p5 = self.fpn_layers[2](c5)
        p4_lateral = self.fpn_layers[1](c4)
        p5_upsampled = tf.cast(tf.image.resize(p5, tf.shape(c4)[1:3]), p4_lateral.dtype)
        p4 = p4_lateral + p5_upsampled
        p3_lateral = self.fpn_layers[0](c3)
        p4_upsampled = tf.cast(tf.image.resize(p4, tf.shape(c3)[1:3]), p3_lateral.dtype)
        p3 = p3_lateral + p4_upsampled
        # Additional layers
        p6_input = tf.cast(tf.image.resize(p5, tf.shape(p5)[1:3] // 2), p5.dtype)
        p6 = self.fpn_layers[3](p6_input)
        p7_input = tf.cast(tf.image.resize(p6, tf.shape(p6)[1:3] // 2), p6.dtype)
        p7 = self.fpn_layers[4](p7_input)
        return [p3, p4, p5, p6, p7]
    
    def call(self, inputs, training=None):
        """Forward pass."""
        if isinstance(inputs, list):
            # Handle batch of images
            outputs = []
            for image in inputs:
                output = self.single_forward(image, training)
                outputs.append(output)
            return outputs
        else:
            return self.single_forward(inputs, training)
    
    def single_forward(self, image, training=None):
        """Forward pass for a single image."""
        # Backbone features
        c3, c4, c5 = self.backbone_feature_extractor(image)
        
        # FPN
        fpn_features = self.build_fpn(c3, c4, c5)
        
        # Classification and regression
        cls_outputs = []
        reg_outputs = []
        
        for feature in fpn_features:
            cls_output = self.classification_head(feature)
            reg_output = self.regression_head(feature)
            
            cls_outputs.append(cls_output)
            reg_outputs.append(reg_output)
        
        # Concatenate outputs
        cls_output = tf.concat(cls_outputs, axis=1)
        reg_output = tf.concat(reg_outputs, axis=1)
        
        return {
            'classification': cls_output,
            'regression': reg_output
        }
    
    def compute_loss(self, targets, predictions):
        """Compute classification and regression losses for binary tick detection."""
        # Treat this as binary classification: does the image contain a tick or not?
        # This is more appropriate for the goal of determining if an image has ticks
        
        batch_size = tf.shape(targets['labels'])[0]
        total_anchors = tf.shape(predictions['classification'])[1]  # e.g., 327360
        num_classes = tf.shape(predictions['classification'])[2]  # e.g., 2
        
        # Get per-image labels: does this image have any ticks?
        # targets['labels'] shape: (batch_size, num_anchors_per_image)
        image_has_ticks = tf.reduce_any(targets['labels'] > 0, axis=1)  # (batch_size,)
        
        # Convert to one-hot encoding for binary classification
        image_labels_onehot = tf.one_hot(tf.cast(image_has_ticks, tf.int32), 
                                        depth=num_classes, 
                                        dtype=tf.float32)  # (batch_size, 2)
        
        # For classification, aggregate anchor predictions to image-level prediction
        # Take the maximum probability across all anchors for each class
        cls_predictions = predictions['classification']  # (batch_size, total_anchors, num_classes)
        
        # Aggregate anchor predictions to image-level: max probability per class
        image_predictions = tf.reduce_max(cls_predictions, axis=1)  # (batch_size, num_classes)
        
        # Compute classification loss using image-level predictions
        cls_loss = self.focal_loss(image_labels_onehot, image_predictions)
        
        # For regression, use a simple regularization term since we're doing classification
        # In a real implementation, you might want to skip regression loss for binary classification
        reg_predictions = predictions['regression']  # (batch_size, total_anchors, 4)
        reg_loss = tf.reduce_mean(tf.square(reg_predictions)) * 0.01  # Small regularization
        
        return {
            'classification_loss': cls_loss,
            'regression_loss': reg_loss,
            'total_loss': cls_loss + tf.cast(reg_loss, cls_loss.dtype)
        }

def create_model(config: Dict[str, Any]) -> RetinaNet:
    """Create and configure the RetinaNet model."""
    model = RetinaNet(config)
    
    # Compile model
    optimizer = keras.optimizers.Adam(
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    model.compile(
        optimizer=optimizer,
        loss=None,  # We'll handle loss computation manually
        metrics=None
    )
    
    return model 