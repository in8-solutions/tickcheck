"""
Data transformations for TensorFlow tick detection model.

This module provides data augmentation and preprocessing functions using Albumentations.
It handles both training and inference transformations for images and bounding boxes.
"""

import albumentations as A
# Removed PyTorch dependency: from albumentations.pytorch import ToTensorV2
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple

def get_transform(config: Dict[str, Any], train: bool = True, inference_only: bool = False) -> A.Compose:
    """Create transformation pipeline for images and bounding boxes.
    
    Args:
        config: Configuration dictionary containing transform settings
        train: Whether to include training augmentations
        inference_only: If True, create simpler transform for inference
        
    Returns:
        Albumentations Compose object with the transformation pipeline
    """
    
    if inference_only:
        # Simple transform for inference
        return A.Compose([
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std'],
                max_pixel_value=255.0
            )
            # Removed ToTensorV2() - will handle tensor conversion in convert_to_tensorflow_format
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    if train:
        # Training transforms with augmentations
        transforms = [
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            )
        ]
        
        # Add augmentations based on config
        aug_config = config.get('augmentation', {}).get('train', {})
        
        if aug_config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if aug_config.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        if aug_config.get('rotate', {}).get('enabled', False):
            limit = aug_config['rotate'].get('limit', 45)
            transforms.append(A.Rotate(limit=limit, p=0.5))
        
        if aug_config.get('scale', {}).get('enabled', False):
            min_scale = aug_config['scale'].get('min', 0.8)
            max_scale = aug_config['scale'].get('max', 1.2)
            transforms.append(A.RandomScale(scale_limit=(min_scale-1, max_scale-1), p=0.5))
            # Add final resize to ensure correct output size
            transforms.append(A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ))
        
        if aug_config.get('brightness', {}).get('enabled', False):
            limit = aug_config['brightness'].get('limit', 0.2)
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=limit,
                contrast_limit=0,
                p=0.5
            ))
        
        if aug_config.get('contrast', {}).get('enabled', False):
            limit = aug_config['contrast'].get('limit', 0.2)
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=limit,
                p=0.5
            ))
        
        if aug_config.get('blur', {}).get('enabled', False):
            limit = aug_config['blur'].get('limit', 3)
            transforms.append(A.GaussianBlur(blur_limit=(3, limit), p=0.3))
        
        if aug_config.get('noise', {}).get('enabled', False):
            limit = aug_config['noise'].get('limit', 0.05)
            transforms.append(A.GaussNoise(std_range=(limit, limit*2), p=0.3))
        
        # Add normalization (removed ToTensorV2)
        transforms.append(A.Normalize(
            mean=config['data']['mean'],
            std=config['data']['std'],
            max_pixel_value=255.0
        ))
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])
        )
    
    else:
        # Validation transforms (no augmentation)
        return A.Compose([
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std'],
                max_pixel_value=255.0
            )
            # Removed ToTensorV2() - will handle tensor conversion in convert_to_tensorflow_format
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def convert_to_tensorflow_format(image: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
    """Convert Albumentations output to TensorFlow format.
    
    Args:
        image: Image tensor from Albumentations
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        labels: Class labels
        
    Returns:
        Tuple of (image_tensor, target_dict)
    """
    # Convert image to TensorFlow format
    if isinstance(image, np.ndarray):
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    else:
        image_tensor = tf.convert_to_tensor(image.numpy(), dtype=tf.float32)
    
    # Ensure image is in channels_last format
    if image_tensor.shape[0] == 3:  # channels_first
        image_tensor = tf.transpose(image_tensor, perm=[1, 2, 0])
    
    # Convert boxes and labels to TensorFlow tensors
    boxes_tensor = tf.convert_to_tensor(boxes, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
    
    # Create target dictionary
    target = {
        'boxes': boxes_tensor,
        'labels': labels_tensor
    }
    
    return image_tensor, target

def resize_boxes(boxes: np.ndarray, original_size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
    """Resize bounding boxes to match new image size.
    
    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        original_size: Original image size (height, width)
        new_size: New image size (height, width)
        
    Returns:
        Resized bounding boxes
    """
    if len(boxes) == 0:
        return boxes
    
    orig_h, orig_w = original_size
    new_h, new_w = new_size
    
    # Calculate scaling factors
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    
    # Scale boxes
    scaled_boxes = boxes.copy()
    scaled_boxes[:, [0, 2]] *= scale_x  # x coordinates
    scaled_boxes[:, [1, 3]] *= scale_y  # y coordinates
    
    return scaled_boxes

def clip_boxes(boxes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Clip bounding boxes to image boundaries.
    
    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        image_size: Image size (height, width)
        
    Returns:
        Clipped bounding boxes
    """
    if len(boxes) == 0:
        return boxes
    
    h, w = image_size
    clipped_boxes = boxes.copy()
    
    # Clip coordinates to image boundaries
    clipped_boxes[:, 0] = np.clip(clipped_boxes[:, 0], 0, w)  # x1
    clipped_boxes[:, 1] = np.clip(clipped_boxes[:, 1], 0, h)  # y1
    clipped_boxes[:, 2] = np.clip(clipped_boxes[:, 2], 0, w)  # x2
    clipped_boxes[:, 3] = np.clip(clipped_boxes[:, 3], 0, h)  # y2
    
    return clipped_boxes

def filter_valid_boxes(boxes: np.ndarray, min_size: int = 1) -> np.ndarray:
    """Filter out invalid bounding boxes.
    
    Args:
        boxes: Bounding boxes in [x1, y1, x2, y2] format
        min_size: Minimum box size (width or height)
        
    Returns:
        Valid bounding boxes
    """
    if len(boxes) == 0:
        return boxes
    
    # Calculate box dimensions
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    # Filter valid boxes
    valid_mask = (widths >= min_size) & (heights >= min_size) & (widths > 0) & (heights > 0)
    
    return boxes[valid_mask] 