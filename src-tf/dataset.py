"""
Dataset handling for TensorFlow tick detection model.

This module implements TensorFlow datasets for object detection using the COCO format.
It handles loading images and their corresponding annotations, applying transformations,
and preparing the data for training and validation.

Key Features:
- Supports COCO format annotations
- Handles bounding box normalization
- Provides configurable data augmentation
- Supports both training and inference transformations
- Multi-chunk dataset support
"""

import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import albumentations as A
from typing import Dict, Any, List, Tuple, Optional
from transforms import get_transform, convert_to_tensorflow_format, resize_boxes, clip_boxes, filter_valid_boxes

class DetectionDataset:
    """Dataset for object detection with COCO format annotations."""
    
    def __init__(self, image_dir: str, annotation_file: Optional[str] = None, 
                 annotations: Optional[Dict] = None, transform: Optional[A.Compose] = None,
                 config: Optional[Dict] = None):
        """
        Args:
            image_dir: Directory containing images
            annotation_file: Path to annotation file (COCO format)
            annotations: Pre-loaded annotations dict
            transform: Optional transform to be applied on a sample
            config: Configuration dictionary containing anchor parameters
        """
        self.image_dir = image_dir
        self.transform = transform
        self.config = config or {}
        
        # Load annotations
        if annotations is not None:
            self.annotations = annotations
        elif annotation_file is not None:
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            raise ValueError("Either annotation_file or annotations must be provided")
        
        # Create mappings
        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.annotations['images']}
        self.image_ids = list(self.image_id_to_filename.keys())
        
        # Create category mapping
        self.cat_mapping = {cat['id']: cat['id'] for cat in self.annotations['categories']}
        
        # Group annotations by image
        self.image_to_anns = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)
        
        # Get anchor parameters from config
        self.anchor_sizes = self.config.get('model', {}).get('anchor_sizes', [16, 32, 64, 128, 256])
        self.anchor_ratios = self.config.get('model', {}).get('anchor_ratios', [0.7, 1.0, 1.3])
        self.num_classes = self.config.get('model', {}).get('num_classes', 2)
        
        self.image_info = {img['id']: img for img in self.annotations['images']}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Get a single sample from the dataset."""
        image_id = self.image_ids[idx]
        filename = self.image_id_to_filename[image_id]
        
        # Get the image info
        image_info = next(img for img in self.annotations['images'] if img['id'] == image_id)
        image_path = os.path.join(self.image_dir, filename)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get image dimensions for normalization
        height, width = image.shape[:2]
        
        # Get annotations for this image
        anns = self.image_to_anns[image_id]
        
        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in anns:
            bbox = ann['bbox']  # [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            
            # Clip coordinates to image boundaries
            x1 = np.clip(x1, 0, width)
            y1 = np.clip(y1, 0, height)
            x2 = np.clip(x2, 0, width)
            y2 = np.clip(y2, 0, height)
            
            # Only add box if it's valid
            if x1 < x2 and y1 < y2 and (x2 - x1) > 1 and (y2 - y1) > 1:
                boxes.append([x1, y1, x2, y2])
                # Map category ID to ensure it's 1 (tick)
                category_id = ann['category_id']
                labels.append(self.cat_mapping.get(category_id, 1))  # Default to 1 (tick) if not found
            
        # If no valid boxes, create empty arrays
        if not boxes:
            boxes = []
            labels = []
            
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        else:
            # Basic normalization
            image = image.astype(np.float32) / 255.0
            boxes = boxes.astype(np.float32)
            labels = labels.astype(np.int64)
        
        # Generate anchor-based labels for RetinaNet
        anchor_labels, anchor_boxes = generate_anchor_labels(
            boxes, labels, (height, width), self.anchor_sizes, self.anchor_ratios, self.num_classes
        )
        
        # Convert to TensorFlow format
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Ensure image is in channels_last format
        if len(image_tensor.shape) == 3 and image_tensor.shape[0] == 3:  # channels_first
            image_tensor = tf.transpose(image_tensor, perm=[1, 2, 0])
        
        # Create target dictionary with anchor-based labels
        target = {
            'boxes': tf.convert_to_tensor(anchor_boxes, dtype=tf.float32),
            'labels': tf.convert_to_tensor(anchor_labels, dtype=tf.int32)
        }
        
        return image_tensor, target

class MultiChunkDataset:
    """Dataset that combines multiple chunks of data."""
    
    def __init__(self, image_dirs: List[str], annotation_files: List[str], 
                 transform: Optional[A.Compose] = None, config: Optional[Dict] = None):
        """
        Args:
            image_dirs: List of image directories
            annotation_files: List of annotation files
            transform: Optional transform to be applied on a sample
            config: Configuration dictionary containing anchor parameters
        """
        self.datasets = []
        self.cumulative_lengths = [0]
        
        for image_dir, annotation_file in zip(image_dirs, annotation_files):
            dataset = DetectionDataset(image_dir, annotation_file, transform=transform, config=config)
            self.datasets.append(dataset)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
    
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    
    def __getitem__(self, idx: int) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Get a single sample from the combined dataset."""
        # Find which dataset this index belongs to
        for i, cumulative_length in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cumulative_length:
                dataset_idx = i - 1
                local_idx = idx - self.cumulative_lengths[i - 1]
                return self.datasets[dataset_idx][local_idx]
        
        raise IndexError(f"Index {idx} out of range")

def create_tensorflow_dataset(dataset, batch_size: int, shuffle: bool = True, 
                            prefetch: bool = True) -> tf.data.Dataset:
    """Create a TensorFlow dataset from the custom dataset."""
    
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]
    
    # Create dataset from generator with anchor-based format
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            {
                'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32)
            }
        )
    )
    
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)
    
    # Pad sequences to handle variable number of anchors
    tf_dataset = tf_dataset.padded_batch(
        batch_size,
        padding_values=(
            tf.constant(0.0, dtype=tf.float32),
            {
                'boxes': tf.constant(0.0, dtype=tf.float32),
                'labels': tf.constant(0, dtype=tf.int32)
            }
        )
    )
    
    if prefetch:
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset

def create_train_val_datasets(config: Dict[str, Any], quick_test: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create training and validation datasets from configuration."""
    
    train_transform = get_transform(config, train=True)
    val_transform = get_transform(config, train=False)
    
    train_dataset = MultiChunkDataset(
        image_dirs=config['data']['train_paths'],
        annotation_files=config['data']['train_annotations'],
        transform=train_transform,
        config=config
    )
    
    val_dataset = MultiChunkDataset(
        image_dirs=config['data']['train_paths'],
        annotation_files=config['data']['train_annotations'],
        transform=val_transform,
        config=config
    )
    
    # For quick test, use only a small subset of data
    if quick_test:
        print("Quick test mode: Using small subset of data")
        # Create a subset of indices - doubled for more meaningful learning curves
        train_indices = list(range(min(100, len(train_dataset))))  # Increased from 50 to 100
        val_indices = list(range(min(40, len(val_dataset))))       # Increased from 20 to 40
        
        # Create subset datasets
        class SubsetDataset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        train_dataset = SubsetDataset(train_dataset, train_indices)
        val_dataset = SubsetDataset(val_dataset, val_indices)
    
    train_tf_dataset = create_tensorflow_dataset(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        prefetch=True
    )
    
    val_tf_dataset = create_tensorflow_dataset(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        prefetch=True
    )
    
    return train_tf_dataset, val_tf_dataset

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

def generate_anchor_labels(boxes: np.ndarray, labels: np.ndarray, image_shape: Tuple[int, int], 
                          anchor_sizes: List[int], anchor_ratios: List[float], 
                          num_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Generate anchor-based labels for RetinaNet training.
    
    Args:
        boxes: Ground truth boxes in [x1, y1, x2, y2] format
        labels: Ground truth labels
        image_shape: (height, width) of the image
        anchor_sizes: List of anchor sizes
        anchor_ratios: List of anchor ratios
        num_classes: Number of classes (background + object classes)
        
    Returns:
        Tuple of (anchor_labels, anchor_boxes)
    """
    height, width = image_shape
    num_anchors = len(anchor_sizes) * len(anchor_ratios)
    
    # Generate anchor boxes for the entire image
    # This is a simplified version - in practice, you'd generate anchors at each FPN level
    anchor_boxes = []
    anchor_labels = []
    
    # For simplicity, create a grid of anchors
    stride = 16  # Typical stride for anchor generation
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            for size in anchor_sizes:
                for ratio in anchor_ratios:
                    # Calculate anchor box dimensions
                    w = size * np.sqrt(ratio)
                    h = size / np.sqrt(ratio)
                    
                    # Anchor center
                    cx = x + stride // 2
                    cy = y + stride // 2
                    
                    # Convert to [x1, y1, x2, y2] format
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    
                    anchor_boxes.append([x1, y1, x2, y2])
                    
                    # Assign label based on IoU with ground truth boxes
                    best_iou = 0
                    best_label = 0  # background
                    
                    for gt_box, gt_label in zip(boxes, labels):
                        iou = calculate_iou([x1, y1, x2, y2], gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_label = gt_label
                    
                    # IoU thresholds for positive/negative anchors
                    if best_iou >= 0.5:
                        anchor_labels.append(best_label)
                    elif best_iou < 0.4:
                        anchor_labels.append(0)  # background
                    else:
                        # Ignore anchors with IoU between 0.4 and 0.5
                        anchor_labels.append(-1)
    
    return np.array(anchor_labels, dtype=np.int32), np.array(anchor_boxes, dtype=np.float32)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0 