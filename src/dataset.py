"""
Dataset handling for tick detection model.

This module implements a PyTorch Dataset for object detection using the COCO format.
It handles loading images and their corresponding annotations, applying transformations,
and preparing the data for training and validation.

Key Features:
- Supports COCO format annotations
- Handles bounding box normalization
- Provides configurable data augmentation
- Supports both training and inference transformations
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DetectionDataset(Dataset):
    """A PyTorch Dataset for object detection tasks using COCO format annotations.
    
    This dataset class handles:
    1. Loading images and annotations from a COCO format dataset
    2. Converting bounding box formats
    3. Applying transformations and augmentations
    4. Preparing data in the format expected by the model
    
    Args:
        image_dir (str): Directory containing the images
        annotation_file (str): Path to COCO format annotation JSON file
        transforms (albumentations.Compose, optional): Transformations to apply
        train (bool): Whether this is a training dataset (affects augmentations)
    """
    def __init__(self, image_dir, annotation_file, transforms=None, train=True):
        self.image_dir = image_dir
        self.transforms = transforms
        self.train = train
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Create image_id to annotations mapping
        self.image_to_annotations = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)
            
        # Create image_id to file mapping for all images, including those without annotations
        self.image_info = {img['id']: img for img in self.annotations['images']}
        
        # Get list of all image ids, including those without annotations
        self.image_ids = [img['id'] for img in self.annotations['images']]
        
        # Create category mapping
        self.cat_mapping = {cat['id']: idx for idx, cat in enumerate(self.annotations['categories'])}
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        # Get image id and info
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations if they exist
        boxes = []
        labels = []
        
        if img_id in self.image_to_annotations:
            anns = self.image_to_annotations[img_id]
            height, width = image.shape[:2]
            
            for ann in anns:
                # Get bbox in [x_min, y_min, width, height] format
                bbox = ann['bbox']
                
                # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
                x1 = float(bbox[0])
                y1 = float(bbox[1])
                x2 = float(bbox[0] + bbox[2])
                y2 = float(bbox[1] + bbox[3])
                
                # Ensure coordinates are valid
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # For Albumentations, we need normalized coordinates
                if self.transforms:
                    # Clip coordinates to image boundaries first
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    # Then normalize to [0,1] range
                    x1_norm = x1 / width
                    y1_norm = y1 / height
                    x2_norm = x2 / width
                    y2_norm = y2 / height
                    
                    # Double-check normalization (shouldn't be needed but just to be safe)
                    x1_norm = max(0.0, min(1.0, x1_norm))
                    y1_norm = max(0.0, min(1.0, y1_norm))
                    x2_norm = max(0.0, min(1.0, x2_norm))
                    y2_norm = max(0.0, min(1.0, y2_norm))
                    
                    # Skip invalid boxes after normalization
                    if x2_norm <= x1_norm or y2_norm <= y1_norm:
                        continue
                        
                    boxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                else:
                    boxes.append([x1, y1, x2, y2])
                    
                labels.append(self.cat_mapping[ann['category_id']])
        
        # Convert to numpy arrays with explicit float32 dtype for boxes and int64 for labels
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        
        # Apply transforms if specified
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
            
            # Denormalize boxes back to absolute coordinates
            height, width = image.shape[1:3]  # Image is now in CxHxW format
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
        else:
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        return image, target

def get_transform(config, train=True, inference_only=False):
    """Create transformation pipeline for images and bounding boxes.
    
    This function creates an Albumentations transformation pipeline that can:
    1. Resize images to a consistent size
    2. Apply data augmentation during training
    3. Normalize pixel values
    4. Convert images to PyTorch tensors
    
    The augmentations are controlled by the config file and include:
    - Horizontal and vertical flips
    - Rotation
    - Scale changes
    - Brightness/contrast adjustments
    - Gaussian blur
    - Noise addition
    
    Args:
        config (dict): Configuration dictionary containing transform settings
        train (bool): Whether to include training augmentations
        inference_only (bool): If True, create simpler transform for inference
        
    Returns:
        albumentations.Compose: Transformation pipeline
    """
    if inference_only:
        # Simple transform for inference without bbox handling
        transform = A.Compose([
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ])
    elif train:
        aug_config = config['augmentation']['train']
        transform = A.Compose([
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ),
            A.HorizontalFlip(p=0.5 if aug_config['horizontal_flip'] else 0),
            A.VerticalFlip(p=0.5 if aug_config['vertical_flip'] else 0),
            A.Rotate(
                limit=aug_config['rotate']['limit'] if aug_config['rotate']['enabled'] else 0,
                p=0.5 if aug_config['rotate']['enabled'] else 0
            ),
            A.RandomScale(
                scale_limit=(aug_config['scale']['min'] - 1.0, aug_config['scale']['max'] - 1.0),
                p=0.5 if aug_config['scale']['enabled'] else 0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config['brightness']['limit'],
                contrast_limit=aug_config['contrast']['limit'],
                p=0.5 if aug_config['brightness']['enabled'] or aug_config['contrast']['enabled'] else 0
            ),
            A.GaussianBlur(
                blur_limit=aug_config['blur']['limit'] if aug_config['blur']['enabled'] else 0,
                p=0.5 if aug_config['blur']['enabled'] else 0
            ),
            A.GaussNoise(
                p=0.5 if aug_config['noise']['enabled'] else 0
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
    else:
        # Validation transform - just resize and normalize
        transform = A.Compose([
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
    
    return transform 