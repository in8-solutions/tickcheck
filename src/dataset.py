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
        annotation_file (str, optional): Path to COCO format annotation JSON file
        annotations: Direct annotations dictionary (optional)
        transform (albumentations.Compose, optional): Transformations to apply
    """
    def __init__(self, image_dir, annotation_file=None, annotations=None, transform=None):
        """
        Args:
            image_dir: Directory with all the images
            annotation_file: Path to the annotation file (optional)
            annotations: Direct annotations (list or dict) (optional)
            transform: Optional transform to be applied on a sample
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Load annotations either from file or directly
        if annotation_file is not None:
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = annotations
            
        # Handle both list and dictionary annotation formats
        if isinstance(self.annotations, list):
            # If annotations is a list, assume it's a list of annotation dictionaries
            self.image_to_anns = {}
            for ann in self.annotations:
                image_id = ann['image_id']
                if image_id not in self.image_to_anns:
                    self.image_to_anns[image_id] = []
                self.image_to_anns[image_id].append(ann)
                
            # Create image id to filename mapping from annotations
            self.image_id_to_filename = {}
            for ann in self.annotations:
                image_id = ann['image_id']
                if image_id not in self.image_id_to_filename:
                    # Get filename from the image info in the annotation
                    if 'image' in ann:
                        self.image_id_to_filename[image_id] = ann['image']['file_name']
                    else:
                        # If no image info, construct filename from image_id
                        self.image_id_to_filename[image_id] = f"image_{image_id}.jpg"
        else:
            # If annotations is a dictionary, use the standard COCO format
            self.image_to_anns = {}
            for ann in self.annotations['annotations']:
                image_id = ann['image_id']
                if image_id not in self.image_to_anns:
                    self.image_to_anns[image_id] = []
                self.image_to_anns[image_id].append(ann)
                
            # Create image id to filename mapping
            self.image_id_to_filename = {
                img['id']: img['file_name'] for img in self.annotations['images']
            }
            
        # Get unique image ids
        self.image_ids = list(self.image_to_anns.keys())
        
        # Create category mapping if categories are provided
        if isinstance(self.annotations, dict) and 'categories' in self.annotations:
            # Map category IDs to ensure they match our class numbering (0=background, 1=tick)
            self.cat_mapping = {cat['id']: 1 for cat in self.annotations['categories']}  # All categories map to tick (1)
        else:
            # If no categories provided, assume all annotations are for class 1 (tick)
            self.cat_mapping = {1: 1}
        
        self.image_info = {img['id']: img for img in self.annotations['images']}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
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
            
        # If no valid boxes, create a dummy box with background label (0)
        if not boxes:
            boxes = [[0, 0, width, height]]
            labels = [0]  # Background class
            
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
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

class MultiChunkDataset(Dataset):
    """A PyTorch Dataset that combines multiple chunks of data.
    
    This dataset class:
    1. Loads images and annotations from multiple chunks
    2. Combines them into a single dataset
    3. Maintains proper image ID mapping across chunks
    4. Applies the same transformations as DetectionDataset
    
    Args:
        image_dirs (List[str]): List of directories containing the images
        annotation_files (List[str]): List of paths to COCO format annotation JSON files
        transform (albumentations.Compose, optional): Transformations to apply
    """
    def __init__(self, image_dirs, annotation_files, transform=None):
        self.image_dirs = image_dirs
        self.transform = transform
        
        # Load and combine annotations from all chunks
        self.annotations = {'images': [], 'annotations': [], 'categories': []}
        self.image_to_anns = {}
        self.image_id_to_filename = {}
        self.image_id_to_dir = {}  # Map image IDs to their source directory
        
        # Process each chunk
        for img_dir, ann_file in zip(image_dirs, annotation_files):
            with open(ann_file, 'r') as f:
                chunk_anns = json.load(f)
            
            # Add images and annotations to combined dataset
            if isinstance(chunk_anns, dict):
                # Handle COCO format
                self.annotations['images'].extend(chunk_anns['images'])
                self.annotations['annotations'].extend(chunk_anns['annotations'])
                if 'categories' in chunk_anns:
                    self.annotations['categories'].extend(chunk_anns['categories'])
                
                # Update mappings
                for img in chunk_anns['images']:
                    img_id = img['id']
                    self.image_id_to_filename[img_id] = img['file_name']
                    self.image_id_to_dir[img_id] = img_dir
                
                for ann in chunk_anns['annotations']:
                    img_id = ann['image_id']
                    if img_id not in self.image_to_anns:
                        self.image_to_anns[img_id] = []
                    self.image_to_anns[img_id].append(ann)
            else:
                # Handle list format
                for ann in chunk_anns:
                    img_id = ann['image_id']
                    if img_id not in self.image_to_anns:
                        self.image_to_anns[img_id] = []
                        if 'image' in ann:
                            self.image_id_to_filename[img_id] = ann['image']['file_name']
                            self.image_id_to_dir[img_id] = img_dir
                    self.image_to_anns[img_id].append(ann)
        
        # Get unique image ids
        self.image_ids = list(self.image_to_anns.keys())
        
        # Create category mapping
        if self.annotations['categories']:
            self.cat_mapping = {cat['id']: cat['id'] for cat in self.annotations['categories']}
        else:
            self.cat_mapping = {1: 1}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.image_id_to_filename[image_id]
        image_dir = self.image_id_to_dir[image_id]
        
        # Load image
        image_path = os.path.join(image_dir, filename)
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
                category_id = ann['category_id']
                labels.append(self.cat_mapping.get(category_id, 1))
        
        if not boxes:  # If no valid boxes, create a dummy box
            boxes = [[0, 0, width, height]]
            labels = [1]
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': torch.tensor([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes], dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        return image, target 