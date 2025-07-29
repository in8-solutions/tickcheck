"""
Data Pipeline for Mobile Tick Detection
Converts object detection annotations to binary classification samples
"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import cv2
from typing import List, Tuple, Dict, Optional
import logging

from config import DATA_CONFIG, AUGMENTATION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TickBinaryDataset(Dataset):
    """
    Binary classification dataset for tick detection
    Converts object detection annotations to classification samples via cropping
    """
    
    def __init__(self, 
                 image_paths: List[Path],
                 annotation_files: List[Path],
                 transform=None,
                 mode='train',
                 crop_padding: float = 0.3,
                 negative_crops_per_image: int = 3):
        
        self.image_paths = image_paths
        self.annotation_files = annotation_files
        self.transform = transform
        self.mode = mode
        self.crop_padding = crop_padding
        self.negative_crops_per_image = negative_crops_per_image
        
        # Load all annotations
        self.samples = self._load_samples()
        logger.info(f"Created {len(self.samples)} samples for {mode} mode")
        
    def _load_samples(self) -> List[Dict]:
        """Load and process annotations to create binary classification samples"""
        samples = []
        

        
        for img_path, ann_path in zip(self.image_paths, self.annotation_files):
            if not img_path.exists() or not ann_path.exists():
                continue
                
            # Load image to get dimensions
            image = Image.open(img_path)
            img_width, img_height = image.size
            
            # Load annotations
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
            
            # Find tick annotations for this image
            image_annotations = []
            
            # Find the image ID for this image
            image_id = None
            for img_info in annotations.get('images', []):
                if img_info['file_name'] == img_path.name:
                    image_id = img_info['id']
                    break
            
            if image_id is not None:
                for ann in annotations.get('annotations', []):
                    if ann['category_id'] == 1 and ann['image_id'] == image_id:  # tick category for this image
                        image_annotations.append(ann)
            
            # Create positive samples from bounding boxes
            for ann in image_annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = bbox
                
                # Add padding around bounding box
                pad_x = int(w * self.crop_padding)
                pad_y = int(h * self.crop_padding)
                
                # Calculate crop coordinates with padding
                crop_x1 = max(0, x - pad_x)
                crop_y1 = max(0, y - pad_y)
                crop_x2 = min(img_width, x + w + pad_x)
                crop_y2 = min(img_height, y + h + pad_y)
                
                # Ensure valid coordinates
                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    continue
                
                # Ensure minimum crop size
                min_size = 50
                if (crop_x2 - crop_x1) < min_size or (crop_y2 - crop_y1) < min_size:
                    # Expand crop to minimum size
                    center_x = (crop_x1 + crop_x2) // 2
                    center_y = (crop_y1 + crop_y2) // 2
                    half_size = min_size // 2
                    crop_x1 = max(0, center_x - half_size)
                    crop_y1 = max(0, center_y - half_size)
                    crop_x2 = min(img_width, center_x + half_size)
                    crop_y2 = min(img_height, center_y + half_size)
                
                samples.append({
                    'image_path': img_path,
                    'crop_coords': (crop_x1, crop_y1, crop_x2, crop_y2),
                    'label': 1,  # tick present
                    'original_bbox': bbox
                })
            
            # Create negative samples from images without ticks or random crops
            if len(image_annotations) == 0:
                # Image has no ticks - create multiple negative samples
                for _ in range(self.negative_crops_per_image):
                    # Generate random crop
                    crop_size = min(img_width, img_height) // 2
                    if crop_size < 50:
                        crop_size = min(img_width, img_height)
                    
                    x1 = random.randint(0, max(0, img_width - crop_size))
                    y1 = random.randint(0, max(0, img_height - crop_size))
                    x2 = x1 + crop_size
                    y2 = y1 + crop_size
                    
                    # Ensure valid coordinates
                    if x2 > img_width or y2 > img_height:
                        continue
                    
                    samples.append({
                        'image_path': img_path,
                        'crop_coords': (x1, y1, x2, y2),
                        'label': 0,  # no tick
                        'original_bbox': None
                    })
            else:
                # Image has ticks - create negative samples using edge-based approach
                # For single-tick images, create negative regions extending away from the tick
                
                negative_samples_created = 0
                target_negative_samples = self.negative_crops_per_image
                target_size = 250  # Target size for negative regions
                
                # Only process single-tick images for this approach
                if len(image_annotations) == 1:
                    bbox = image_annotations[0]['bbox']
                    tx, ty, tw, th = bbox
                    
                    # Try to create negative regions for each edge of the bounding box
                    # Each region extends away from the tick (flipped approach)
                    
                    # 1. Left edge - extend left (horizontally flipped)
                    if tx >= target_size:
                        x1 = tx - target_size
                        y1 = max(0, ty - target_size // 2)
                        x2 = tx
                        y2 = min(img_height, ty + th + target_size // 2)
                        
                        if y2 - y1 >= target_size // 2:  # Ensure minimum height
                            samples.append({
                                'image_path': img_path,
                                'crop_coords': (x1, y1, x2, y2),
                                'label': 0,  # no tick
                                'original_bbox': None,
                                'flip_horizontal': True  # Flip to avoid edge bias
                            })
                            negative_samples_created += 1
                    
                    # 2. Right edge - extend right (horizontally flipped)
                    if tx + tw + target_size <= img_width:
                        x1 = tx + tw
                        y1 = max(0, ty - target_size // 2)
                        x2 = tx + tw + target_size
                        y2 = min(img_height, ty + th + target_size // 2)
                        
                        if y2 - y1 >= target_size // 2:  # Ensure minimum height
                            samples.append({
                                'image_path': img_path,
                                'crop_coords': (x1, y1, x2, y2),
                                'label': 0,  # no tick
                                'original_bbox': None,
                                'flip_horizontal': True  # Flip to avoid edge bias
                            })
                            negative_samples_created += 1
                    
                    # 3. Top edge - extend up (vertically flipped)
                    if ty >= target_size:
                        x1 = max(0, tx - target_size // 2)
                        y1 = ty - target_size
                        x2 = min(img_width, tx + tw + target_size // 2)
                        y2 = ty
                        
                        if x2 - x1 >= target_size // 2:  # Ensure minimum width
                            samples.append({
                                'image_path': img_path,
                                'crop_coords': (x1, y1, x2, y2),
                                'label': 0,  # no tick
                                'original_bbox': None,
                                'flip_vertical': True  # Flip to avoid edge bias
                            })
                            negative_samples_created += 1
                    
                    # 4. Bottom edge - extend down (vertically flipped)
                    if ty + th + target_size <= img_height:
                        x1 = max(0, tx - target_size // 2)
                        y1 = ty + th
                        x2 = min(img_width, tx + tw + target_size // 2)
                        y2 = ty + th + target_size
                        
                        if x2 - x1 >= target_size // 2:  # Ensure minimum width
                            samples.append({
                                'image_path': img_path,
                                'crop_coords': (x1, y1, x2, y2),
                                'label': 0,  # no tick
                                'original_bbox': None,
                                'flip_vertical': True  # Flip to avoid edge bias
                            })
                            negative_samples_created += 1
                
                # Note: Some images may not be able to create the full target number of negative samples
                # This is expected for small images or images where ticks take up most of the space
        

        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and crop image
        image = Image.open(sample['image_path'])
        # Ensure image is in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        x1, y1, x2, y2 = sample['crop_coords']
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # Convert to numpy array for albumentations
        image_array = np.array(cropped_image)
        
        # Apply flipping to avoid edge bias for negative samples
        if sample.get('flip_horizontal', False):
            image_array = np.fliplr(image_array).copy()
        if sample.get('flip_vertical', False):
            image_array = np.flipud(image_array).copy()
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image_array)
            image_array = transformed['image']
        
        # Convert to tensor if not already done by transform
        if not isinstance(image_array, torch.Tensor):
            image_array = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image_array,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'image_path': str(sample['image_path']),
            'crop_coords': sample['crop_coords']
        }


def get_transforms(mode: str = 'train') -> A.Compose:
    """Get data augmentation transforms"""
    config = AUGMENTATION_CONFIG[mode]
    
    transforms = []
    
    if mode == 'train':
        # Training augmentations
        if config['horizontal_flip']:
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if config['vertical_flip']:
            transforms.append(A.VerticalFlip(p=0.5))
        
        if config['rotate']['enabled']:
            transforms.append(A.Rotate(
                limit=config['rotate']['limit'],
                p=0.5
            ))
        
        if config['scale']['enabled']:
            transforms.append(A.RandomScale(
                scale_limit=(config['scale']['min'] - 1, config['scale']['max'] - 1),
                p=0.5
            ))
        
        if config['brightness']['enabled']:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config['brightness']['limit'],
                contrast_limit=config['contrast']['limit'],
                p=0.5
            ))
        
        if config['blur']['enabled']:
            transforms.append(A.GaussianBlur(
                blur_limit=config['blur']['limit'],
                p=0.3
            ))
        
        if config['noise']['enabled']:
            transforms.append(A.GaussNoise(
                var_limit=(0, config['noise']['limit']),
                p=0.3
            ))
    
    # Resize to target size
    transforms.append(A.Resize(
        height=DATA_CONFIG['input_size'][0],
        width=DATA_CONFIG['input_size'][1]
    ))
    
    # Normalize
    transforms.append(A.Normalize(
        mean=DATA_CONFIG['mean'],
        std=DATA_CONFIG['std']
    ))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def create_data_loaders(data_dir: Path, 
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Find chunk directories up to max_chunk from config
    chunk_dirs = []
    max_chunk = DATA_CONFIG.get('max_chunk', 7)  # Default to 7 if not specified
    for chunk_num in range(1, max_chunk + 1):
        chunk_dir = data_dir / f"chunk_{chunk_num:03d}"
        if chunk_dir.exists():
            chunk_dirs.append(chunk_dir)
        else:
            logger.warning(f"Chunk {chunk_num} directory not found: {chunk_dir}")
    
    chunk_dirs.sort()
    logger.info(f"Using chunks: {[d.name for d in chunk_dirs]}")
    
    # Split chunks for train/val
    train_chunks = chunk_dirs[:int(len(chunk_dirs) * DATA_CONFIG['train_split'])]
    val_chunks = chunk_dirs[int(len(chunk_dirs) * DATA_CONFIG['train_split']):]
    
    # Collect image paths and annotation files
    train_image_paths = []
    train_annotation_paths = []
    val_image_paths = []
    val_annotation_paths = []
    
    for chunk_dir in train_chunks:
        images_dir = chunk_dir / "images"
        ann_file = chunk_dir / "annotations.json"
        if images_dir.exists() and ann_file.exists():
            # Filter out hidden files and ensure they're valid images
            image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')]
            train_image_paths.extend(image_files)
            train_annotation_paths.extend([ann_file] * len(image_files))
    
    for chunk_dir in val_chunks:
        images_dir = chunk_dir / "images"
        ann_file = chunk_dir / "annotations.json"
        if images_dir.exists() and ann_file.exists():
            # Filter out hidden files and ensure they're valid images
            image_files = [f for f in images_dir.glob("*.jpg") if not f.name.startswith('.')]
            val_image_paths.extend(image_files)
            val_annotation_paths.extend([ann_file] * len(image_files))
    
    logger.info(f"Found {len(train_image_paths)} training images, {len(val_image_paths)} validation images")
    
    # Create datasets
    train_dataset = TickBinaryDataset(
        image_paths=train_image_paths,
        annotation_files=train_annotation_paths,
        transform=get_transforms('train'),
        mode='train',
        crop_padding=DATA_CONFIG['crop_padding'],
        negative_crops_per_image=DATA_CONFIG['negative_crops_per_image']
    )
    
    val_dataset = TickBinaryDataset(
        image_paths=val_image_paths,
        annotation_files=val_annotation_paths,
        transform=get_transforms('val'),
        mode='val',
        crop_padding=DATA_CONFIG['crop_padding'],
        negative_crops_per_image=DATA_CONFIG['negative_crops_per_image']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the data pipeline
    from config import DATA_DIR
    
    train_loader, val_loader = create_data_loaders(DATA_DIR, batch_size=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch shape: {batch['image'].shape}")
        print(f"Labels: {batch['label']}")
        print(f"Label distribution: {torch.bincount(batch['label'])}")
        break 