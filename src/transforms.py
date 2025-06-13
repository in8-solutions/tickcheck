import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transform(config, train=True, inference_only=False):
    """
    Get transforms for training or validation.
    
    Args:
        config: Configuration dictionary
        train: Whether to get training transforms (True) or validation transforms (False)
        inference_only: Whether to get transforms for inference only
        
    Returns:
        Albumentations Compose transform
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
        # Training transforms
        transform = A.Compose([
            A.Resize(
                height=config['data']['input_size'][0],
                width=config['data']['input_size'][1]
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.Normalize(
                mean=config['data']['mean'],
                std=config['data']['std']
            ),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        # Validation transforms
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
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    return transform 