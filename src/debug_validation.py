import torch
from utils import load_config, create_directories, collate_fn
from dataset import DetectionDataset, get_transform
from model import create_model
from torch.utils.data import DataLoader, random_split, Subset
from torch.amp import autocast
from tqdm import tqdm

def debug_validate(model, data_loader, device):
    """Debug version of validation function"""
    model.eval()
    
    with torch.no_grad(), autocast('cuda'):
        # Only process first batch
        images, targets = next(iter(data_loader))
        
        # Move to device
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        
        # Get model output and print detailed info
        print("\nModel input:")
        print(f"Number of images: {len(images)}")
        print(f"Image shapes: {[img.shape for img in images]}")
        print(f"Target structures: {[{k: v.shape if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]}")
        
        print("\nRunning model.eval() output:")
        loss_dict = model(images, targets)
        
        print("\nModel output structure:")
        print(f"Type of output: {type(loss_dict)}")
        if isinstance(loss_dict, dict):
            for k, v in loss_dict.items():
                print(f"Key: {k}")
                print(f"Value type: {type(v)}")
                if isinstance(v, torch.Tensor):
                    print(f"Value shape: {v.shape}")
                    print(f"Value: {v}")
        elif isinstance(loss_dict, (list, tuple)):
            for i, item in enumerate(loss_dict):
                print(f"\nItem {i}:")
                print(f"Type: {type(item)}")
                if isinstance(item, dict):
                    for k, v in item.items():
                        print(f"  Key: {k}")
                        print(f"  Value type: {type(v)}")
                        if isinstance(v, torch.Tensor):
                            print(f"  Value shape: {v.shape}")
                            print(f"  Value: {v}")
                elif isinstance(item, torch.Tensor):
                    print(f"Shape: {item.shape}")
                    print(f"Value: {item}")

def main():
    # Load configuration
    config = load_config('config.yaml')
    
    # Set smaller batch size for debugging
    config['training']['batch_size'] = 2
    
    # Create model
    model = create_model(config)
    device = next(model.parameters()).device
    
    # Create a small dataset
    dataset = DetectionDataset(
        config['data']['train_path'],
        config['data']['train_annotations'],
        transforms=get_transform(config, train=False)
    )
    
    # Use only 4 images for quick testing
    dataset = Subset(dataset, range(4))
    
    # Create data loader
    loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'num_workers': 0,  # No multiprocessing for debugging
        'collate_fn': collate_fn,
        'pin_memory': False,  # Disable for debugging
        'shuffle': False  # Keep order consistent
    }
    
    data_loader = DataLoader(dataset, **loader_kwargs)
    
    # Run validation debug
    print("Starting validation debug...")
    debug_validate(model, data_loader, device)

if __name__ == '__main__':
    main() 