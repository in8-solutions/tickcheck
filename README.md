# Tick-detection via fine-tuned RetinaNet Model

## Setup

1. Install system dependencies:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-tk  # Required for GUI tools
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Verify your setup:
```bash
python src/verify_system.py
```
This will check:
- Python version (3.8 or higher required)
- GPU and CUDA availability
- Required Python packages (PyTorch, TorchVision, NumPy, Pillow, OpenCV, Matplotlib, Albumentations, tqdm, PyYAML, Pandas)
- System packages (python3-tk)
- GUI functionality

5. Prepare your dataset:
   - Place your images in the `data/images` directory
   - Create annotations in COCO format using the provided annotation tool
   - Save the annotation file as `data/annotations.json`

6. Configure training:
   - Modify `config.yaml` to set your training parameters
   - Update the class names in `config.yaml`

## Project Structure

```
.
├── data/
│   ├── images/                  # Your training images
│   └── annotations.json         # COCO format annotations
├── src/
│   ├── model.py                # RetinaNet model definition
│   ├── dataset.py              # Dataset handling and preprocessing
│   ├── train.py                # Training script
│   ├── utils.py                # Utility functions
│   ├── inference.py            # Model inference and prediction
│   ├── evaluate_model.py       # Model evaluation and metrics
│   ├── verify_system.py        # System verification script
│   └── visualize_annotations.py # Annotation visualization tools
├── config.yaml                 # Configuration file
└── requirements.txt            # Python dependencies
```

## Usage

1. data content is being stored elsewhere (images + annotations); they are needed

2. Train the model:
```bash
python src/train.py
```

3. Monitor training:
   - Training metrics are logged to the console
   - Visualizations are saved in the `outputs` directory

4. Evaluate model:
```bash
python src/evaluate_model.py --model outputs/training/checkpoints/latest_model.pth --input data/chunk_001/images --output outputs/evaluation_test --threshold 0.5
```

## Model Details

This implementation uses RetinaNet with:
- Backbone: ResNet50 FPN
- Focal Loss for classification
- Smooth L1 Loss for box regression
- Data augmentation using albumentations 

# TickCheck

A tool for verifying tick annotations in images.

## Image Storage and Setup

The repository contains annotation files and empty image directories. The actual image files are stored externally on Google Drive.

### Setting up images locally:

1. Download the images from Google Drive (link: [TODO: Add Google Drive link])
2. The images are organized in chunks (chunk_001 through chunk_015)
3. Copy the images from the downloaded folder into the corresponding `data/chunk_XXX/images/` directories
   - For example, copy images from `downloaded/chunk_001/images/*` to `data/chunk_001/images/`
   - You only need to copy the chunks you're working with

Note: Image files (*.jpg, *.jpeg, *.png) are git-ignored, so you can safely copy them into the data directory without affecting git.
