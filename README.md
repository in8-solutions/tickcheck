# TickCheck - Binary Tick Detection Classifier

A binary classification system for detecting ticks in images using a fine-tuned RetinaNet model. This project implements a "tick vs no tick" classifier that can identify the presence of ticks in images.

## Project Overview

This is a **binary classifier** that distinguishes between:
- **Tick present**: Images containing one or more ticks
- **No tick**: Images without any ticks

The system uses a RetinaNet model with ResNet50 backbone for object detection, trained to identify tick instances in images.

## Setup

1. Install system dependencies:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-tk  # Required for GUI tools
```

2. Create and activate a virtual environment:
```bash
# First time setup only - create the virtual environment
python -m venv venv

# Every time you work on the project - activate the virtual environment
source venv/bin/activate
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

## Primary Tools

### Core Training and Evaluation

- **`src/train.py`** - Main training script for the tick detection model
  - Handles data loading, training loops, validation, and checkpointing
  - Supports mixed precision training and GPU optimization
  - Generates training curves and saves model checkpoints

- **`src/evaluate_model.py`** - Model evaluation and testing script
  - Tests the final trained model on test images
  - Provides comprehensive metrics for binary classification (tick vs no tick)
  - Generates visualizations of predictions with confidence scores
  - Creates detailed evaluation reports

- **`src/verify_annotations.py`** - Annotation review and cleanup tool
  - GUI-based tool for reviewing and cleaning up training data
  - Allows visual inspection of annotations across different data chunks
  - Supports adding, deleting, and modifying bounding box annotations
  - Navigate between images and chunks with keyboard shortcuts

### Utility Tools

- **`src/verify_system.py`** - System verification and setup checker
  - Ensures development system is correctly configured
  - Checks Python version, GPU/CUDA availability, required packages
  - Tests GUI functionality for annotation tools
  - Provides detailed setup verification report

- **`src/save_artifacts.py`** - Model and training artifact archiver
  - Archives models and associated configuration, training data, etc.
  - Bundles best model checkpoint, training curves, config files
  - Creates metadata and summary reports for training runs
  - Useful for versioning and sharing successful training runs

- **`src/manage_chunks.py`** - Dataset chunk management utility
  - Splits large annotation files into manageable chunks
  - Lists chunk statistics and information
  - Creates chunk-specific configuration files
  - Cleans up chunk directories when needed

## Project Structure

```
.
├── data/
│   ├── chunk_001/           # Data chunks for training
│   │   ├── images/          # Training images
│   │   └── annotations.json # COCO format annotations
│   ├── chunk_002/
│   │   ├── images/
│   │   └── annotations.json
│   └── ...                  # Additional chunks
├── src/
│   ├── model.py                # RetinaNet model definition
│   ├── dataset.py              # Dataset handling and preprocessing
│   ├── train.py                # Training script
│   ├── evaluate_model.py       # Model evaluation and metrics
│   ├── verify_annotations.py   # Annotation review and cleanup
│   ├── verify_system.py        # System verification script
│   ├── save_artifacts.py       # Training artifact archiver
│   ├── manage_chunks.py        # Dataset chunk management
│   ├── utils.py                # Utility functions
│   ├── inference.py            # Model inference and prediction
│   └── visualize_annotations.py # Annotation visualization tools
├── config.yaml                 # Configuration file
└── requirements.txt            # Python dependencies
```

## Usage

### Prerequisites

#### 1. System Verification

First, ensure your system is properly configured:

```bash
python src/verify_system.py
```

This checks:
- Python version (3.8 or higher required)
- GPU and CUDA availability
- Required Python packages
- System packages (python3-tk)
- GUI functionality

#### 2. Review and Clean Annotations

Before training, review and clean your training data:

```bash
python src/verify_annotations.py
```

This opens a GUI tool for:
- Navigating through images and chunks
- Reviewing bounding box annotations
- Adding or deleting annotations
- Saving changes to annotation files

### Training the Model

1. data content is being stored elsewhere (images + annotations); they are needed

2. Train the model:
```bash
python src/train.py
```

3. Monitor training:
   - Training metrics are logged to the console
   - Visualizations are saved in the `outputs` directory

### Evaluating the Model

```bash
python src/evaluate_model.py --model outputs/training/checkpoints/latest_model.pth --input test_images --output outputs/evaluation_test --threshold 0.5
```

### Archiving Training Results

```bash
python src/save_artifacts.py --name "training_run_v1" --output "archives/"
```

## Model Details

This implementation uses RetinaNet with:
- **Backbone**: ResNet50 FPN
- **Loss Function**: Focal Loss for classification, Smooth L1 Loss for box regression
- **Data Augmentation**: Albumentations library
- **Binary Classification**: Tick vs No Tick detection

## Image Storage and Setup

The repository contains annotation files and empty image directories. The actual image files are stored externally on Google Drive.

### Setting up images locally:

1. Download the images from Google Drive: [https://drive.google.com/drive/folders/1rfOLcwgnosodbiRYgvbG8Y67W8C8idF4?usp=drive_link](https://drive.google.com/drive/folders/1rfOLcwgnosodbiRYgvbG8Y67W8C8idF4?usp=drive_link)
2. The images are organized in chunks (chunk_001 through chunk_015)
3. Copy the images from the downloaded folder into the corresponding `data/chunk_XXX/images/` directories
   - For example, copy images from `downloaded/chunk_001/images/*` to `data/chunk_001/images/`
   - You only need to copy the chunks you're working with

Note: Image files (*.jpg, *.jpeg, *.png) are git-ignored, so you can safely copy them into the data directory without affecting git.
