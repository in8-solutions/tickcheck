# Tick-detection via fine-tuned RetinaNet Model

## Setup

1. Install system dependencies:
```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3-tk  # Required for GUI tools
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place your images in the `data/images` directory
   - Create annotations in COCO format using the provided annotation tool
   - Save the annotation file as `data/annotations.json`

4. Configure training:
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

4. Evalate model

## Model Details

This implementation uses RetinaNet with:
- Backbone: ResNet50 FPN
- Focal Loss for classification
- Smooth L1 Loss for box regression
- Data augmentation using albumentations 

# TickCheck

A tool for verifying tick annotations in images.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tickcheck.git
cd tickcheck
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Image Storage and Setup

The repository contains annotation files and empty image directories. The actual image files are stored externally on Google Drive.

### Setting up images locally:

1. Download the images from Google Drive (link: [TODO: Add Google Drive link])
2. The images are organized in chunks (chunk_001 through chunk_015)
3. Copy the images from the downloaded folder into the corresponding `data/chunk_XXX/images/` directories
   - For example, copy images from `downloaded/chunk_001/images/*` to `data/chunk_001/images/`
   - You only need to copy the chunks you're working with

Note: Image files (*.jpg, *.jpeg, *.png) are git-ignored, so you can safely copy them into the data directory without affecting git.

## Usage

[Rest of existing README content...]
