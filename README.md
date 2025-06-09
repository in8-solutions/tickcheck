# Tick-detection via fine-tuned RetinaNet Model

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your images in the `data/images` directory
   - Create annotations in COCO format using the provided annotation tool
   - Save the annotation file as `data/annotations.json`

3. Configure training:
   - Modify `config.yaml` to set your training parameters
   - Update the class names in `config.yaml`

## Project Structure

```
.
├── data/
│   ├── images/          # Your training images
│   └── annotations.json # COCO format annotations
├── src/
│   ├── model.py        # RetinaNet model definition
│   ├── dataset.py      # Dataset handling
│   ├── train.py        # Training script
│   └── utils.py        # Utility functions
├── config.yaml         # Configuration file
└── requirements.txt    # Python dependencies
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
