import json
import cv2
import os
from pathlib import Path
import random
import argparse

def has_gui_support():
    """Check if OpenCV has GUI support."""
    try:
        cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('Test')
        return True
    except:
        return False

def draw_boxes(image_path, annotations, categories, output_path=None, show=True):
    """Draw bounding boxes on an image and optionally save/display it."""
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Draw each bounding box
    for ann in annotations:
        # COCO format uses [x, y, width, height], convert to [x1, y1, x2, y2]
        x, y, w, h = [int(coord) for coord in ann['bbox']]
        x2, y2 = x + w, y + h
        
        # Draw rectangle in BGR color (green)
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        
        # Get category name
        category_id = ann['category_id']
        category_name = next((cat['name'] for cat in categories if cat['id'] == category_id), 'Unknown')
        
        # Add label with confidence if available
        label = category_name
        if 'score' in ann:
            label += f" {ann['score']:.2f}"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save the image if output path is provided or if we can't show it
    if output_path or not has_gui_support():
        if not output_path:
            # If no output path provided but we can't show the image, create one
            output_dir = Path('outputs/visualizations')
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"annotated_{Path(image_path).name}"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), image)
        print(f"Saved annotated image to: {output_path}")
    
    # Display the image if show is True and we have GUI support
    if show and has_gui_support():
        cv2.imshow('Annotations', image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key

def main():
    parser = argparse.ArgumentParser(description='Visualize bounding box annotations')
    parser.add_argument('--annotations', default='data/annotations.json', help='Path to COCO format annotations JSON file')
    parser.add_argument('--images-dir', default='data/images', help='Directory containing images')
    parser.add_argument('--output-dir', default='outputs/visualizations', help='Directory to save visualizations')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of random images to visualize')
    parser.add_argument('--save', action='store_true', help='Save visualizations to output directory')
    parser.add_argument('--no-display', action='store_true', help='Do not display images')
    args = parser.parse_args()

    # Check GUI support
    gui_supported = has_gui_support()
    if not gui_supported:
        print("OpenCV GUI support not available. Images will be saved to disk instead.")
        args.save = True  # Force save if we can't display

    # Load COCO format annotations
    with open(args.annotations, 'r') as f:
        coco_data = json.load(f)
    
    # Create image id to filename mapping
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Create image id to annotations mapping
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_id_to_anns:
            image_id_to_anns[image_id] = []
        image_id_to_anns[image_id].append(ann)
    
    # Get list of image IDs
    image_ids = list(image_id_to_anns.keys())
    
    # Select random samples if num_samples is less than total images
    if args.num_samples < len(image_ids):
        image_ids = random.sample(image_ids, args.num_samples)
    
    # Process each image
    for image_id in image_ids:
        image_file = image_id_to_file[image_id]
        image_path = Path(args.images_dir) / image_file
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        
        # Get annotations for this image
        image_annotations = image_id_to_anns[image_id]
        
        # Prepare output path if saving
        output_path = None
        if args.save:
            output_path = Path(args.output_dir) / f"annotated_{image_file}"
        
        # Draw boxes and display/save
        key = draw_boxes(
            image_path, 
            image_annotations,
            coco_data['categories'],
            output_path=output_path,
            show=not args.no_display
        )
        
        # Break if 'q' is pressed and we have GUI support
        if gui_supported and not args.no_display and key is not None and (key & 0xFF == ord('q')):
            break

if __name__ == '__main__':
    main() 