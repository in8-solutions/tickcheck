import json
import os
from pathlib import Path
from collections import defaultdict

def validate_annotations():
    # Load annotations
    try:
        with open('data/annotations.json', 'r') as f:
            annotations = json.load(f)
    except json.JSONDecodeError:
        print("ERROR: annotations.json is not valid JSON!")
        return
    except FileNotFoundError:
        print("ERROR: annotations.json not found!")
        return

    # Get list of image files
    image_files = set()
    try:
        for file in os.listdir('data/images'):
            if file.endswith('.jpg'):
                image_files.add(file)
    except FileNotFoundError:
        print("ERROR: images directory not found!")
        return

    # Get image filenames from annotations
    annotated_images = set()
    try:
        for image in annotations['images']:
            annotated_images.add(image['file_name'])
    except KeyError:
        print("ERROR: Annotations file missing 'images' key or has wrong structure!")
        return

    # Check for mismatches
    missing_annotations = image_files - annotated_images
    missing_images = annotated_images - image_files

    # Analyze annotation distribution
    image_id_to_file = {img['id']: img['file_name'] for img in annotations['images']}
    annotations_by_image = defaultdict(list)
    orphaned_annotations = []
    
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id in image_id_to_file:
            img_file = image_id_to_file[img_id]
            annotations_by_image[img_file].append(ann)
        else:
            orphaned_annotations.append(ann)

    # Calculate statistics
    total_annotations = sum(len(anns) for anns in annotations_by_image.values())
    images_with_annotations = len(annotations_by_image)
    avg_annotations = total_annotations / images_with_annotations if images_with_annotations > 0 else 0
    
    # Distribution of annotations per image
    annotation_counts = defaultdict(int)
    for img, anns in annotations_by_image.items():
        annotation_counts[len(anns)] += 1

    print("\nAnnotation Validation Summary")
    print("=" * 50)
    
    print("\n1. Dataset Overview")
    print("-" * 20)
    print(f"• Physical images in directory:     {len(image_files):,}")
    print(f"• Images referenced in annotations: {len(annotated_images):,}")
    print(f"• Images with valid annotations:    {images_with_annotations:,}")
    print(f"• Total annotation entries:         {total_annotations:,}")
    print(f"• Average annotations per image:    {avg_annotations:.2f}")
    
    print("\n2. Annotation Distribution")
    print("-" * 20)
    for count in sorted(annotation_counts.keys()):
        images = annotation_counts[count]
        percentage = (images / images_with_annotations) * 100
        print(f"• {images:,} images ({percentage:.1f}%) have {count} annotation{'s' if count != 1 else ''}")
    
    print("\n3. Data Integrity Issues")
    print("-" * 20)
    
    if missing_annotations:
        print(f"\n⚠️  Found {len(missing_annotations)} images without annotations:")
        for img in sorted(missing_annotations):
            print(f"   • {img}")
    
    if missing_images:
        print(f"\n⚠️  Found {len(missing_images)} annotations referencing missing images:")
        for img in sorted(missing_images):
            print(f"   • {img}")
            
    if orphaned_annotations:
        print(f"\n⚠️  Found {len(orphaned_annotations)} annotations with invalid image IDs")
    
    if not any([missing_annotations, missing_images, orphaned_annotations]):
        print("✓ No integrity issues found!")

    print("\n4. Recommended Actions")
    print("-" * 20)
    if missing_annotations:
        print("• Review and annotate images without annotations")
    if missing_images:
        print("• Remove annotations referencing non-existent images")
    if orphaned_annotations:
        print("• Clean up annotations with invalid image IDs")
    if not any([missing_annotations, missing_images, orphaned_annotations]):
        print("✓ No actions needed - dataset is clean!")

if __name__ == '__main__':
    validate_annotations() 