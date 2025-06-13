import os
import shutil
from pathlib import Path

def create_readme(images_dir):
    """Create a README file in the images directory."""
    readme_content = """# Images Directory

The image files for this chunk have been moved to external storage.

## Image Location
The images for this chunk are stored in external storage at:
`/path/to/external/storage/images/chunk_XXX/`

Replace `/path/to/external/storage/` with the actual path to your image storage.
"""
    readme_path = os.path.join(images_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)

def migrate_images():
    # Create new directory for images
    new_base = "external_images"
    os.makedirs(new_base, exist_ok=True)
    
    # Process each chunk
    data_dir = "data"
    for chunk in sorted(os.listdir(data_dir)):
        if not chunk.startswith("chunk_"):
            continue
            
        print(f"Processing {chunk}...")
        
        # Create new directory structure
        new_chunk_dir = os.path.join(new_base, chunk)
        new_images_dir = os.path.join(new_chunk_dir, "images")
        os.makedirs(new_images_dir, exist_ok=True)
        
        # Copy images
        old_images_dir = os.path.join(data_dir, chunk, "images")
        if os.path.exists(old_images_dir):
            for img in os.listdir(old_images_dir):
                if img.endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(old_images_dir, img)
                    dst = os.path.join(new_images_dir, img)
                    shutil.copy2(src, dst)
        
        # Remove original images
        if os.path.exists(old_images_dir):
            shutil.rmtree(old_images_dir)
            os.makedirs(old_images_dir)  # Create empty images directory
            
        # Create README in the empty images directory
        create_readme(old_images_dir)
        
        print(f"Completed {chunk}")

if __name__ == "__main__":
    migrate_images() 