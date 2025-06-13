import os
import json
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

class AnnotationViewer:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = data_dir
        self.current_chunk_idx = 0
        self.current_image_idx = 0
        self.selected_box = None
        self.drawing_new = False
        self.new_box_start = None
        self.scale = 1.0
        
        # Get chunk directories
        self.chunk_dirs = self.get_chunk_dirs()
        if not self.chunk_dirs:
            raise ValueError("No chunk directories found in data/")
        
        # Load current chunk
        self.load_chunk(self.current_chunk_idx)
        
        # Setup UI
        self.setup_ui()
        
        # Load first image
        if self.images:
            self.load_current_image()
        
        # Save position on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def get_chunk_dirs(self):
        """Get list of chunk directories."""
        chunk_dirs = []
        for item in os.listdir(self.data_dir):
            if item.startswith('chunk_'):
                chunk_path = os.path.join(self.data_dir, item)
                if os.path.isdir(chunk_path):
                    chunk_dirs.append(chunk_path)
        return sorted(chunk_dirs)
    
    def load_chunk(self, chunk_idx):
        """Load a specific chunk's data."""
        chunk_dir = self.chunk_dirs[chunk_idx]
        self.current_chunk = os.path.basename(chunk_dir)
        
        # Load annotations
        annotations_file = os.path.join(chunk_dir, "annotations.json")
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Track the next available annotation ID
        self.next_annotation_id = max([ann['id'] for ann in self.annotations['annotations']], default=0) + 1
        
        # Create image_id to annotations mapping
        self.image_to_annotations = {}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_to_annotations:
                self.image_to_annotations[img_id] = []
            self.image_to_annotations[img_id].append(ann)
        
        # Create image info mapping
        self.images = self.annotations['images']
        self.current_image_idx = 0
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        top_control = ttk.Frame(main_container)
        top_control.pack(fill=tk.X)
        
        # Chunk navigation
        chunk_frame = ttk.Frame(top_control)
        chunk_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(chunk_frame, text="Chunk:").pack(side=tk.LEFT)
        self.chunk_var = tk.StringVar(value=f"{self.current_chunk}")
        chunk_label = ttk.Label(chunk_frame, textvariable=self.chunk_var)
        chunk_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(chunk_frame, text="Previous Chunk", 
                  command=self.prev_chunk).pack(side=tk.LEFT, padx=5)
        ttk.Button(chunk_frame, text="Next Chunk", 
                  command=self.next_chunk).pack(side=tk.LEFT, padx=5)
        
        # Jump to chunk frame
        jump_chunk_frame = ttk.Frame(top_control)
        jump_chunk_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(jump_chunk_frame, text="Jump to Chunk:").pack(side=tk.LEFT)
        self.jump_chunk_var = tk.StringVar()
        self.jump_chunk_entry = ttk.Entry(jump_chunk_frame, textvariable=self.jump_chunk_var, width=8)
        self.jump_chunk_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_chunk_frame, text="Go", command=self.jump_to_chunk).pack(side=tk.LEFT)
        
        # Jump to image frame
        jump_frame = ttk.Frame(top_control)
        jump_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(jump_frame, text="Jump to Image:").pack(side=tk.LEFT)
        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(jump_frame, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(jump_frame, text="Go", command=self.jump_to_image).pack(side=tk.LEFT)
        
        # Canvas for image display
        self.canvas = tk.Canvas(main_container)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Bottom control panel
        bottom_control = ttk.Frame(main_container)
        bottom_control.pack(fill=tk.X)
        
        # Navigation and edit buttons
        ttk.Button(bottom_control, text="Previous", command=self.prev_image).pack(side=tk.LEFT)
        ttk.Button(bottom_control, text="Next", command=self.next_image).pack(side=tk.LEFT)
        ttk.Button(bottom_control, text="Delete Box", command=self.delete_selected_box).pack(side=tk.LEFT)
        ttk.Button(bottom_control, text="Save Changes", command=self.save_changes).pack(side=tk.RIGHT)
        
        # Image counter and info
        self.info_label = ttk.Label(bottom_control, text="")
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Delete>', lambda e: self.delete_selected_box())
        self.root.bind('<Control-s>', lambda e: self.save_changes())
        self.root.bind('<Return>', lambda e: self.jump_to_image())
    
    def prev_chunk(self):
        """Go to previous chunk."""
        if self.current_chunk_idx > 0:
            self.save_changes()  # Save current chunk
            self.current_chunk_idx -= 1
            self.load_chunk(self.current_chunk_idx)
            self.chunk_var.set(self.current_chunk)
            self.load_current_image()
    
    def next_chunk(self):
        """Go to next chunk."""
        if self.current_chunk_idx < len(self.chunk_dirs) - 1:
            self.save_changes()  # Save current chunk
            self.current_chunk_idx += 1
            self.load_chunk(self.current_chunk_idx)
            self.chunk_var.set(self.current_chunk)
            self.load_current_image()
    
    def jump_to_chunk(self, *args):
        """Jump to a specific chunk number."""
        try:
            # Get the target chunk number (1-based index for user-friendliness)
            target = int(self.jump_chunk_var.get()) - 1
            if 0 <= target < len(self.chunk_dirs):
                self.save_changes()  # Save current chunk
                self.current_chunk_idx = target
                self.load_chunk(self.current_chunk_idx)
                self.chunk_var.set(self.current_chunk)
                self.load_current_image()
            else:
                messagebox.showwarning("Invalid Chunk Number", 
                                     f"Please enter a number between 1 and {len(self.chunk_dirs)}")
        except ValueError:
            messagebox.showwarning("Invalid Input", 
                                 "Please enter a valid number")
    
    def jump_to_image(self, *args):
        """Jump to a specific image number."""
        try:
            # Get the target image number (1-based index for user-friendliness)
            target = int(self.jump_var.get()) - 1
            if 0 <= target < len(self.images):
                self.current_image_idx = target
                self.selected_box = None
                self.load_current_image()
            else:
                messagebox.showwarning("Invalid Image Number", 
                                     f"Please enter a number between 1 and {len(self.images)}")
        except ValueError:
            messagebox.showwarning("Invalid Input", 
                                 "Please enter a valid number")
    
    def on_click(self, event):
        """Handle mouse click."""
        if not hasattr(self, 'current_image'):
            return
            
        # Convert click coordinates to image coordinates
        x = event.x / self.scale
        y = event.y / self.scale
        
        # Start drawing a new box
        self.drawing_new = True
        self.new_box_start = (x, y)
        self.selected_box = None
    
    def on_drag(self, event):
        """Handle mouse drag."""
        if not hasattr(self, 'current_image'):
            return
            
        # Convert coordinates to image coordinates
        x = event.x / self.scale
        y = event.y / self.scale
        
        if self.drawing_new:
            # Drawing a new box
            self.draw_temp_box(x, y)
    
    def draw_temp_box(self, current_x, current_y):
        """Draw temporary box while dragging."""
        if not hasattr(self, 'current_image'):
            return
            
        # Clear previous temporary box
        self.load_current_image()
        
        # Draw new temporary box
        start_x, start_y = self.new_box_start
        x1 = min(start_x, current_x)
        y1 = min(start_y, current_y)
        x2 = max(start_x, current_x)
        y2 = max(start_y, current_y)
        
        # Scale coordinates for display
        x1, y1 = x1 * self.scale, y1 * self.scale
        x2, y2 = x2 * self.scale, y2 * self.scale
        
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
    
    def on_release(self, event):
        """Handle mouse release."""
        if not hasattr(self, 'current_image'):
            return
            
        if self.drawing_new:
            # Convert coordinates to image coordinates
            x = event.x / self.scale
            y = event.y / self.scale
            
            # Calculate box coordinates
            start_x, start_y = self.new_box_start
            x1 = min(start_x, x)
            y1 = min(start_y, y)
            x2 = max(start_x, x)
            y2 = max(start_y, y)
            
            # Add new box
            self.add_new_box(x1, y1, x2 - x1, y2 - y1)
            
            self.drawing_new = False
            self.new_box_start = None
    
    def add_new_box(self, x, y, width, height):
        """Add a new bounding box annotation."""
        img_info = self.images[self.current_image_idx]
        
        # Create new annotation
        new_ann = {
            'id': self.next_annotation_id,
            'image_id': img_info['id'],
            'category_id': 1,  # Assuming single category
            'bbox': [x, y, width, height],
            'area': width * height,
            'iscrowd': 0
        }
        
        # Add to annotations
        self.annotations['annotations'].append(new_ann)
        self.next_annotation_id += 1
        
        # Update image mapping
        if img_info['id'] not in self.image_to_annotations:
            self.image_to_annotations[img_info['id']] = []
        self.image_to_annotations[img_info['id']].append(new_ann)
        
        # Update display
        self.load_current_image()
    
    def delete_selected_box(self):
        """Delete the selected bounding box."""
        if self.selected_box is not None:
            img_info = self.images[self.current_image_idx]
            ann = self.image_to_annotations[img_info['id']][self.selected_box]
            
            # Remove from annotations
            self.annotations['annotations'].remove(ann)
            
            # Remove from image mapping
            self.image_to_annotations[img_info['id']].pop(self.selected_box)
            
            self.selected_box = None
            self.load_current_image()
    
    def save_changes(self):
        """Save changes to the current chunk's annotation file."""
        chunk_dir = self.chunk_dirs[self.current_chunk_idx]
        annotations_file = os.path.join(chunk_dir, "annotations.json")
        
        with open(annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        messagebox.showinfo("Save Successful", 
                          f"Changes saved to {self.current_chunk}/annotations.json")
    
    def load_current_image(self):
        """Load and display the current image with its annotations."""
        if not self.images:
            return
            
        img_info = self.images[self.current_image_idx]
        img_path = os.path.join(self.data_dir, self.current_chunk, "images", os.path.basename(img_info['file_name']))
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {img_path}")
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate scale to fit window
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Ensure canvas is ready
            img_height, img_width = img.shape[:2]
            width_scale = canvas_width / img_width
            height_scale = canvas_height / img_height
            self.scale = min(width_scale, height_scale)
            
            # Resize image
            new_width = int(img_width * self.scale)
            new_height = int(img_height * self.scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Convert to PhotoImage
        img = Image.fromarray(img)
        self.current_image = ImageTk.PhotoImage(img)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_image)
        
        # Draw annotations
        if img_info['id'] in self.image_to_annotations:
            for i, ann in enumerate(self.image_to_annotations[img_info['id']]):
                bbox = ann['bbox']
                x1, y1 = bbox[0] * self.scale, bbox[1] * self.scale
                x2, y2 = (bbox[0] + bbox[2]) * self.scale, (bbox[1] + bbox[3]) * self.scale
                
                # Draw box
                color = 'red' if i == self.selected_box else 'green'
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
        
        # Update info label
        self.info_label.config(
            text=f"Image {self.current_image_idx + 1}/{len(self.images)} - {self.current_chunk}"
        )
    
    def prev_image(self):
        """Go to previous image."""
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.selected_box = None
            self.load_current_image()
    
    def next_image(self):
        """Go to next image."""
        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            self.selected_box = None
            self.load_current_image()
    
    def on_closing(self):
        """Handle window closing."""
        self.save_changes()
        self.root.destroy()

def main():
    root = tk.Tk()
    root.title("Tick Annotation Verifier")
    
    # Set window size
    root.geometry("1200x800")
    
    app = AnnotationViewer(root, "data")
    root.mainloop()

if __name__ == "__main__":
    main() 