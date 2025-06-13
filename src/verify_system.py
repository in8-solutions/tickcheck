import sys
import os
import subprocess
import importlib
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return version.major >= 3 and version.minor >= 8

def check_gpu():
    """Check GPU availability and CUDA."""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        return True
    except ImportError:
        print("PyTorch not installed")
        return False

def check_packages():
    """Check required Python packages."""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'albumentations': 'Albumentations',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'pandas': 'Pandas'
    }
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            all_installed = False
    return all_installed

def check_system_packages():
    """Check required system packages."""
    try:
        # Check for python3-tk
        result = subprocess.run(['dpkg', '-l', 'python3-tk'], 
                              capture_output=True, text=True)
        if 'python3-tk' in result.stdout:
            print("✓ python3-tk installed")
            return True
        else:
            print("✗ python3-tk not installed")
            return False
    except Exception as e:
        print(f"Error checking system packages: {e}")
        return False

def test_gui():
    """Test GUI functionality."""
    try:
        # Create a test window
        root = tk.Tk()
        root.title("GUI Test")
        
        # Create a label
        label = ttk.Label(root, text="GUI Test Window")
        label.pack(padx=20, pady=20)
        
        # Create a test image using OpenCV
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 2)
        cv2.putText(img, "OpenCV Test", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert OpenCV image to PIL format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        
        # Create an image label
        img_label = ttk.Label(root, image=img)
        img_label.image = img
        img_label.pack(padx=20, pady=20)
        
        # Add a quit button
        quit_button = ttk.Button(root, text="Close Test Window", command=root.quit)
        quit_button.pack(pady=20)
        
        print("✓ GUI test window opened successfully")
        root.mainloop()
        return True
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False

def main():
    print("=== System Verification ===")
    print("\n1. Checking Python version...")
    python_ok = check_python_version()
    
    print("\n2. Checking GPU and CUDA...")
    gpu_ok = check_gpu()
    
    print("\n3. Checking Python packages...")
    packages_ok = check_packages()
    
    print("\n4. Checking system packages...")
    system_ok = check_system_packages()
    
    print("\n5. Testing GUI functionality...")
    print("A window should appear. Close it to continue.")
    gui_ok = test_gui()
    
    print("\n=== Verification Summary ===")
    print(f"Python version: {'✓' if python_ok else '✗'}")
    print(f"GPU/CUDA: {'✓' if gpu_ok else '✗'}")
    print(f"Python packages: {'✓' if packages_ok else '✗'}")
    print(f"System packages: {'✓' if system_ok else '✗'}")
    print(f"GUI functionality: {'✓' if gui_ok else '✗'}")
    
    if all([python_ok, gpu_ok, packages_ok, system_ok, gui_ok]):
        print("\n✓ All checks passed! Your system is properly configured.")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")

if __name__ == "__main__":
    main() 