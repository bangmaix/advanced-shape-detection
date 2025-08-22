#!/usr/bin/env python3
"""
Test script untuk sistem deteksi shape
Menguji semua komponen dan memberikan feedback yang jelas
"""

import sys
import os

def test_imports():
    """Test semua import yang diperlukan"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib: {plt.matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        import json
        print("✅ JSON: Built-in module")
    except ImportError as e:
        print(f"❌ JSON: {e}")
        return False
    
    return True

def test_image_files():
    """Test keberadaan file gambar"""
    print("\n📁 Testing image files...")
    
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if image_files:
        print(f"✅ Found {len(image_files)} image files:")
        for i, file in enumerate(image_files[:5]):  # Show first 5
            print(f"   {i+1}. {file}")
        if len(image_files) > 5:
            print(f"   ... and {len(image_files) - 5} more")
        return image_files
    else:
        print("❌ No image files found in current directory")
        return []

def test_simple_detection():
    """Test deteksi sederhana"""
    print("\n🎯 Testing simple shape detection...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_image = np.zeros((300, 400, 3), dtype=np.uint8)
        test_image.fill(255)  # White background
        
        # Draw some rectangles
        cv2.rectangle(test_image, (50, 50), (150, 100), (0, 0, 0), 2)
        cv2.rectangle(test_image, (200, 150), (350, 250), (0, 0, 0), 2)
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"✅ Successfully detected {len(contours)} test shapes")
        return True
        
    except Exception as e:
        print(f"❌ Shape detection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 SISTEM DETEKSI SHAPE - DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing dependencies:")
        print("   pip install opencv-python numpy matplotlib")
        return
    
    # Test 2: Image files
    image_files = test_image_files()
    
    # Test 3: Simple detection
    if not test_simple_detection():
        print("\n❌ Detection test failed.")
        return
    
    print("\n🎉 ALL TESTS PASSED!")
    print("\nSistem siap digunakan. Anda bisa menjalankan:")
    print("   python adaptive_detector.py    # Untuk deteksi adaptif")
    print("   python demo_shape_detector.py  # Untuk deteksi sederhana")
    print("   python simple_precise_detector.py  # Untuk deteksi presisi")
    
    if image_files:
        print(f"\nGambar yang tersedia: {image_files[0]}")
    else:
        print("\n⚠️  Letakkan file gambar (.png, .jpg, dll) di folder ini untuk mulai deteksi")

if __name__ == "__main__":
    main()

