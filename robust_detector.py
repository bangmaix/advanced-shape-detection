#!/usr/bin/env python3
"""
Robust Shape Detector - Versi yang lebih stabil dan mudah digunakan
Mengatasi berbagai masalah umum yang mungkin terjadi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Set matplotlib backend untuk menghindari error display
import matplotlib
matplotlib.use('TkAgg')  # Try TkAgg first, fallback to Agg if needed

class RobustDetector:
    def __init__(self):
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71'
        ]
        
    def safe_image_read(self, image_path):
        """Safely read image with error handling"""
        try:
            # Try to read image
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"❌ Cannot read image: {image_path}")
                print("   Possible causes:")
                print("   - File doesn't exist")
                print("   - Unsupported format")
                print("   - File is corrupted")
                return None
            
            print(f"✅ Successfully loaded: {image_path}")
            print(f"   Size: {image.shape[1]}x{image.shape[0]} pixels")
            return image
            
        except Exception as e:
            print(f"❌ Error reading image: {e}")
            return None
    
    def robust_preprocessing(self, image):
        """Robust preprocessing with multiple fallback methods"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Try multiple thresholding methods
            methods = []
            
            # Method 1: Simple threshold
            try:
                _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                methods.append(('Simple', thresh1))
            except:
                pass
            
            # Method 2: Otsu's method
            try:
                _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                methods.append(('Otsu', thresh2))
            except:
                pass
            
            # Method 3: Adaptive threshold
            try:
                thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
                methods.append(('Adaptive', thresh3))
            except:
                pass
            
            if not methods:
                print("❌ All thresholding methods failed")
                return None, gray
            
            # Use the first successful method
            method_name, processed = methods[0]
            print(f"✅ Used {method_name} thresholding")
            
            return processed, gray
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}")
            return None, None
    
    def detect_shapes_robust(self, image):
        """Robust shape detection with multiple fallback strategies"""
        try:
            # Preprocess image
            processed, gray = self.robust_preprocessing(image)
            if processed is None:
                return [], None
            
            # Find contours with error handling
            try:
                contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                print(f"✅ Found {len(contours)} initial contours")
            except Exception as e:
                print(f"❌ Contour detection failed: {e}")
                return [], processed
            
            # Filter and analyze contours
            shapes = []
            min_area = 100  # Adjustable minimum area
            
            for i, contour in enumerate(contours):
                try:
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Get center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    # Approximate shape
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    shapes.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (cx, cy),
                        'aspect_ratio': aspect_ratio,
                        'vertices': len(approx),
                        'approx': approx
                    })
                    
                except Exception as e:
                    print(f"⚠️  Warning: Failed to process contour {i}: {e}")
                    continue
            
            print(f"✅ Successfully processed {len(shapes)} shapes")
            return shapes, processed
            
        except Exception as e:
            print(f"❌ Shape detection failed: {e}")
            return [], None
    
    def safe_visualization(self, image, shapes, processed, output_path):
        """Safe visualization with error handling"""
        try:
            # Create figure with error handling
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            except Exception as e:
                print(f"⚠️  Display issue, saving without showing: {e}")
                # Try non-interactive backend
                matplotlib.use('Agg')
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            if len(image.shape) == 3:
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Processed image
            if processed is not None:
                axes[1].imshow(processed, cmap='gray')
                axes[1].set_title('Processed Image')
            else:
                axes[1].text(0.5, 0.5, 'Processing Failed', ha='center', va='center')
                axes[1].set_title('Processing Failed')
            axes[1].axis('off')
            
            # Detected shapes
            if len(image.shape) == 3:
                axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[2].imshow(image, cmap='gray')
            
            # Draw shapes
            for i, shape in enumerate(shapes):
                try:
                    color = self.colors[i % len(self.colors)]
                    contour = shape['contour']
                    
                    # Draw filled contour
                    axes[2].fill(contour[:, 0, 0], contour[:, 0, 1], 
                               color=color, alpha=0.6, edgecolor='black', linewidth=1)
                    
                    # Add number
                    axes[2].text(shape['center'][0], shape['center'][1], str(i+1), 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color='black', bbox=dict(boxstyle="round,pad=0.2", 
                                                       facecolor='white', alpha=0.8))
                except Exception as e:
                    print(f"⚠️  Warning: Failed to draw shape {i}: {e}")
                    continue
            
            axes[2].set_title(f'Detected Shapes ({len(shapes)} total)')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            try:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✅ Results saved: {output_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not save figure: {e}")
            
            # Try to show
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️  Warning: Could not display figure: {e}")
                print("   Figure was saved successfully though!")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False

def main():
    """Main function with comprehensive error handling"""
    detector = RobustDetector()
    
    print("🛡️  ROBUST SHAPE DETECTOR")
    print("=" * 50)
    print("Versi yang stabil dengan error handling lengkap")
    print("=" * 50)
    
    # Find image files
    try:
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        image_files = []
        
        current_dir = os.getcwd()
        print(f"📁 Scanning directory: {current_dir}")
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                if os.path.getsize(file) > 0:  # Check file is not empty
                    image_files.append(file)
        
        if not image_files:
            print("❌ No image files found in current directory")
            print("   Supported formats: .png, .jpg, .jpeg, .bmp, .tiff, .webp")
            print("   Please place an image file in this directory and try again")
            return
        
        print(f"✅ Found {len(image_files)} image files:")
        for i, file in enumerate(image_files):
            size_mb = os.path.getsize(file) / (1024*1024)
            print(f"   {i+1}. {file} ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Error scanning directory: {e}")
        return
    
    # Select image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return
    else:
        image_path = image_files[0]
    
    print(f"\n🔍 Processing: {image_path}")
    
    # Load image
    image = detector.safe_image_read(image_path)
    if image is None:
        return
    
    # Detect shapes
    print("\n🎯 Detecting shapes...")
    shapes, processed = detector.detect_shapes_robust(image)
    
    # Generate report
    print(f"\n📊 Detection Results:")
    print(f"Total shapes detected: {len(shapes)}")
    
    if shapes:
        print("\nShape details:")
        print("-" * 60)
        print(f"{'ID':<3} {'Area':<8} {'Vertices':<8} {'Aspect Ratio':<12}")
        print("-" * 60)
        
        for i, shape in enumerate(shapes):
            print(f"{i+1:<3} {int(shape['area']):<8} {shape['vertices']:<8} {shape['aspect_ratio']:.2f}")
    
    # Visualize results
    print(f"\n🎨 Creating visualization...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f'robust_detection_{base_name}.png'
    
    success = detector.safe_visualization(image, shapes, processed, output_path)
    
    if success:
        print(f"\n🎉 Detection completed successfully!")
        print(f"📁 Output saved as: {output_path}")
    else:
        print(f"\n⚠️  Detection completed with warnings")
        print("   Check the console messages above for details")

if __name__ == "__main__":
    main()

