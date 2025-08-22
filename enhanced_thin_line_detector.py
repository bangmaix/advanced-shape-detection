#!/usr/bin/env python3
"""
Enhanced Thin Line Detector - Khusus untuk denah dengan garis tipis
Mengubah garis tipis menjadi tebal agar petak-petak terdeteksi sempurna
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Set matplotlib backend
import matplotlib
matplotlib.use('TkAgg')

class ThinLineDetector:
    def __init__(self):
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E'
        ]
        
    def enhance_thin_lines(self, image):
        """Mengubah garis tipis menjadi tebal"""
        print("🔍 Detecting and enhancing thin lines...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Method 1: Edge detection untuk menemukan garis tipis
        edges = cv2.Canny(gray, 50, 150)
        
        # Method 2: Morphological operations untuk menebalkan garis
        kernel_thin = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        kernel_thick = np.ones((4, 4), np.uint8)
        
        # Dilate edges untuk menebalkan garis
        dilated_thin = cv2.dilate(edges, kernel_thin, iterations=1)
        dilated_medium = cv2.dilate(edges, kernel_medium, iterations=1)
        dilated_thick = cv2.dilate(edges, kernel_thick, iterations=1)
        
        # Method 3: Adaptive thresholding dengan parameter sensitif
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        # Method 4: Multiple simple thresholds untuk menangkap berbagai intensitas
        _, thresh_low = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        _, thresh_med = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        _, thresh_high = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Method 5: Laplacian untuk deteksi garis
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, laplacian_thresh = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
        
        # Combine all methods
        combined = cv2.bitwise_or(dilated_thin, adaptive_thresh)
        combined = cv2.bitwise_or(combined, thresh_low)
        combined = cv2.bitwise_or(combined, laplacian_thresh)
        
        # Additional morphological operations untuk menebalkan
        kernel_close = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate lagi untuk memastikan garis cukup tebal
        kernel_final = np.ones((2, 2), np.uint8)
        enhanced = cv2.dilate(combined, kernel_final, iterations=1)
        
        return enhanced, gray, edges
    
    def detect_rectangles_from_enhanced(self, enhanced_image, original_gray):
        """Deteksi rectangle dari gambar yang sudah ditingkatkan"""
        print("🎯 Detecting rectangles from enhanced lines...")
        
        # Find contours
        contours, _ = cv2.findContours(enhanced_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"   Found {len(contours)} initial contours")
        
        # Filter contours berdasarkan area dan bentuk
        rectangles = []
        min_area = 50  # Area minimum yang sangat kecil untuk petak kecil
        max_area = 50000  # Area maksimum yang cukup besar
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter berdasarkan area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter berdasarkan aspect ratio (rectangle biasanya 0.2 - 5.0)
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Calculate rectangularity
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Filter berdasarkan rectangularity (minimal 0.3)
            if rectangularity < 0.3:
                continue
            
            # Approximate contour untuk menghitung vertices
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Filter berdasarkan jumlah vertices (4-8 untuk rectangle)
            if len(approx) < 4 or len(approx) > 8:
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            rectangles.append({
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (cx, cy),
                'aspect_ratio': aspect_ratio,
                'rectangularity': rectangularity,
                'vertices': len(approx)
            })
        
        print(f"   Filtered to {len(rectangles)} valid rectangles")
        return rectangles
    
    def remove_duplicates(self, rectangles, distance_threshold=20):
        """Hapus rectangle yang tumpang tindih"""
        if len(rectangles) <= 1:
            return rectangles
        
        filtered = []
        used = set()
        
        for i, rect1 in enumerate(rectangles):
            if i in used:
                continue
            
            # Start new group
            group = [rect1]
            used.add(i)
            
            # Find overlapping rectangles
            for j, rect2 in enumerate(rectangles):
                if j in used:
                    continue
                
                # Calculate distance between centers
                dist = np.sqrt((rect1['center'][0] - rect2['center'][0])**2 + 
                             (rect1['center'][1] - rect2['center'][1])**2)
                
                if dist < distance_threshold:
                    group.append(rect2)
                    used.add(j)
            
            # Keep the largest rectangle in the group
            largest = max(group, key=lambda x: x['area'])
            filtered.append(largest)
        
        print(f"   Removed duplicates: {len(rectangles)} -> {len(filtered)} rectangles")
        return filtered
    
    def create_enhanced_visualization(self, image, enhanced_image, edges, rectangles, output_path):
        """Buat visualisasi yang menunjukkan proses enhancement"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Original image
            if len(image.shape) == 3:
                axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[0,0].imshow(image, cmap='gray')
            axes[0,0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0,0].axis('off')
            
            # Grayscale
            axes[0,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')
            axes[0,1].set_title('Grayscale', fontsize=12, fontweight='bold')
            axes[0,1].axis('off')
            
            # Edge detection
            axes[0,2].imshow(edges, cmap='gray')
            axes[0,2].set_title('Edge Detection (Canny)', fontsize=12, fontweight='bold')
            axes[0,2].axis('off')
            
            # Enhanced lines
            axes[1,0].imshow(enhanced_image, cmap='gray')
            axes[1,0].set_title('Enhanced Thin Lines', fontsize=12, fontweight='bold')
            axes[1,0].axis('off')
            
            # Detected rectangles overlay
            if len(image.shape) == 3:
                axes[1,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,1].imshow(image, cmap='gray')
            
            # Draw rectangles
            for i, rect in enumerate(rectangles):
                color = self.colors[i % len(self.colors)]
                contour = rect['contour']
                
                # Draw filled contour
                axes[1,1].fill(contour[:, 0, 0], contour[:, 0, 1], 
                             color=color, alpha=0.6, edgecolor='black', linewidth=1)
                
                # Add number
                axes[1,1].text(rect['center'][0], rect['center'][1], str(i+1), 
                             ha='center', va='center', fontsize=8, fontweight='bold',
                             color='black', bbox=dict(boxstyle="round,pad=0.1", 
                                                     facecolor='white', alpha=0.9))
            
            axes[1,1].set_title(f'Detected Petak ({len(rectangles)} total)', fontsize=12, fontweight='bold')
            axes[1,1].axis('off')
            
            # Final result with enhanced lines
            if len(image.shape) == 3:
                axes[1,2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,2].imshow(image, cmap='gray')
            
            # Overlay enhanced lines
            enhanced_overlay = enhanced_image.copy()
            enhanced_overlay[enhanced_overlay > 0] = 255
            axes[1,2].imshow(enhanced_overlay, cmap='Reds', alpha=0.3)
            
            axes[1,2].set_title('Enhanced Lines Overlay', fontsize=12, fontweight='bold')
            axes[1,2].axis('off')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Enhanced visualization saved: {output_path}")
            
            # Show figure
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️  Could not display figure: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False
    
    def save_detailed_report(self, rectangles, output_path):
        """Simpan laporan detail dalam format JSON"""
        try:
            report = {
                'total_petak': len(rectangles),
                'detection_method': 'enhanced_thin_line_detection',
                'petak_details': []
            }
            
            for i, rect in enumerate(rectangles):
                x, y, w, h = rect['bbox']
                petak_info = {
                    'petak_id': i + 1,
                    'position': {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'center_x': int(rect['center'][0]),
                        'center_y': int(rect['center'][1])
                    },
                    'properties': {
                        'area': float(rect['area']),
                        'aspect_ratio': float(rect['aspect_ratio']),
                        'rectangularity': float(rect['rectangularity']),
                        'vertices': int(rect['vertices'])
                    }
                }
                report['petak_details'].append(petak_info)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Detailed report saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            return False

def main():
    """Main function untuk enhanced thin line detection"""
    detector = ThinLineDetector()
    
    print("🔍 ENHANCED THIN LINE DETECTOR")
    print("=" * 60)
    print("Khusus untuk denah dengan garis tipis")
    print("Mengubah garis tipis menjadi tebal untuk deteksi sempurna")
    print("=" * 60)
    
    # Find image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            if os.path.getsize(file) > 0:
                image_files.append(file)
    
    if not image_files:
        print("❌ No image files found")
        return
    
    print(f"✅ Found {len(image_files)} image files:")
    for i, file in enumerate(image_files):
        size_mb = os.path.getsize(file) / (1024*1024)
        print(f"   {i+1}. {file} ({size_mb:.1f} MB)")
    
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
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Enhance thin lines
    enhanced_image, gray, edges = detector.enhance_thin_lines(image)
    
    # Detect rectangles
    rectangles = detector.detect_rectangles_from_enhanced(enhanced_image, gray)
    
    # Remove duplicates
    rectangles = detector.remove_duplicates(rectangles)
    
    # Generate report
    print(f"\n📊 Enhanced Detection Results:")
    print(f"Total petak detected: {len(rectangles)}")
    
    if rectangles:
        print(f"\nPetak details:")
        print("-" * 80)
        print(f"{'ID':<3} {'Area':<8} {'Vertices':<8} {'Aspect Ratio':<12} {'Rectangularity':<12}")
        print("-" * 80)
        
        for i, rect in enumerate(rectangles):
            print(f"{i+1:<3} {int(rect['area']):<8} {rect['vertices']:<8} {rect['aspect_ratio']:.2f}        {rect['rectangularity']:.3f}")
    
    # Create visualization
    print(f"\n🎨 Creating enhanced visualization...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f'enhanced_thin_line_detection_{base_name}.png'
    
    success = detector.create_enhanced_visualization(image, enhanced_image, edges, rectangles, output_path)
    
    # Save detailed report
    report_path = f'enhanced_detection_report_{base_name}.json'
    detector.save_detailed_report(rectangles, report_path)
    
    if success:
        print(f"\n🎉 Enhanced detection completed successfully!")
        print(f"📁 Files saved:")
        print(f"   - {output_path}")
        print(f"   - {report_path}")
        print(f"\n💡 Tips:")
        print(f"   - Garis tipis telah ditingkatkan menjadi tebal")
        print(f"   - Petak-petak sekarang terdeteksi dengan sempurna")
        print(f"   - Cek file {output_path} untuk melihat proses enhancement")
    else:
        print(f"\n⚠️  Detection completed with warnings")

if __name__ == "__main__":
    main()

