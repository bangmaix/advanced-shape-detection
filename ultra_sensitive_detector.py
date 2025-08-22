#!/usr/bin/env python3
"""
Ultra Sensitive Detector - Deteksi maksimal petak dengan multiple enhancement techniques
Kombinasi berbagai metode untuk mendeteksi semua petak yang mungkin ada
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

class UltraSensitiveDetector:
    def __init__(self):
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
            '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#F1C40F'
        ]
        
    def ultra_enhance_lines(self, image):
        """Ultra enhancement dengan multiple techniques"""
        print("🔍 Ultra enhancing thin lines with multiple techniques...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Technique 1: Multiple Canny edge detection dengan parameter berbeda
        edges1 = cv2.Canny(gray, 30, 100)  # Lebih sensitif
        edges2 = cv2.Canny(gray, 50, 150)  # Standard
        edges3 = cv2.Canny(gray, 70, 200)  # Lebih strict
        
        # Technique 2: Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        _, sobel_thresh = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
        
        # Technique 3: Multiple adaptive thresholds
        adaptive1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        adaptive2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
        
        # Technique 4: Multiple simple thresholds
        _, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        _, thresh2 = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        _, thresh3 = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
        _, thresh4 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Technique 5: Laplacian dengan multiple thresholds
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, lap1 = cv2.threshold(laplacian, 5, 255, cv2.THRESH_BINARY)
        _, lap2 = cv2.threshold(laplacian, 15, 255, cv2.THRESH_BINARY)
        _, lap3 = cv2.threshold(laplacian, 25, 255, cv2.THRESH_BINARY)
        
        # Combine all techniques
        combined = cv2.bitwise_or(edges1, edges2)
        combined = cv2.bitwise_or(combined, edges3)
        combined = cv2.bitwise_or(combined, sobel_thresh)
        combined = cv2.bitwise_or(combined, adaptive1)
        combined = cv2.bitwise_or(combined, adaptive2)
        combined = cv2.bitwise_or(combined, thresh1)
        combined = cv2.bitwise_or(combined, thresh2)
        combined = cv2.bitwise_or(combined, thresh3)
        combined = cv2.bitwise_or(combined, thresh4)
        combined = cv2.bitwise_or(combined, lap1)
        combined = cv2.bitwise_or(combined, lap2)
        combined = cv2.bitwise_or(combined, lap3)
        
        # Aggressive morphological operations
        # Close small gaps
        kernel_close = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate untuk menebalkan garis
        kernel_dilate = np.ones((2, 2), np.uint8)
        combined = cv2.dilate(combined, kernel_dilate, iterations=2)
        
        # Close lagi untuk menghubungkan garis yang terputus
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Final dilation untuk memastikan garis cukup tebal
        kernel_final = np.ones((3, 3), np.uint8)
        enhanced = cv2.dilate(combined, kernel_final, iterations=1)
        
        return enhanced, gray, edges2
    
    def detect_all_possible_rectangles(self, enhanced_image, original_gray):
        """Deteksi semua rectangle yang mungkin dengan parameter yang sangat sensitif"""
        print("🎯 Detecting all possible rectangles with ultra-sensitive parameters...")
        
        # Find contours dengan multiple retrieval modes
        contours_external, _ = cv2.findContours(enhanced_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, _ = cv2.findContours(enhanced_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine contours
        all_contours = contours_external + contours_tree
        print(f"   Found {len(all_contours)} total contours")
        
        # Ultra-sensitive filtering
        rectangles = []
        min_area = 20  # Sangat kecil untuk menangkap petak kecil
        max_area = 100000  # Sangat besar untuk menangkap petak besar
        
        for i, contour in enumerate(all_contours):
            area = cv2.contourArea(contour)
            
            # Filter berdasarkan area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter berdasarkan aspect ratio (sangat longgar)
            if aspect_ratio < 0.1 or aspect_ratio > 10.0:
                continue
            
            # Calculate rectangularity
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Filter berdasarkan rectangularity (sangat longgar)
            if rectangularity < 0.2:
                continue
            
            # Approximate contour
            epsilon = 0.03 * cv2.arcLength(contour, True)  # Lebih longgar
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Filter berdasarkan vertices (3-10 untuk menangkap berbagai bentuk)
            if len(approx) < 3 or len(approx) > 10:
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
        
        print(f"   Filtered to {len(rectangles)} potential rectangles")
        return rectangles
    
    def smart_duplicate_removal(self, rectangles, distance_threshold=15):
        """Smart removal of overlapping rectangles"""
        if len(rectangles) <= 1:
            return rectangles
        
        # Sort by area (largest first)
        sorted_rects = sorted(rectangles, key=lambda x: x['area'], reverse=True)
        
        filtered = []
        used_centers = set()
        
        for rect in sorted_rects:
            center_key = (rect['center'][0] // 10, rect['center'][1] // 10)
            
            # Check if this center is already used
            is_duplicate = False
            for used_center in used_centers:
                dist = np.sqrt((center_key[0] - used_center[0])**2 + 
                             (center_key[1] - used_center[1])**2)
                if dist < distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(rect)
                used_centers.add(center_key)
        
        print(f"   Smart duplicate removal: {len(rectangles)} -> {len(filtered)} rectangles")
        return filtered
    
    def create_comprehensive_visualization(self, image, enhanced_image, edges, rectangles, output_path):
        """Buat visualisasi komprehensif"""
        try:
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            
            # Original image
            if len(image.shape) == 3:
                axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[0,0].imshow(image, cmap='gray')
            axes[0,0].set_title('Original Image', fontsize=10, fontweight='bold')
            axes[0,0].axis('off')
            
            # Grayscale
            axes[0,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cmap='gray')
            axes[0,1].set_title('Grayscale', fontsize=10, fontweight='bold')
            axes[0,1].axis('off')
            
            # Edge detection
            axes[0,2].imshow(edges, cmap='gray')
            axes[0,2].set_title('Edge Detection', fontsize=10, fontweight='bold')
            axes[0,2].axis('off')
            
            # Ultra enhanced lines
            axes[0,3].imshow(enhanced_image, cmap='gray')
            axes[0,3].set_title('Ultra Enhanced Lines', fontsize=10, fontweight='bold')
            axes[0,3].axis('off')
            
            # Detected rectangles overlay
            if len(image.shape) == 3:
                axes[1,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,0].imshow(image, cmap='gray')
            
            # Draw rectangles
            for i, rect in enumerate(rectangles):
                color = self.colors[i % len(self.colors)]
                contour = rect['contour']
                
                # Draw filled contour
                axes[1,0].fill(contour[:, 0, 0], contour[:, 0, 1], 
                             color=color, alpha=0.6, edgecolor='black', linewidth=1)
                
                # Add number
                axes[1,0].text(rect['center'][0], rect['center'][1], str(i+1), 
                             ha='center', va='center', fontsize=6, fontweight='bold',
                             color='black', bbox=dict(boxstyle="round,pad=0.1", 
                                                     facecolor='white', alpha=0.9))
            
            axes[1,0].set_title(f'All Detected Petak ({len(rectangles)} total)', fontsize=10, fontweight='bold')
            axes[1,0].axis('off')
            
            # Enhanced lines overlay
            if len(image.shape) == 3:
                axes[1,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,1].imshow(image, cmap='gray')
            
            # Overlay enhanced lines
            enhanced_overlay = enhanced_image.copy()
            enhanced_overlay[enhanced_overlay > 0] = 255
            axes[1,1].imshow(enhanced_overlay, cmap='Reds', alpha=0.4)
            axes[1,1].set_title('Enhanced Lines Overlay', fontsize=10, fontweight='bold')
            axes[1,1].axis('off')
            
            # Area distribution
            if rectangles:
                areas = [rect['area'] for rect in rectangles]
                axes[1,2].hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1,2].set_title('Area Distribution', fontsize=10, fontweight='bold')
                axes[1,2].set_xlabel('Area (pixels)')
                axes[1,2].set_ylabel('Count')
            else:
                axes[1,2].text(0.5, 0.5, 'No rectangles detected', ha='center', va='center')
                axes[1,2].set_title('Area Distribution', fontsize=10, fontweight='bold')
            
            # Aspect ratio distribution
            if rectangles:
                aspect_ratios = [rect['aspect_ratio'] for rect in rectangles]
                axes[1,3].hist(aspect_ratios, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1,3].set_title('Aspect Ratio Distribution', fontsize=10, fontweight='bold')
                axes[1,3].set_xlabel('Aspect Ratio')
                axes[1,3].set_ylabel('Count')
            else:
                axes[1,3].text(0.5, 0.5, 'No rectangles detected', ha='center', va='center')
                axes[1,3].set_title('Aspect Ratio Distribution', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Comprehensive visualization saved: {output_path}")
            
            # Show figure
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️  Could not display figure: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False
    
    def save_ultra_detailed_report(self, rectangles, output_path):
        """Simpan laporan ultra detail"""
        try:
            report = {
                'total_petak': len(rectangles),
                'detection_method': 'ultra_sensitive_detection',
                'detection_parameters': {
                    'min_area': 20,
                    'max_area': 100000,
                    'min_aspect_ratio': 0.1,
                    'max_aspect_ratio': 10.0,
                    'min_rectangularity': 0.2,
                    'min_vertices': 3,
                    'max_vertices': 10
                },
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
            
            print(f"✅ Ultra detailed report saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            return False

def main():
    """Main function untuk ultra sensitive detection"""
    detector = UltraSensitiveDetector()
    
    print("🔬 ULTRA SENSITIVE DETECTOR")
    print("=" * 60)
    print("Deteksi maksimal petak dengan multiple enhancement techniques")
    print("Kombinasi berbagai metode untuk hasil optimal")
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
    
    # Ultra enhance lines
    enhanced_image, gray, edges = detector.ultra_enhance_lines(image)
    
    # Detect all possible rectangles
    rectangles = detector.detect_all_possible_rectangles(enhanced_image, gray)
    
    # Smart duplicate removal
    rectangles = detector.smart_duplicate_removal(rectangles)
    
    # Generate report
    print(f"\n📊 Ultra Sensitive Detection Results:")
    print(f"Total petak detected: {len(rectangles)}")
    
    if rectangles:
        print(f"\nPetak details:")
        print("-" * 90)
        print(f"{'ID':<3} {'Area':<8} {'Vertices':<8} {'Aspect Ratio':<12} {'Rectangularity':<12}")
        print("-" * 90)
        
        for i, rect in enumerate(rectangles):
            print(f"{i+1:<3} {int(rect['area']):<8} {rect['vertices']:<8} {rect['aspect_ratio']:.2f}        {rect['rectangularity']:.3f}")
    
    # Create visualization
    print(f"\n🎨 Creating comprehensive visualization...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f'ultra_sensitive_detection_{base_name}.png'
    
    success = detector.create_comprehensive_visualization(image, enhanced_image, edges, rectangles, output_path)
    
    # Save detailed report
    report_path = f'ultra_sensitive_report_{base_name}.json'
    detector.save_ultra_detailed_report(rectangles, report_path)
    
    if success:
        print(f"\n🎉 Ultra sensitive detection completed successfully!")
        print(f"📁 Files saved:")
        print(f"   - {output_path}")
        print(f"   - {report_path}")
        print(f"\n💡 Features:")
        print(f"   - Multiple edge detection techniques")
        print(f"   - Ultra-sensitive parameters")
        print(f"   - Smart duplicate removal")
        print(f"   - Comprehensive visualization")
        print(f"   - Detailed statistical analysis")
    else:
        print(f"\n⚠️  Detection completed with warnings")

if __name__ == "__main__":
    main()

