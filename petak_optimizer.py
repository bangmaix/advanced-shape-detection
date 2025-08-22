#!/usr/bin/env python3
"""
Petak Optimizer - Khusus untuk mendeteksi petak-petak kecil dengan parameter optimal
Fokus pada deteksi maksimal petak dengan ukuran yang bervariasi
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

class PetakOptimizer:
    def __init__(self):
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
            '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#F1C40F',
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'
        ]
        
    def optimize_for_petak(self, image):
        """Optimasi khusus untuk deteksi petak"""
        print("🔧 Optimizing image specifically for petak detection...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Noise reduction dengan bilateral filter
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Step 2: Multiple edge detection dengan parameter yang dioptimalkan
        # Canny dengan parameter sensitif untuk garis tipis
        edges1 = cv2.Canny(denoised, 20, 60)  # Sangat sensitif
        edges2 = cv2.Canny(denoised, 40, 120)  # Sensitif
        edges3 = cv2.Canny(denoised, 60, 180)  # Standard
        
        # Step 3: Sobel dengan parameter yang dioptimalkan
        sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        _, sobel_thresh = cv2.threshold(sobel, 30, 255, cv2.THRESH_BINARY)
        
        # Step 4: Adaptive thresholding dengan parameter yang dioptimalkan
        adaptive1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 9, 1)  # Lebih sensitif
        adaptive2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, 9, 1)
        
        # Step 5: Multiple simple thresholds untuk menangkap berbagai intensitas
        _, thresh1 = cv2.threshold(denoised, 60, 255, cv2.THRESH_BINARY_INV)
        _, thresh2 = cv2.threshold(denoised, 100, 255, cv2.THRESH_BINARY_INV)
        _, thresh3 = cv2.threshold(denoised, 140, 255, cv2.THRESH_BINARY_INV)
        _, thresh4 = cv2.threshold(denoised, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Step 6: Laplacian untuk deteksi garis halus
        laplacian = cv2.Laplacian(denoised, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, lap1 = cv2.threshold(laplacian, 3, 255, cv2.THRESH_BINARY)
        _, lap2 = cv2.threshold(laplacian, 8, 255, cv2.THRESH_BINARY)
        _, lap3 = cv2.threshold(laplacian, 15, 255, cv2.THRESH_BINARY)
        
        # Step 7: Combine semua teknik
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
        
        # Step 8: Morphological operations yang dioptimalkan untuk petak
        # Close gaps kecil
        kernel_close_small = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_small)
        
        # Dilate untuk menebalkan garis tipis
        kernel_dilate = np.ones((2, 2), np.uint8)
        combined = cv2.dilate(combined, kernel_dilate, iterations=1)
        
        # Close lagi untuk menghubungkan garis
        kernel_close_medium = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_medium)
        
        # Final dilation untuk memastikan garis cukup tebal
        kernel_final = np.ones((2, 2), np.uint8)
        optimized = cv2.dilate(combined, kernel_final, iterations=1)
        
        return optimized, gray, edges2
    
    def detect_petak_with_optimized_params(self, optimized_image, original_gray):
        """Deteksi petak dengan parameter yang dioptimalkan"""
        print("🎯 Detecting petak with optimized parameters...")
        
        # Find contours dengan multiple retrieval modes
        contours_external, _ = cv2.findContours(optimized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, _ = cv2.findContours(optimized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine contours
        all_contours = contours_external + contours_tree
        print(f"   Found {len(all_contours)} total contours")
        
        # Parameter yang dioptimalkan untuk petak
        petak_list = []
        min_area = 10  # Sangat kecil untuk petak kecil
        max_area = 50000  # Cukup besar untuk petak besar
        
        for i, contour in enumerate(all_contours):
            area = cv2.contourArea(contour)
            
            # Filter berdasarkan area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter berdasarkan aspect ratio (sesuai karakteristik petak)
            if aspect_ratio < 0.2 or aspect_ratio > 8.0:
                continue
            
            # Calculate rectangularity
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Filter berdasarkan rectangularity (lebih longgar untuk petak)
            if rectangularity < 0.15:
                continue
            
            # Approximate contour
            epsilon = 0.04 * cv2.arcLength(contour, True)  # Lebih longgar
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Filter berdasarkan vertices (3-12 untuk menangkap berbagai bentuk petak)
            if len(approx) < 3 or len(approx) > 12:
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            petak_list.append({
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (cx, cy),
                'aspect_ratio': aspect_ratio,
                'rectangularity': rectangularity,
                'vertices': len(approx)
            })
        
        print(f"   Filtered to {len(petak_list)} potential petak")
        return petak_list
    
    def advanced_duplicate_removal(self, petak_list, distance_threshold=12):
        """Advanced duplicate removal dengan clustering"""
        if len(petak_list) <= 1:
            return petak_list
        
        # Sort by area (largest first)
        sorted_petak = sorted(petak_list, key=lambda x: x['area'], reverse=True)
        
        filtered = []
        used_positions = set()
        
        for petak in sorted_petak:
            # Create position key dengan granularity yang lebih halus
            pos_key = (petak['center'][0] // 8, petak['center'][1] // 8)
            
            # Check if position is already used
            is_duplicate = False
            for used_pos in used_positions:
                dist = np.sqrt((pos_key[0] - used_pos[0])**2 + 
                             (pos_key[1] - used_pos[1])**2)
                if dist < distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(petak)
                used_positions.add(pos_key)
        
        print(f"   Advanced duplicate removal: {len(petak_list)} -> {len(filtered)} petak")
        return filtered
    
    def create_petak_visualization(self, image, optimized_image, edges, petak_list, output_path):
        """Buat visualisasi khusus untuk petak"""
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
            
            # Optimized image
            axes[0,3].imshow(optimized_image, cmap='gray')
            axes[0,3].set_title('Optimized for Petak', fontsize=10, fontweight='bold')
            axes[0,3].axis('off')
            
            # Detected petak overlay
            if len(image.shape) == 3:
                axes[1,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,0].imshow(image, cmap='gray')
            
            # Draw petak
            for i, petak in enumerate(petak_list):
                color = self.colors[i % len(self.colors)]
                contour = petak['contour']
                
                # Draw filled contour
                axes[1,0].fill(contour[:, 0, 0], contour[:, 0, 1], 
                             color=color, alpha=0.6, edgecolor='black', linewidth=1)
                
                # Add number
                axes[1,0].text(petak['center'][0], petak['center'][1], str(i+1), 
                             ha='center', va='center', fontsize=6, fontweight='bold',
                             color='black', bbox=dict(boxstyle="round,pad=0.1", 
                                                     facecolor='white', alpha=0.9))
            
            axes[1,0].set_title(f'Detected Petak ({len(petak_list)} total)', fontsize=10, fontweight='bold')
            axes[1,0].axis('off')
            
            # Optimized lines overlay
            if len(image.shape) == 3:
                axes[1,1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,1].imshow(image, cmap='gray')
            
            # Overlay optimized lines
            optimized_overlay = optimized_image.copy()
            optimized_overlay[optimized_overlay > 0] = 255
            axes[1,1].imshow(optimized_overlay, cmap='Reds', alpha=0.4)
            axes[1,1].set_title('Optimized Lines Overlay', fontsize=10, fontweight='bold')
            axes[1,1].axis('off')
            
            # Area distribution
            if petak_list:
                areas = [petak['area'] for petak in petak_list]
                axes[1,2].hist(areas, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                axes[1,2].set_title('Petak Area Distribution', fontsize=10, fontweight='bold')
                axes[1,2].set_xlabel('Area (pixels)')
                axes[1,2].set_ylabel('Count')
            else:
                axes[1,2].text(0.5, 0.5, 'No petak detected', ha='center', va='center')
                axes[1,2].set_title('Petak Area Distribution', fontsize=10, fontweight='bold')
            
            # Aspect ratio distribution
            if petak_list:
                aspect_ratios = [petak['aspect_ratio'] for petak in petak_list]
                axes[1,3].hist(aspect_ratios, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[1,3].set_title('Petak Aspect Ratio Distribution', fontsize=10, fontweight='bold')
                axes[1,3].set_xlabel('Aspect Ratio')
                axes[1,3].set_ylabel('Count')
            else:
                axes[1,3].text(0.5, 0.5, 'No petak detected', ha='center', va='center')
                axes[1,3].set_title('Petak Aspect Ratio Distribution', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Petak visualization saved: {output_path}")
            
            # Show figure
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️  Could not display figure: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False
    
    def save_petak_report(self, petak_list, output_path):
        """Simpan laporan petak yang dioptimalkan"""
        try:
            report = {
                'total_petak': len(petak_list),
                'detection_method': 'petak_optimized_detection',
                'optimization_parameters': {
                    'min_area': 10,
                    'max_area': 50000,
                    'min_aspect_ratio': 0.2,
                    'max_aspect_ratio': 8.0,
                    'min_rectangularity': 0.15,
                    'min_vertices': 3,
                    'max_vertices': 12
                },
                'petak_details': []
            }
            
            for i, petak in enumerate(petak_list):
                x, y, w, h = petak['bbox']
                petak_info = {
                    'petak_id': i + 1,
                    'position': {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h),
                        'center_x': int(petak['center'][0]),
                        'center_y': int(petak['center'][1])
                    },
                    'properties': {
                        'area': float(petak['area']),
                        'aspect_ratio': float(petak['aspect_ratio']),
                        'rectangularity': float(petak['rectangularity']),
                        'vertices': int(petak['vertices'])
                    }
                }
                report['petak_details'].append(petak_info)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Petak report saved: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            return False

def main():
    """Main function untuk petak optimization"""
    optimizer = PetakOptimizer()
    
    print("🎯 PETAK OPTIMIZER")
    print("=" * 60)
    print("Khusus untuk mendeteksi petak-petak dengan parameter optimal")
    print("Fokus pada deteksi maksimal petak dengan ukuran bervariasi")
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
    
    # Optimize for petak
    optimized_image, gray, edges = optimizer.optimize_for_petak(image)
    
    # Detect petak with optimized parameters
    petak_list = optimizer.detect_petak_with_optimized_params(optimized_image, gray)
    
    # Advanced duplicate removal
    petak_list = optimizer.advanced_duplicate_removal(petak_list)
    
    # Generate report
    print(f"\n📊 Petak Optimization Results:")
    print(f"Total petak detected: {len(petak_list)}")
    
    if petak_list:
        print(f"\nPetak details:")
        print("-" * 90)
        print(f"{'ID':<3} {'Area':<8} {'Vertices':<8} {'Aspect Ratio':<12} {'Rectangularity':<12}")
        print("-" * 90)
        
        for i, petak in enumerate(petak_list):
            print(f"{i+1:<3} {int(petak['area']):<8} {petak['vertices']:<8} {petak['aspect_ratio']:.2f}        {petak['rectangularity']:.3f}")
    
    # Create visualization
    print(f"\n🎨 Creating petak visualization...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f'petak_optimized_detection_{base_name}.png'
    
    success = optimizer.create_petak_visualization(image, optimized_image, edges, petak_list, output_path)
    
    # Save detailed report
    report_path = f'petak_optimized_report_{base_name}.json'
    optimizer.save_petak_report(petak_list, report_path)
    
    if success:
        print(f"\n🎉 Petak optimization completed successfully!")
        print(f"📁 Files saved:")
        print(f"   - {output_path}")
        print(f"   - {report_path}")
        print(f"\n💡 Optimization Features:")
        print(f"   - Noise reduction dengan bilateral filter")
        print(f"   - Multiple edge detection techniques")
        print(f"   - Optimized parameters untuk petak")
        print(f"   - Advanced duplicate removal")
        print(f"   - Comprehensive petak analysis")
    else:
        print(f"\n⚠️  Optimization completed with warnings")

if __name__ == "__main__":
    main()

