#!/usr/bin/env python3
"""
Image Map Coordinate Extractor - Khusus untuk denah LOS EX AL IBAD
Mengekstrak koordinat image map dari 300 petak dengan presisi tinggi
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

class ImageMapExtractor:
    def __init__(self):
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
            '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#F1C40F'
        ]
        
    def extract_image_map_coordinates(self, image):
        """Ekstrak koordinat image map dengan teknik khusus"""
        print("🗺️  Extracting image map coordinates...")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Advanced preprocessing untuk denah
        # Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Step 2: Multiple edge detection untuk garis tipis
        edges1 = cv2.Canny(denoised, 15, 45)  # Sangat sensitif untuk garis tipis
        edges2 = cv2.Canny(denoised, 30, 90)  # Sensitif
        edges3 = cv2.Canny(denoised, 50, 150)  # Standard
        
        # Step 3: Sobel untuk deteksi garis halus
        sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel * 255 / np.max(sobel))
        _, sobel_thresh = cv2.threshold(sobel, 20, 255, cv2.THRESH_BINARY)
        
        # Step 4: Adaptive thresholding untuk berbagai intensitas
        adaptive1 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 7, 1)  # Lebih sensitif
        adaptive2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY_INV, 7, 1)
        
        # Step 5: Multiple simple thresholds
        _, thresh1 = cv2.threshold(denoised, 40, 255, cv2.THRESH_BINARY_INV)
        _, thresh2 = cv2.threshold(denoised, 80, 255, cv2.THRESH_BINARY_INV)
        _, thresh3 = cv2.threshold(denoised, 120, 255, cv2.THRESH_BINARY_INV)
        _, thresh4 = cv2.threshold(denoised, 160, 255, cv2.THRESH_BINARY_INV)
        
        # Step 6: Laplacian untuk deteksi garis halus
        laplacian = cv2.Laplacian(denoised, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, lap1 = cv2.threshold(laplacian, 2, 255, cv2.THRESH_BINARY)
        _, lap2 = cv2.threshold(laplacian, 5, 255, cv2.THRESH_BINARY)
        _, lap3 = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
        
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
        
        # Step 8: Morphological operations untuk menebalkan garis
        # Close gaps kecil
        kernel_close = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Dilate untuk menebalkan garis tipis
        kernel_dilate = np.ones((2, 2), np.uint8)
        combined = cv2.dilate(combined, kernel_dilate, iterations=1)
        
        # Close lagi untuk menghubungkan garis
        kernel_close_medium = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_medium)
        
        # Final dilation
        kernel_final = np.ones((2, 2), np.uint8)
        enhanced = cv2.dilate(combined, kernel_final, iterations=1)
        
        return enhanced, gray, edges2
    
    def detect_all_petak_coordinates(self, enhanced_image, original_gray):
        """Deteksi semua koordinat petak dengan parameter yang sangat sensitif"""
        print("🎯 Detecting all petak coordinates...")
        
        # Find contours dengan multiple retrieval modes
        contours_external, _ = cv2.findContours(enhanced_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, _ = cv2.findContours(enhanced_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine contours
        all_contours = contours_external + contours_tree
        print(f"   Found {len(all_contours)} total contours")
        
        # Parameter yang sangat sensitif untuk menangkap semua petak
        petak_coordinates = []
        min_area = 5  # Sangat kecil untuk petak kecil
        max_area = 100000  # Sangat besar untuk petak besar
        
        for i, contour in enumerate(all_contours):
            area = cv2.contourArea(contour)
            
            # Filter berdasarkan area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter berdasarkan aspect ratio (sangat longgar)
            if aspect_ratio < 0.1 or aspect_ratio > 15.0:
                continue
            
            # Calculate rectangularity
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Filter berdasarkan rectangularity (sangat longgar)
            if rectangularity < 0.1:
                continue
            
            # Approximate contour
            epsilon = 0.05 * cv2.arcLength(contour, True)  # Sangat longgar
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Filter berdasarkan vertices (2-15 untuk menangkap berbagai bentuk)
            if len(approx) < 2 or len(approx) > 15:
                continue
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Get corner coordinates
            corners = []
            for point in approx:
                corners.append([int(point[0][0]), int(point[0][1])])
            
            petak_coordinates.append({
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (cx, cy),
                'corners': corners,
                'aspect_ratio': aspect_ratio,
                'rectangularity': rectangularity,
                'vertices': len(approx)
            })
        
        print(f"   Filtered to {len(petak_coordinates)} potential petak")
        return petak_coordinates
    
    def smart_coordinate_filtering(self, petak_coordinates, distance_threshold=10):
        """Smart filtering untuk menghapus duplikat dan mengoptimalkan koordinat"""
        if len(petak_coordinates) <= 1:
            return petak_coordinates
        
        # Sort by area (largest first)
        sorted_petak = sorted(petak_coordinates, key=lambda x: x['area'], reverse=True)
        
        filtered = []
        used_positions = set()
        
        for petak in sorted_petak:
            # Create position key dengan granularity yang sangat halus
            pos_key = (petak['center'][0] // 5, petak['center'][1] // 5)
            
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
        
        print(f"   Smart filtering: {len(petak_coordinates)} -> {len(filtered)} petak")
        return filtered
    
    def create_coordinate_visualization(self, image, enhanced_image, edges, petak_coordinates, output_path):
        """Buat visualisasi koordinat image map"""
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
            
            # Enhanced lines
            axes[0,2].imshow(enhanced_image, cmap='gray')
            axes[0,2].set_title('Enhanced Lines', fontsize=12, fontweight='bold')
            axes[0,2].axis('off')
            
            # Detected coordinates overlay
            if len(image.shape) == 3:
                axes[1,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                axes[1,0].imshow(image, cmap='gray')
            
            # Draw petak with coordinates
            for i, petak in enumerate(petak_coordinates):
                color = self.colors[i % len(self.colors)]
                contour = petak['contour']
                
                # Draw filled contour
                axes[1,0].fill(contour[:, 0, 0], contour[:, 0, 1], 
                             color=color, alpha=0.6, edgecolor='black', linewidth=1)
                
                # Add coordinate number
                axes[1,0].text(petak['center'][0], petak['center'][1], str(i+1), 
                             ha='center', va='center', fontsize=6, fontweight='bold',
                             color='black', bbox=dict(boxstyle="round,pad=0.1", 
                                                     facecolor='white', alpha=0.9))
                
                # Draw corner points
                for j, corner in enumerate(petak['corners']):
                    axes[1,0].plot(corner[0], corner[1], 'ro', markersize=3)
            
            axes[1,0].set_title(f'Image Map Coordinates ({len(petak_coordinates)} total)', fontsize=12, fontweight='bold')
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
            axes[1,1].set_title('Enhanced Lines Overlay', fontsize=12, fontweight='bold')
            axes[1,1].axis('off')
            
            # Coordinate distribution
            if petak_coordinates:
                centers_x = [petak['center'][0] for petak in petak_coordinates]
                centers_y = [petak['center'][1] for petak in petak_coordinates]
                
                axes[1,2].scatter(centers_x, centers_y, alpha=0.6, s=20, c='blue')
                axes[1,2].set_title('Coordinate Distribution', fontsize=12, fontweight='bold')
                axes[1,2].set_xlabel('X Coordinate')
                axes[1,2].set_ylabel('Y Coordinate')
                axes[1,2].grid(True, alpha=0.3)
            else:
                axes[1,2].text(0.5, 0.5, 'No coordinates detected', ha='center', va='center')
                axes[1,2].set_title('Coordinate Distribution', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Coordinate visualization saved: {output_path}")
            
            # Show figure
            try:
                plt.show()
            except Exception as e:
                print(f"⚠️  Could not display figure: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False
    
    def save_image_map_coordinates(self, petak_coordinates, output_path):
        """Simpan koordinat image map dalam format HTML dan JSON"""
        try:
            # HTML Image Map format
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Image Map Coordinates - LOS EX AL IBAD</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .info {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .coordinates {{ background: #fff; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        .coordinate-item {{ margin: 5px 0; padding: 5px; background: #f9f9f9; border-left: 3px solid #3498db; }}
        .area-info {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>🗺️ Image Map Coordinates - LOS EX AL IBAD</h1>
    
    <div class="info">
        <h3>📊 Summary</h3>
        <p><strong>Total Petak:</strong> {len(petak_coordinates)}</p>
        <p><strong>Expected:</strong> 300 petak</p>
        <p><strong>Detection Rate:</strong> {(len(petak_coordinates)/300)*100:.1f}%</p>
    </div>
    
    <div class="coordinates">
        <h3>📍 Coordinate Details</h3>
        <p>Format: area="coordinate_id" coords="x1,y1,x2,y2,x3,y3,x4,y4" href="#"</p>
        
        <div class="coordinate-list">
"""
            
            # JSON format
            json_data = {
                'image_map_info': {
                    'title': 'LOS EX AL IBAD',
                    'total_petak': len(petak_coordinates),
                    'expected_petak': 300,
                    'detection_rate': f"{(len(petak_coordinates)/300)*100:.1f}%"
                },
                'coordinates': []
            }
            
            for i, petak in enumerate(petak_coordinates):
                # Flatten corner coordinates for HTML image map
                coords_str = ""
                for corner in petak['corners']:
                    coords_str += f"{corner[0]},{corner[1]},"
                coords_str = coords_str.rstrip(',')
                
                # Add to HTML
                html_content += f"""
            <div class="coordinate-item">
                <strong>Petak {i+1}:</strong> area="{i+1}" coords="{coords_str}" href="#"
                <div class="area-info">
                    Center: ({petak['center'][0]}, {petak['center'][1]}) | 
                    Area: {int(petak['area'])} pixels | 
                    Vertices: {petak['vertices']}
                </div>
            </div>"""
                
                # Add to JSON
                json_data['coordinates'].append({
                    'petak_id': i + 1,
                    'center': {
                        'x': int(petak['center'][0]),
                        'y': int(petak['center'][1])
                    },
                    'corners': petak['corners'],
                    'bbox': {
                        'x': int(petak['bbox'][0]),
                        'y': int(petak['bbox'][1]),
                        'width': int(petak['bbox'][2]),
                        'height': int(petak['bbox'][3])
                    },
                    'properties': {
                        'area': float(petak['area']),
                        'aspect_ratio': float(petak['aspect_ratio']),
                        'rectangularity': float(petak['rectangularity']),
                        'vertices': int(petak['vertices'])
                    }
                })
            
            html_content += """
        </div>
    </div>
    
    <div class="info">
        <h3>💡 Usage Instructions</h3>
        <p>1. Copy the coordinate data above</p>
        <p>2. Use in HTML image map format</p>
        <p>3. Each coordinate represents a clickable area</p>
        <p>4. Modify href attributes as needed</p>
    </div>
</body>
</html>"""
            
            # Save HTML file
            html_path = output_path.replace('.json', '.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Image map coordinates saved:")
            print(f"   - HTML: {html_path}")
            print(f"   - JSON: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save coordinates: {e}")
            return False

def main():
    """Main function untuk image map coordinate extraction"""
    extractor = ImageMapExtractor()
    
    print("🗺️  IMAGE MAP COORDINATE EXTRACTOR")
    print("=" * 60)
    print("Khusus untuk denah LOS EX AL IBAD dengan 300 petak")
    print("Mengekstrak koordinat image map dengan presisi tinggi")
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
    
    # Extract coordinates
    enhanced_image, gray, edges = extractor.extract_image_map_coordinates(image)
    
    # Detect all petak coordinates
    petak_coordinates = extractor.detect_all_petak_coordinates(enhanced_image, gray)
    
    # Smart filtering
    petak_coordinates = extractor.smart_coordinate_filtering(petak_coordinates)
    
    # Generate report
    print(f"\n📊 Image Map Coordinate Results:")
    print(f"Total coordinates extracted: {len(petak_coordinates)}")
    print(f"Expected petak: 300")
    print(f"Detection rate: {(len(petak_coordinates)/300)*100:.1f}%")
    
    if petak_coordinates:
        print(f"\nCoordinate details:")
        print("-" * 100)
        print(f"{'ID':<3} {'Center':<15} {'Corners':<8} {'Area':<8} {'Vertices':<8}")
        print("-" * 100)
        
        for i, petak in enumerate(petak_coordinates):
            center_str = f"({petak['center'][0]},{petak['center'][1]})"
            print(f"{i+1:<3} {center_str:<15} {len(petak['corners']):<8} {int(petak['area']):<8} {petak['vertices']:<8}")
    
    # Create visualization
    print(f"\n🎨 Creating coordinate visualization...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f'image_map_coordinates_{base_name}.png'
    
    success = extractor.create_coordinate_visualization(image, enhanced_image, edges, petak_coordinates, output_path)
    
    # Save coordinates
    json_path = f'image_map_coordinates_{base_name}.json'
    extractor.save_image_map_coordinates(petak_coordinates, json_path)
    
    if success:
        print(f"\n🎉 Image map coordinate extraction completed successfully!")
        print(f"📁 Files saved:")
        print(f"   - {output_path}")
        print(f"   - {json_path}")
        print(f"   - {json_path.replace('.json', '.html')}")
        print(f"\n💡 Features:")
        print(f"   - Extracted {len(petak_coordinates)} coordinates")
        print(f"   - HTML image map format ready")
        print(f"   - JSON format for programmatic use")
        print(f"   - Visual coordinate overlay")
        print(f"   - Detection rate: {(len(petak_coordinates)/300)*100:.1f}%")
    else:
        print(f"\n⚠️  Extraction completed with warnings")

if __name__ == "__main__":
    main()

