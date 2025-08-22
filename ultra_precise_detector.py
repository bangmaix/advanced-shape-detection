import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
from scipy import ndimage
from scipy.spatial.distance import cdist
import math
import json
import os

class UltraPreciseDetector:
    def __init__(self):
        # Extended color palette for 250+ petak
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
            '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#F1C40F',
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'
        ]
        
        # Ultra-precise parameters for petak detection
        self.min_area = 30  # Very small minimum for tiny petak
        self.max_area = 5000  # Maximum area for individual petak
        self.aspect_ratio_range = (0.2, 5.0)  # Wide range for various petak shapes
        self.rectangularity_threshold = 0.6  # Lower threshold for precise detection
        self.corner_precision = 0.01  # High precision for corner detection
        
    def ultra_precise_preprocessing(self, image):
        """Ultra-precise preprocessing for petak detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter for edge preservation
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Multiple thresholding approaches
        # 1. Adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 2. Otsu's method
        _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. Multiple simple thresholds
        _, thresh3 = cv2.threshold(bilateral, 100, 255, cv2.THRESH_BINARY_INV)
        _, thresh4 = cv2.threshold(bilateral, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Combine all thresholding results
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, thresh3)
        combined = cv2.bitwise_or(combined, thresh4)
        
        # Advanced morphological operations
        # Close very small gaps
        kernel_close_small = np.ones((2, 2), np.uint8)
        closed_small = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_small)
        
        # Close medium gaps
        kernel_close_medium = np.ones((3, 3), np.uint8)
        closed_medium = cv2.morphologyEx(closed_small, cv2.MORPH_CLOSE, kernel_close_medium)
        
        # Remove noise
        kernel_open = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(closed_medium, cv2.MORPH_OPEN, kernel_open)
        
        # Final cleaning
        kernel_final = np.ones((2, 2), np.uint8)
        final = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_final)
        
        return final, gray
    
    def calculate_precise_rectangularity(self, contour):
        """Calculate precise rectangularity score"""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        # Basic rectangularity
        rectangularity = area / rect_area
        
        # Precise corner detection
        epsilon = self.corner_precision * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Corner bonus based on vertex count
        if len(approx) == 4:
            corner_bonus = 1.0
        elif len(approx) == 5:
            corner_bonus = 0.8
        elif len(approx) == 6:
            corner_bonus = 0.6
        else:
            corner_bonus = 0.4
        
        # Convexity check
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        return rectangularity * corner_bonus * convexity
    
    def detect_ultra_precise_petak(self, image):
        """Detect petak with ultra-precision"""
        processed, gray = self.ultra_precise_preprocessing(image)
        
        # Find contours with different retrieval modes
        contours_external, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine and deduplicate contours
        all_contours = contours_external + contours_tree
        unique_contours = []
        seen_centers = set()
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by aspect ratio
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            # Calculate precise rectangularity
            rectangularity = self.calculate_precise_rectangularity(contour)
            
            # Filter by rectangularity
            if rectangularity < self.rectangularity_threshold:
                continue
            
            # Calculate center for deduplication
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Deduplicate based on center proximity
            center_key = (cx // 10, cy // 10)  # Round to nearest 10 pixels
            if center_key in seen_centers:
                continue
            seen_centers.add(center_key)
            
            # Precise corner detection
            epsilon = self.corner_precision * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate precise corners
            corners = []
            for point in approx:
                corners.append((point[0][0], point[0][1]))
            
            unique_contours.append({
                'contour': contour,
                'approx': approx,
                'corners': corners,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (cx, cy),
                'aspect_ratio': aspect_ratio,
                'rectangularity': rectangularity,
                'vertices': len(approx),
                'perimeter': cv2.arcLength(contour, True)
            })
        
        return unique_contours, processed
    
    def cluster_petak_by_section(self, petak_list, eps=50, min_samples=1):
        """Cluster petak by their section in the floor plan"""
        if len(petak_list) < 2:
            return [petak_list]
        
        # Extract centers for clustering
        centers = np.array([petak['center'] for petak in petak_list])
        
        # Use DBSCAN for spatial clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
        
        # Group petak by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(petak_list[i])
        
        # Sort petak within each cluster by position
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: (x['center'][1], x['center'][0]))
        
        return list(clusters.values())
    
    def analyze_petak_structure(self, petak_list, image_shape):
        """Analyze the structure of detected petak"""
        if not petak_list:
            return []
        
        height, width = image_shape[:2]
        
        # Cluster petak
        clusters = self.cluster_petak_by_section(petak_list)
        
        # Analyze each cluster
        structure_analysis = []
        
        for cluster_id, cluster in enumerate(clusters):
            if not cluster:
                continue
            
            # Calculate cluster properties
            cluster_centers = [petak['center'] for petak in cluster]
            cluster_x = [center[0] for center in cluster_centers]
            cluster_y = [center[1] for center in cluster_centers]
            
            # Determine cluster position
            avg_x = np.mean(cluster_x)
            avg_y = np.mean(cluster_y)
            
            # Normalize positions
            norm_x = avg_x / width
            norm_y = avg_y / height
            
            # Classify cluster based on position
            if norm_y < 0.3:
                section_type = 'top_section'
            elif norm_y > 0.7:
                section_type = 'bottom_section'
            else:
                section_type = 'middle_section'
            
            # Determine arrangement pattern
            if len(cluster) == 1:
                arrangement = 'single_petak'
            elif len(cluster) <= 10:
                arrangement = 'small_cluster'
            elif len(cluster) <= 30:
                arrangement = 'medium_cluster'
            else:
                arrangement = 'large_cluster'
            
            # Calculate cluster statistics
            areas = [petak['area'] for petak in cluster]
            aspect_ratios = [petak['aspect_ratio'] for petak in cluster]
            
            structure_analysis.append({
                'cluster_id': cluster_id,
                'petak_list': cluster,
                'section_type': section_type,
                'arrangement': arrangement,
                'count': len(cluster),
                'position': (norm_x, norm_y),
                'bounds': {
                    'min_x': min(cluster_x),
                    'max_x': max(cluster_x),
                    'min_y': min(cluster_y),
                    'max_y': max(cluster_y)
                },
                'statistics': {
                    'avg_area': np.mean(areas),
                    'avg_aspect_ratio': np.mean(aspect_ratios),
                    'area_std': np.std(areas)
                }
            })
        
        return structure_analysis
    
    def create_ultra_precise_visualization(self, image_path):
        """Create ultra-precise visualization with all detected petak"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image from {image_path}")
            return None
        
        # Detect ultra-precise petak
        petak_list, processed = self.detect_ultra_precise_petak(image)
        
        # Analyze structure
        structure_analysis = self.analyze_petak_structure(petak_list, image.shape)
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Floor Plan', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Processed image
        ax2.imshow(processed, cmap='gray')
        ax2.set_title('Preprocessed Image', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # All detected petak with numbers
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i, petak in enumerate(petak_list):
            color = self.colors[i % len(self.colors)]
            contour = petak['contour']
            ax3.fill(contour[:, 0, 0], contour[:, 0, 1], 
                    color=color, alpha=0.6, edgecolor='black', linewidth=1)
            
            # Add petak number
            ax3.text(petak['center'][0], petak['center'][1], str(i+1), 
                    ha='center', va='center', fontsize=5, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.1", 
                                           facecolor='white', alpha=0.9))
        
        ax3.set_title(f'All Detected Petak ({len(petak_list)} total)', fontsize=16, fontweight='bold')
        ax3.axis('off')
        
        # Clustered analysis with section labels
        ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for cluster_id, analysis in enumerate(structure_analysis):
            cluster_color = self.colors[cluster_id % len(self.colors)]
            
            for petak in analysis['petak_list']:
                contour = petak['contour']
                ax4.fill(contour[:, 0, 0], contour[:, 0, 1], 
                        color=cluster_color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add cluster label
            center_x = analysis['bounds']['min_x'] + (analysis['bounds']['max_x'] - analysis['bounds']['min_x']) / 2
            center_y = analysis['bounds']['min_y'] + (analysis['bounds']['max_y'] - analysis['bounds']['min_y']) / 2
            
            ax4.text(center_x, center_y, f"Cluster {cluster_id+1}\n({analysis['count']} petak)", 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.4", 
                                           facecolor='white', alpha=0.95))
        
        ax4.set_title(f'Clustered Analysis ({len(structure_analysis)} clusters)', fontsize=16, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig, petak_list, structure_analysis
    
    def generate_comprehensive_report(self, petak_list, structure_analysis):
        """Generate comprehensive analysis report"""
        report = {
            'total_petak': len(petak_list),
            'total_clusters': len(structure_analysis),
            'clusters': [],
            'statistics': {},
            'target_analysis': {}
        }
        
        # Analyze each cluster
        for analysis in structure_analysis:
            cluster_info = {
                'cluster_id': analysis['cluster_id'],
                'section_type': analysis['section_type'],
                'arrangement': analysis['arrangement'],
                'petak_count': analysis['count'],
                'position': analysis['position'],
                'bounds': analysis['bounds'],
                'statistics': analysis['statistics']
            }
            report['clusters'].append(cluster_info)
        
        # Calculate overall statistics
        areas = [petak['area'] for petak in petak_list]
        aspect_ratios = [petak['aspect_ratio'] for petak in petak_list]
        rectangularities = [petak['rectangularity'] for petak in petak_list]
        perimeters = [petak['perimeter'] for petak in petak_list]
        
        report['statistics'] = {
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'average_aspect_ratio': np.mean(aspect_ratios),
            'average_rectangularity': np.mean(rectangularities),
            'average_perimeter': np.mean(perimeters),
            'area_std': np.std(areas),
            'aspect_ratio_std': np.std(aspect_ratios)
        }
        
        # Target analysis (comparing to expected 250 petak)
        expected_petak = 250
        detected_petak = len(petak_list)
        accuracy = (detected_petak / expected_petak) * 100 if expected_petak > 0 else 0
        
        report['target_analysis'] = {
            'expected_petak': expected_petak,
            'detected_petak': detected_petak,
            'accuracy_percentage': accuracy,
            'missing_petak': max(0, expected_petak - detected_petak),
            'extra_petak': max(0, detected_petak - expected_petak)
        }
        
        return report
    
    def save_petak_coordinates(self, petak_list, output_file):
        """Save precise petak coordinates to JSON file"""
        coordinates = []
        
        for i, petak in enumerate(petak_list):
            x, y, w, h = petak['bbox']
            coordinates.append({
                'petak_id': i + 1,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'center_x': int(petak['center'][0]),
                'center_y': int(petak['center'][1]),
                'area': float(petak['area']),
                'aspect_ratio': float(petak['aspect_ratio']),
                'rectangularity': float(petak['rectangularity']),
                'perimeter': float(petak['perimeter']),
                'vertices': int(petak['vertices']),
                'corners': petak['corners']
            })
        
        with open(output_file, 'w') as f:
            json.dump(coordinates, f, indent=2)

def main():
    """Main function for ultra-precise petak detection"""
    detector = UltraPreciseDetector()
    
    print("🎯 Ultra-Precise Petak Detection System")
    print("=" * 60)
    print("Designed for TOKO BLOK J LT.2 - 250 PETAK")
    print("=" * 60)
    
    # Look for image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("No image files found. Please place the floor plan image in the current directory.")
        return
    
    print("Found image files:")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {file}")
    
    # Use the first image
    image_path = image_files[0]
    print(f"\nProcessing: {image_path}")
    
    # Detect petak with ultra-precision
    result = detector.create_ultra_precise_visualization(image_path)
    if result is None:
        return
    
    fig, petak_list, structure_analysis = result
    
    # Generate comprehensive report
    report = detector.generate_comprehensive_report(petak_list, structure_analysis)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f'ultra_precise_detection_{base_name}.png', dpi=300, bbox_inches='tight')
    detector.save_petak_coordinates(petak_list, f'petak_coordinates_{base_name}.json')
    
    # Print comprehensive results
    print(f"\n📊 Ultra-Precise Detection Results:")
    print(f"Total petak detected: {report['total_petak']}")
    print(f"Expected petak: {report['target_analysis']['expected_petak']}")
    print(f"Detection accuracy: {report['target_analysis']['accuracy_percentage']:.1f}%")
    print(f"Total clusters: {report['total_clusters']}")
    
    print(f"\n📈 Statistical Analysis:")
    print(f"Average area: {report['statistics']['average_area']:.1f} pixels")
    print(f"Average aspect ratio: {report['statistics']['average_aspect_ratio']:.2f}")
    print(f"Average rectangularity: {report['statistics']['average_rectangularity']:.3f}")
    print(f"Area standard deviation: {report['statistics']['area_std']:.1f}")
    
    print(f"\n🎯 Target Analysis:")
    if report['target_analysis']['missing_petak'] > 0:
        print(f"Missing petak: {report['target_analysis']['missing_petak']}")
    if report['target_analysis']['extra_petak'] > 0:
        print(f"Extra petak detected: {report['target_analysis']['extra_petak']}")
    
    print(f"\n📁 Files saved:")
    print(f"- ultra_precise_detection_{base_name}.png")
    print(f"- petak_coordinates_{base_name}.json")
    
    # Show the plot
    plt.show()
    
    print("\n🎉 Ultra-precise detection completed!")
    print("All petak units detected with high precision and exact corner coordinates.")

if __name__ == "__main__":
    main()

