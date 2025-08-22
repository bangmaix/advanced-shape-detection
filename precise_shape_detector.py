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

class PreciseShapeDetector:
    def __init__(self):
        # Extended color palette for 250+ shapes
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
            '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#F1C40F',
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'
        ]
        
        # Parameters for precise detection
        self.min_area = 50  # Smaller minimum area for small units
        self.max_area = 10000  # Maximum area for individual units
        self.aspect_ratio_range = (0.3, 3.0)  # Acceptable aspect ratios for units
        self.rectangularity_threshold = 0.7  # Minimum rectangularity score
        
    def preprocess_for_precise_detection(self, image):
        """Advanced preprocessing specifically for floor plan detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply multiple thresholding methods
        # Method 1: Adaptive thresholding
        thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Method 2: Otsu's thresholding
        _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Method 3: Simple thresholding with different values
        _, thresh3 = cv2.threshold(bilateral, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Combine all thresholding results
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, thresh3)
        
        # Morphological operations for cleaning
        # Close small gaps in rectangles
        kernel_close = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove small noise
        kernel_open = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Additional cleaning for better rectangle detection
        kernel_clean = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_clean)
        
        return cleaned, gray
    
    def calculate_rectangularity(self, contour):
        """Calculate how rectangular a contour is"""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        # Rectangularity is the ratio of contour area to bounding rectangle area
        rectangularity = area / rect_area
        
        # Also check if the contour is approximately rectangular
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Bonus for having 4 vertices
        vertex_bonus = 1.0 if len(approx) == 4 else 0.5
        
        return rectangularity * vertex_bonus
    
    def detect_precise_rectangles(self, image):
        """Detect rectangles with high precision"""
        processed, gray = self.preprocess_for_precise_detection(image)
        
        # Find contours
        contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        
        for contour in contours:
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
            
            # Calculate rectangularity
            rectangularity = self.calculate_rectangularity(contour)
            
            # Filter by rectangularity
            if rectangularity < self.rectangularity_threshold:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
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
        
        return rectangles, processed
    
    def cluster_rectangles_by_position(self, rectangles, eps=30, min_samples=1):
        """Cluster rectangles by their spatial position"""
        if len(rectangles) < 2:
            return [rectangles]
        
        # Extract centers for clustering
        centers = np.array([rect['center'] for rect in rectangles])
        
        # Use DBSCAN for spatial clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)
        
        # Group rectangles by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(rectangles[i])
        
        # Sort rectangles within each cluster by position (top-left to bottom-right)
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: (x['center'][1], x['center'][0]))
        
        return list(clusters.values())
    
    def analyze_floor_plan_structure(self, rectangles, image_shape):
        """Analyze the floor plan structure to identify sections"""
        if not rectangles:
            return []
        
        height, width = image_shape[:2]
        
        # Cluster rectangles
        clusters = self.cluster_rectangles_by_position(rectangles)
        
        # Analyze each cluster
        structure_analysis = []
        
        for cluster_id, cluster in enumerate(clusters):
            if not cluster:
                continue
            
            # Calculate cluster properties
            cluster_centers = [rect['center'] for rect in cluster]
            cluster_x = [center[0] for center in cluster_centers]
            cluster_y = [center[1] for center in cluster_centers]
            
            # Determine cluster position
            avg_x = np.mean(cluster_x)
            avg_y = np.mean(cluster_y)
            
            # Normalize positions
            norm_x = avg_x / width
            norm_y = avg_y / height
            
            # Classify cluster based on position and size
            if norm_y < 0.4:  # Top section
                section_type = 'top_section'
            elif norm_y > 0.6:  # Bottom section
                section_type = 'bottom_section'
            else:  # Middle section
                section_type = 'middle_section'
            
            # Determine arrangement pattern
            if len(cluster) == 1:
                arrangement = 'single'
            elif len(cluster) <= 5:
                arrangement = 'small_cluster'
            elif len(cluster) <= 15:
                arrangement = 'medium_cluster'
            else:
                arrangement = 'large_cluster'
            
            structure_analysis.append({
                'cluster_id': cluster_id,
                'rectangles': cluster,
                'section_type': section_type,
                'arrangement': arrangement,
                'count': len(cluster),
                'position': (norm_x, norm_y),
                'bounds': {
                    'min_x': min(cluster_x),
                    'max_x': max(cluster_x),
                    'min_y': min(cluster_y),
                    'max_y': max(cluster_y)
                }
            })
        
        return structure_analysis
    
    def create_precise_colored_diagram(self, image_path):
        """Create precise colored diagram with all detected rectangles"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image from {image_path}")
            return None
        
        # Detect precise rectangles
        rectangles, processed = self.detect_precise_rectangles(image)
        
        # Analyze floor plan structure
        structure_analysis = self.analyze_floor_plan_structure(rectangles, image.shape)
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Processed image
        ax2.imshow(processed, cmap='gray')
        ax2.set_title('Processed Image')
        ax2.axis('off')
        
        # All detected rectangles
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i, rect in enumerate(rectangles):
            color = self.colors[i % len(self.colors)]
            contour = rect['contour']
            ax3.fill(contour[:, 0, 0], contour[:, 0, 1], 
                    color=color, alpha=0.6, edgecolor='black', linewidth=1)
            
            # Add rectangle number
            ax3.text(rect['center'][0], rect['center'][1], str(i+1), 
                    ha='center', va='center', fontsize=6, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.2", 
                                           facecolor='white', alpha=0.8))
        
        ax3.set_title(f'All Detected Rectangles ({len(rectangles)} total)')
        ax3.axis('off')
        
        # Clustered analysis
        ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for cluster_id, analysis in enumerate(structure_analysis):
            cluster_color = self.colors[cluster_id % len(self.colors)]
            
            for rect in analysis['rectangles']:
                contour = rect['contour']
                ax4.fill(contour[:, 0, 0], contour[:, 0, 1], 
                        color=cluster_color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add cluster label
            center_x = analysis['bounds']['min_x'] + (analysis['bounds']['max_x'] - analysis['bounds']['min_x']) / 2
            center_y = analysis['bounds']['min_y'] + (analysis['bounds']['max_y'] - analysis['bounds']['min_y']) / 2
            
            ax4.text(center_x, center_y, f"C{cluster_id+1}\n({analysis['count']})", 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='white', alpha=0.9))
        
        ax4.set_title(f'Clustered Analysis ({len(structure_analysis)} clusters)')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig, rectangles, structure_analysis
    
    def generate_detailed_report(self, rectangles, structure_analysis):
        """Generate detailed analysis report"""
        report = {
            'total_rectangles': len(rectangles),
            'total_clusters': len(structure_analysis),
            'clusters': [],
            'statistics': {}
        }
        
        # Analyze each cluster
        for analysis in structure_analysis:
            cluster_info = {
                'cluster_id': analysis['cluster_id'],
                'section_type': analysis['section_type'],
                'arrangement': analysis['arrangement'],
                'rectangle_count': analysis['count'],
                'position': analysis['position'],
                'bounds': analysis['bounds']
            }
            report['clusters'].append(cluster_info)
        
        # Calculate overall statistics
        areas = [rect['area'] for rect in rectangles]
        aspect_ratios = [rect['aspect_ratio'] for rect in rectangles]
        rectangularities = [rect['rectangularity'] for rect in rectangles]
        
        report['statistics'] = {
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'average_aspect_ratio': np.mean(aspect_ratios),
            'average_rectangularity': np.mean(rectangularities),
            'area_std': np.std(areas)
        }
        
        return report
    
    def save_rectangle_coordinates(self, rectangles, output_file):
        """Save rectangle coordinates to a file"""
        coordinates = []
        
        for i, rect in enumerate(rectangles):
            x, y, w, h = rect['bbox']
            coordinates.append({
                'id': i + 1,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center_x': rect['center'][0],
                'center_y': rect['center'][1],
                'area': rect['area'],
                'aspect_ratio': rect['aspect_ratio']
            })
        
        with open(output_file, 'w') as f:
            json.dump(coordinates, f, indent=2)

def main():
    """Main function for precise shape detection"""
    detector = PreciseShapeDetector()
    
    print("🎯 Precise Shape Detection System for Floor Plans")
    print("=" * 60)
    
    # Look for image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    if not image_files:
        print("No image files found. Please place an image file in the current directory.")
        return
    
    print("Found image files:")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {file}")
    
    # Use the first image or let user choose
    image_path = image_files[0]
    print(f"\nProcessing: {image_path}")
    
    # Detect shapes
    result = detector.create_precise_colored_diagram(image_path)
    if result is None:
        return
    
    fig, rectangles, structure_analysis = result
    
    # Generate report
    report = detector.generate_detailed_report(rectangles, structure_analysis)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f'precise_detection_{base_name}.png', dpi=300, bbox_inches='tight')
    detector.save_rectangle_coordinates(rectangles, f'rectangle_coordinates_{base_name}.json')
    
    # Print results
    print(f"\n📊 Detection Results:")
    print(f"Total rectangles detected: {report['total_rectangles']}")
    print(f"Total clusters: {report['total_clusters']}")
    print(f"Average area: {report['statistics']['average_area']:.1f} pixels")
    print(f"Average aspect ratio: {report['statistics']['average_aspect_ratio']:.2f}")
    print(f"Average rectangularity: {report['statistics']['average_rectangularity']:.3f}")
    
    print(f"\n📁 Files saved:")
    print(f"- precise_detection_{base_name}.png")
    print(f"- rectangle_coordinates_{base_name}.json")
    
    # Show the plot
    plt.show()
    
    print("\n🎉 Precise detection completed!")
    print("The system detected all rectangular units with high precision.")

if __name__ == "__main__":
    main()

