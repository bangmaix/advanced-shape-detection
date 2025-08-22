import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches
from sklearn.cluster import KMeans, DBSCAN
from scipy import ndimage
from scipy.spatial.distance import cdist
import math
import json

class IntelligentShapeDetector:
    def __init__(self):
        self.colors = [
            '#FF6B6B',  # Red - DEKRANASDA
            '#4ECDC4',  # Teal - Tennis Court
            '#45B7D1',  # Blue - Kiosk 1
            '#96CEB4',  # Green - Kiosk 2
            '#FFEAA7',  # Yellow - Kiosk 3
            '#DDA0DD',  # Plum - Kiosk 4
            '#98D8C8',  # Mint - Kiosk 5
            '#F7DC6F',  # Gold - Kiosk 6
            '#BB8FCE',  # Lavender - Kiosk 7
            '#85C1E9',  # Sky Blue - Kiosk 8
            '#F8C471',  # Orange - IGD
            '#BDC3C7'   # Gray - Street
        ]
        
        # Predefined shape patterns for tennis court layout
        self.shape_patterns = {
            'dekranasda': {'position': 'left', 'size': 'large', 'aspect_ratio': 0.5},
            'tennis_court': {'position': 'center', 'size': 'large', 'aspect_ratio': 1.33},
            'igd': {'position': 'right', 'size': 'large', 'aspect_ratio': 0.67},
            'kiosk': {'position': 'bottom', 'size': 'small', 'aspect_ratio': 0.5},
            'street': {'position': 'bottom', 'size': 'medium', 'aspect_ratio': 28.33}
        }
    
    def preprocess_image_advanced(self, image):
        """Advanced image preprocessing for better shape detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to preserve edges while reducing noise
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding with different methods
        thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        thresh2 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine both thresholding results
        combined = cv2.bitwise_or(thresh1, thresh2)
        
        # Morphological operations
        kernel_close = np.ones((5, 5), np.uint8)
        kernel_open = np.ones((3, 3), np.uint8)
        
        # Close gaps in shapes
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        # Remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        return opened, gray
    
    def detect_shapes_advanced(self, image):
        """Advanced shape detection using multiple techniques"""
        processed, gray = self.preprocess_image_advanced(image)
        
        # Find contours
        contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze each contour
        shapes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter very small contours
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate shape properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Determine shape type based on properties
            shape_type = self.classify_shape(approx, area, aspect_ratio, circularity)
            
            shapes.append({
                'contour': contour,
                'approx': approx,
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w//2, y + h//2),
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'type': shape_type,
                'vertices': len(approx)
            })
        
        return shapes, processed
    
    def classify_shape(self, approx, area, aspect_ratio, circularity):
        """Classify shape based on geometric properties"""
        vertices = len(approx)
        
        # Rectangle detection
        if vertices == 4:
            # Check if it's approximately rectangular
            if 0.8 < circularity < 1.2:
                return 'rectangle'
        
        # Square detection
        if vertices == 4 and 0.9 < aspect_ratio < 1.1:
            return 'square'
        
        # Line detection (very long rectangles)
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            return 'line'
        
        # Polygon detection
        if vertices > 4:
            return 'polygon'
        
        return 'unknown'
    
    def analyze_layout_pattern(self, shapes, image_shape):
        """Analyze the layout pattern to identify specific sections"""
        if not shapes:
            return []
        
        height, width = image_shape[:2]
        
        # Sort shapes by area
        sorted_shapes = sorted(shapes, key=lambda x: x['area'], reverse=True)
        
        # Identify main sections based on position and size
        layout_analysis = []
        
        for shape in sorted_shapes:
            x, y, w, h = shape['bbox']
            center_x, center_y = shape['center']
            
            # Normalize positions
            norm_x = center_x / width
            norm_y = center_y / height
            
            # Classify based on position and size
            if norm_y < 0.3:  # Top section
                if norm_x < 0.3:
                    section_type = 'dekranasda'
                elif norm_x > 0.7:
                    section_type = 'igd'
                else:
                    section_type = 'tennis_court'
            elif norm_y > 0.6:  # Bottom section
                if shape['aspect_ratio'] > 10:  # Very wide
                    section_type = 'street'
                else:
                    section_type = 'kiosk'
            else:
                section_type = 'unknown'
            
            layout_analysis.append({
                'shape': shape,
                'section_type': section_type,
                'normalized_position': (norm_x, norm_y)
            })
        
        return layout_analysis
    
    def cluster_kiosks(self, shapes):
        """Cluster small rectangular shapes that are likely kiosks"""
        kiosk_candidates = []
        
        for shape in shapes:
            if shape['type'] == 'rectangle' and 100 < shape['area'] < 2000:
                kiosk_candidates.append(shape)
        
        if len(kiosk_candidates) < 2:
            return kiosk_candidates
        
        # Extract centers for clustering
        centers = np.array([shape['center'] for shape in kiosk_candidates])
        
        # Use DBSCAN to cluster nearby kiosks
        clustering = DBSCAN(eps=50, min_samples=1).fit(centers)
        
        # Group by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(kiosk_candidates[i])
        
        # Sort kiosks within each cluster by x-position
        for cluster_id in clusters:
            clusters[cluster_id].sort(key=lambda x: x['center'][0])
        
        return clusters
    
    def create_intelligent_colored_diagram(self, image_path):
        """Create colored diagram using intelligent shape detection"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image from {image_path}")
            return None
        
        # Detect shapes
        shapes, processed = self.detect_shapes_advanced(image)
        
        # Analyze layout
        layout_analysis = self.analyze_layout_pattern(shapes, image.shape)
        
        # Cluster kiosks
        kiosk_clusters = self.cluster_kiosks(shapes)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Processed image
        ax2.imshow(processed, cmap='gray')
        ax2.set_title('Processed Image')
        ax2.axis('off')
        
        # Shape detection results
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i, shape in enumerate(shapes):
            contour = shape['contour']
            color = self.colors[i % len(self.colors)]
            ax3.fill(contour[:, 0, 0], contour[:, 0, 1], 
                    color=color, alpha=0.6, edgecolor='black', linewidth=1)
            ax3.text(shape['center'][0], shape['center'][1], str(i+1), 
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.2", 
                                           facecolor='white', alpha=0.8))
        ax3.set_title('Detected Shapes')
        ax3.axis('off')
        
        # Layout analysis
        ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for analysis in layout_analysis:
            shape = analysis['shape']
            section_type = analysis['section_type']
            
            # Assign color based on section type
            if section_type == 'dekranasda':
                color = self.colors[0]
            elif section_type == 'tennis_court':
                color = self.colors[1]
            elif section_type == 'igd':
                color = self.colors[10]
            elif section_type == 'kiosk':
                color = self.colors[2]
            elif section_type == 'street':
                color = self.colors[11]
            else:
                color = '#CCCCCC'
            
            contour = shape['contour']
            ax4.fill(contour[:, 0, 0], contour[:, 0, 1], 
                    color=color, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Add label
            ax4.text(shape['center'][0], shape['center'][1], section_type.upper(), 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='white', alpha=0.9))
        
        ax4.set_title('Layout Analysis')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig, layout_analysis
    
    def generate_shape_report(self, layout_analysis):
        """Generate a detailed report of detected shapes"""
        report = {
            'total_shapes': len(layout_analysis),
            'sections': {},
            'statistics': {}
        }
        
        # Count sections by type
        section_counts = {}
        for analysis in layout_analysis:
            section_type = analysis['section_type']
            if section_type not in section_counts:
                section_counts[section_type] = 0
            section_counts[section_type] += 1
        
        report['sections'] = section_counts
        
        # Calculate statistics
        areas = [analysis['shape']['area'] for analysis in layout_analysis]
        report['statistics'] = {
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'largest_shape': max(areas),
            'smallest_shape': min(areas)
        }
        
        return report

def main():
    """Main function to demonstrate intelligent shape detection"""
    detector = IntelligentShapeDetector()
    
    print("Intelligent Shape Detection System")
    print("=" * 50)
    
    # Example usage with a sample image (if available)
    # Uncomment the following lines if you have the original image:
    
    # print("Detecting shapes from original image...")
    # fig, layout_analysis = detector.create_intelligent_colored_diagram('original_image.png')
    # if fig:
    #     plt.savefig('intelligent_detection_results.png', dpi=300, bbox_inches='tight')
    #     print("Results saved as 'intelligent_detection_results.png'")
    #     
    #     # Generate report
    #     report = detector.generate_shape_report(layout_analysis)
    #     print("\nShape Detection Report:")
    #     print(json.dumps(report, indent=2))
    
    print("\nThis intelligent system provides:")
    print("✅ Advanced image preprocessing with multiple techniques")
    print("✅ Intelligent shape classification based on geometric properties")
    print("✅ Layout pattern analysis for tennis court diagrams")
    print("✅ Automatic kiosk clustering and numbering")
    print("✅ Section type identification (DEKRANASDA, Tennis Court, IGD, etc.)")
    print("✅ Detailed shape analysis and reporting")
    print("✅ Multiple visualization views (original, processed, detected, analyzed)")
    
    print("\nTo use with your original image:")
    print("1. Place your image file in the same directory")
    print("2. Update the image path in the main() function")
    print("3. Run the script to get intelligent shape detection results")

if __name__ == "__main__":
    main()

