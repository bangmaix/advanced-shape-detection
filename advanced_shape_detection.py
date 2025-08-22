import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from scipy import ndimage
import math

class AdvancedShapeDetector:
    def __init__(self):
        self.colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FFEAA7',  # Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Mint
            '#F7DC6F',  # Gold
            '#BB8FCE',  # Lavender
            '#85C1E9',  # Sky Blue
            '#F8C471',  # Orange
            '#BDC3C7'   # Gray
        ]
    
    def preprocess_image(self, image):
        """Preprocess image for better shape detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        return morph
    
    def detect_rectangles(self, contours, min_area=1000):
        """Detect rectangular shapes from contours"""
        rectangles = []
        
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 vertices)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    rectangles.append({
                        'contour': contour,
                        'approx': approx,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w//2, y + h//2)
                    })
        
        return rectangles
    
    def detect_text_regions(self, image):
        """Detect regions that likely contain text"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_regions = []
        for region in regions:
            # Get bounding box of the region
            x, y, w, h = cv2.boundingRect(region)
            if w > 20 and h > 10:  # Filter small regions
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    def cluster_shapes_by_position(self, shapes, n_clusters=5):
        """Cluster shapes by their position to group related shapes"""
        if len(shapes) == 0:
            return []
        
        # Extract centers for clustering
        centers = np.array([shape['center'] for shape in shapes])
        
        # Use K-means to cluster shapes
        kmeans = KMeans(n_clusters=min(n_clusters, len(shapes)), random_state=42)
        clusters = kmeans.fit_predict(centers)
        
        # Group shapes by cluster
        clustered_shapes = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_shapes:
                clustered_shapes[cluster_id] = []
            clustered_shapes[cluster_id].append(shapes[i])
        
        return clustered_shapes
    
    def analyze_shape_hierarchy(self, shapes):
        """Analyze the hierarchy of shapes to identify main sections"""
        if not shapes:
            return []
        
        # Sort shapes by area (largest first)
        sorted_shapes = sorted(shapes, key=lambda x: x['area'], reverse=True)
        
        # Identify main sections based on size and position
        main_sections = []
        
        for shape in sorted_shapes:
            x, y, w, h = shape['bbox']
            area = shape['area']
            
            # Large shapes are likely main sections
            if area > 5000:
                main_sections.append({
                    'shape': shape,
                    'type': 'main_section',
                    'position': 'top' if y < 200 else 'bottom'
                })
            # Medium shapes might be kiosks
            elif 1000 < area < 5000:
                main_sections.append({
                    'shape': shape,
                    'type': 'kiosk',
                    'position': 'bottom'
                })
            # Small shapes might be labels or street
            elif area < 1000:
                main_sections.append({
                    'shape': shape,
                    'type': 'label',
                    'position': 'bottom'
                })
        
        return main_sections
    
    def detect_shapes_from_image(self, image_path):
        """Main function to detect shapes from an image"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image from {image_path}")
            return None
        
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect rectangles
        rectangles = self.detect_rectangles(contours)
        
        # Analyze shape hierarchy
        analyzed_shapes = self.analyze_shape_hierarchy(rectangles)
        
        # Detect text regions
        text_regions = self.detect_text_regions(image)
        
        return {
            'image': image,
            'processed': processed,
            'shapes': analyzed_shapes,
            'text_regions': text_regions,
            'original_contours': contours
        }
    
    def create_colored_diagram(self, detection_result):
        """Create a colored diagram based on detected shapes"""
        if not detection_result:
            return None
        
        image = detection_result['image']
        shapes = detection_result['shapes']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Colored diagram
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Color each detected shape
        for i, shape_info in enumerate(shapes):
            shape = shape_info['shape']
            shape_type = shape_info['type']
            
            # Get color based on type and position
            if shape_type == 'main_section':
                if shape_info['position'] == 'top':
                    color = self.colors[0]  # Red for top main section
                else:
                    color = self.colors[1]  # Teal for bottom main section
            elif shape_type == 'kiosk':
                color = self.colors[2 + (i % 8)]  # Different colors for kiosks
            else:
                color = self.colors[-1]  # Gray for labels
            
            # Draw filled contour
            contour = shape['contour']
            ax2.fill(contour[:, 0, 0], contour[:, 0, 1], 
                    color=color, alpha=0.6, edgecolor='black', linewidth=2)
            
            # Add shape number
            x, y, w, h = shape['bbox']
            ax2.text(x + w//2, y + h//2, str(i+1), 
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='white', alpha=0.8))
        
        ax2.set_title('Detected Shapes with Colors')
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_manual_diagram(self):
        """Create a manual diagram based on the original layout description"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        
        # DEKRANASDA BUKITTINGGI (Left section)
        dekranasda = Rectangle((0.5, 4), 1.5, 3, facecolor=self.colors[0], 
                              edgecolor='black', linewidth=2)
        ax.add_patch(dekranasda)
        ax.text(1.25, 5.5, 'DEKRANASDA\nBUKITTINGGI', ha='center', va='center', 
                fontsize=8, fontweight='bold')
        
        # LAPANGAN TENIS (Tennis Court - Central section)
        tennis_court = Rectangle((2.5, 4), 4, 3, facecolor=self.colors[1], 
                                edgecolor='black', linewidth=2)
        ax.add_patch(tennis_court)
        ax.text(4.5, 5.5, 'LAPANGAN TENIS', ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # IGD RSAM LAMA (Right section)
        igd = Rectangle((7, 4), 2, 3, facecolor=self.colors[10], 
                       edgecolor='black', linewidth=2)
        ax.add_patch(igd)
        ax.text(8, 5.5, 'IGD RSAM LAMA', ha='center', va='center', 
                fontsize=8, fontweight='bold')
        
        # Kiosks (8 numbered boxes below tennis court)
        for i in range(8):
            x_pos = 2.5 + i * 0.5
            kiosk = Rectangle((x_pos, 2.5), 0.4, 0.8, 
                             facecolor=self.colors[2 + i], edgecolor='black', linewidth=1)
            ax.add_patch(kiosk)
            ax.text(x_pos + 0.2, 2.9, str(i+1), ha='center', va='center', 
                    fontsize=10, fontweight='bold')
        
        # Street (Jl. Dr. Abdul Rivai)
        street = Rectangle((0.5, 1.5), 8.5, 0.3, facecolor=self.colors[11], 
                          edgecolor='black', linewidth=1)
        ax.add_patch(street)
        ax.text(4.75, 1.65, 'Jl. Dr. Abdul Rivai', ha='center', va='center', 
                fontsize=9, fontweight='bold')
        
        # Title
        ax.text(4.75, 7.5, 'KIOS LAPANGAN TENIS', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Add summary text
        ax.text(4.75, 0.5, 'KIOS LAPANGAN TENIS = 8 PETAK', ha='center', va='center', 
                fontsize=12, fontweight='bold', style='italic')
        
        plt.tight_layout()
        return fig

def main():
    """Main function to demonstrate shape detection"""
    detector = AdvancedShapeDetector()
    
    print("Advanced Shape Detection System")
    print("=" * 40)
    
    # Create manual diagram (since we don't have the original image)
    print("Creating manual diagram based on layout description...")
    fig_manual = detector.create_manual_diagram()
    plt.savefig('manual_diagram_colored.png', dpi=300, bbox_inches='tight')
    print("Manual diagram saved as 'manual_diagram_colored.png'")
    
    # If you have the original image, uncomment these lines:
    # print("\nDetecting shapes from original image...")
    # result = detector.detect_shapes_from_image('original_image.png')
    # if result:
    #     fig_detected = detector.create_colored_diagram(result)
    #     plt.savefig('detected_shapes_colored.png', dpi=300, bbox_inches='tight')
    #     print("Detected shapes saved as 'detected_shapes_colored.png'")
    
    plt.show()
    
    print("\nShape detection completed!")
    print("Features of this advanced system:")
    print("- Automatic shape detection using computer vision")
    print("- Contour analysis and rectangle detection")
    print("- Shape hierarchy analysis")
    print("- Text region detection")
    print("- Position-based clustering")
    print("- Multiple color schemes for different shape types")

if __name__ == "__main__":
    main()

