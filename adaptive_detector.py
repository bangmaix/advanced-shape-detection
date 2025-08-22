import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class AdaptiveDetector:
    def __init__(self):
        # Extended color palette
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
            '#F8C471', '#BDC3C7', '#E74C3C', '#3498DB', '#2ECC71',
            '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
            '#16A085', '#27AE60', '#2980B9', '#8E44AD', '#F1C40F',
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6'
        ]
        
    def analyze_image_content(self, image):
        """Analyze image content to determine optimal parameters"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Detect edges to understand structure
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Analyze histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / np.sum(hist)
        
        # Determine if image is mostly dark or light
        dark_pixels = np.sum(gray < 128)
        light_pixels = np.sum(gray > 128)
        is_dark = dark_pixels > light_pixels
        
        # Estimate typical shape size
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 10]
        
        if areas:
            median_area = np.median(areas)
            mean_area = np.mean(areas)
        else:
            median_area = 100
            mean_area = 100
        
        return {
            'image_size': (width, height),
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'edge_density': edge_density,
            'is_dark': is_dark,
            'median_area': median_area,
            'mean_area': mean_area,
            'total_pixels': height * width
        }
    
    def determine_optimal_parameters(self, analysis):
        """Determine optimal detection parameters based on image analysis"""
        width, height = analysis['image_size']
        median_area = analysis['median_area']
        edge_density = analysis['edge_density']
        is_dark = analysis['is_dark']
        
        # Adaptive minimum area (0.1% to 1% of median area)
        min_area = max(10, int(median_area * 0.1))
        
        # Adaptive maximum area (10x median area, but not too large)
        max_area = min(int(median_area * 10), int(width * height * 0.1))
        
        # Adaptive aspect ratio range
        if edge_density > 0.1:  # High edge density = complex shapes
            aspect_ratio_range = (0.1, 10.0)
        else:
            aspect_ratio_range = (0.3, 3.0)
        
        # Adaptive rectangularity threshold
        if is_dark:
            rectangularity_threshold = 0.5  # Lower for dark images
        else:
            rectangularity_threshold = 0.6
        
        # Adaptive morphological kernel size
        if width > 1000 or height > 1000:  # Large image
            kernel_size = 3
        else:
            kernel_size = 2
        
        return {
            'min_area': min_area,
            'max_area': max_area,
            'aspect_ratio_range': aspect_ratio_range,
            'rectangularity_threshold': rectangularity_threshold,
            'kernel_size': kernel_size
        }
    
    def adaptive_preprocessing(self, image, params):
        """Adaptive preprocessing based on image characteristics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Multiple thresholding approaches
        thresh1 = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        _, thresh2 = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Adaptive simple thresholds
        mean_val = np.mean(bilateral)
        _, thresh3 = cv2.threshold(bilateral, int(mean_val * 0.7), 255, cv2.THRESH_BINARY_INV)
        _, thresh4 = cv2.threshold(bilateral, int(mean_val * 0.9), 255, cv2.THRESH_BINARY_INV)
        
        # Combine thresholds
        combined = cv2.bitwise_or(thresh1, thresh2)
        combined = cv2.bitwise_or(combined, thresh3)
        combined = cv2.bitwise_or(combined, thresh4)
        
        # Adaptive morphological operations
        kernel_size = params['kernel_size']
        kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
        kernel_open = np.ones((kernel_size-1, kernel_size-1), np.uint8)
        
        # Close gaps
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Final cleaning
        final = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_open)
        
        return final, gray
    
    def calculate_rectangularity(self, contour, params):
        """Calculate rectangularity score with adaptive parameters"""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        
        # Basic rectangularity
        rectangularity = area / rect_area
        
        # Corner detection
        epsilon = 0.02 * cv2.arcLength(contour, True)
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
    
    def detect_shapes_adaptively(self, image):
        """Detect shapes with adaptive parameters"""
        # Analyze image content
        analysis = self.analyze_image_content(image)
        
        # Determine optimal parameters
        params = self.determine_optimal_parameters(analysis)
        
        # Preprocess image
        processed, gray = self.adaptive_preprocessing(image, params)
        
        # Find contours with different retrieval modes
        contours_external, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_tree, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine contours
        all_contours = contours_external + contours_tree
        unique_shapes = []
        seen_centers = set()
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < params['min_area'] or area > params['max_area']:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by aspect ratio
            min_ratio, max_ratio = params['aspect_ratio_range']
            if not (min_ratio <= aspect_ratio <= max_ratio):
                continue
            
            # Calculate rectangularity
            rectangularity = self.calculate_rectangularity(contour, params)
            
            # Filter by rectangularity
            if rectangularity < params['rectangularity_threshold']:
                continue
            
            # Calculate center for deduplication
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
            
            # Deduplicate based on center proximity
            center_key = (cx // 10, cy // 10)
            if center_key in seen_centers:
                continue
            seen_centers.add(center_key)
            
            # Corner detection
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calculate corners
            corners = []
            for point in approx:
                corners.append((point[0][0], point[0][1]))
            
            unique_shapes.append({
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
        
        return unique_shapes, processed, analysis, params
    
    def simple_clustering(self, shapes, distance_threshold=50):
        """Simple clustering based on distance"""
        if len(shapes) < 2:
            return [shapes]
        
        clusters = []
        used = set()
        
        for i, shape in enumerate(shapes):
            if i in used:
                continue
            
            # Start new cluster
            cluster = [shapes[i]]
            used.add(i)
            
            # Find nearby shapes
            for j, other_shape in enumerate(shapes):
                if j in used:
                    continue
                
                # Calculate distance between centers
                dist = np.sqrt((shape['center'][0] - other_shape['center'][0])**2 + 
                             (shape['center'][1] - other_shape['center'][1])**2)
                
                if dist <= distance_threshold:
                    cluster.append(other_shape)
                    used.add(j)
            
            # Sort cluster by position
            cluster.sort(key=lambda x: (x['center'][1], x['center'][0]))
            clusters.append(cluster)
        
        return clusters
    
    def create_adaptive_visualization(self, image_path):
        """Create adaptive visualization with all detected shapes"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image from {image_path}")
            return None
        
        # Detect shapes adaptively
        shapes, processed, analysis, params = self.detect_shapes_adaptively(image)
        
        # Cluster shapes
        clusters = self.simple_clustering(shapes)
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 18))
        
        # Original image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Processed image
        ax2.imshow(processed, cmap='gray')
        ax2.set_title('Adaptive Preprocessing', fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        # All detected shapes with numbers
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for i, shape in enumerate(shapes):
            color = self.colors[i % len(self.colors)]
            contour = shape['contour']
            ax3.fill(contour[:, 0, 0], contour[:, 0, 1], 
                    color=color, alpha=0.6, edgecolor='black', linewidth=1)
            
            # Add shape number
            ax3.text(shape['center'][0], shape['center'][1], str(i+1), 
                    ha='center', va='center', fontsize=6, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.1", 
                                           facecolor='white', alpha=0.9))
        
        ax3.set_title(f'All Detected Shapes ({len(shapes)} total)', fontsize=16, fontweight='bold')
        ax3.axis('off')
        
        # Clustered analysis
        ax4.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for cluster_id, cluster in enumerate(clusters):
            cluster_color = self.colors[cluster_id % len(self.colors)]
            
            for shape in cluster:
                contour = shape['contour']
                ax4.fill(contour[:, 0, 0], contour[:, 0, 1], 
                        color=cluster_color, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add cluster label
            centers = [shape['center'] for shape in cluster]
            center_x = np.mean([c[0] for c in centers])
            center_y = np.mean([c[1] for c in centers])
            
            ax4.text(center_x, center_y, f"Cluster {cluster_id+1}\n({len(cluster)} shapes)", 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='black', bbox=dict(boxstyle="round,pad=0.3", 
                                           facecolor='white', alpha=0.95))
        
        ax4.set_title(f'Clustered Analysis ({len(clusters)} clusters)', fontsize=16, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        return fig, shapes, clusters, analysis, params
    
    def generate_adaptive_report(self, shapes, clusters, analysis, params):
        """Generate comprehensive adaptive analysis report"""
        report = {
            'total_shapes': len(shapes),
            'total_clusters': len(clusters),
            'image_analysis': analysis,
            'detection_params': params,
            'clusters': [],
            'statistics': {}
        }
        
        # Analyze each cluster
        for cluster_id, cluster in enumerate(clusters):
            cluster_info = {
                'cluster_id': cluster_id,
                'shape_count': len(cluster),
                'shapes': []
            }
            
            for shape in cluster:
                shape_info = {
                    'shape_id': len(cluster_info['shapes']) + 1,
                    'area': float(shape['area']),
                    'aspect_ratio': float(shape['aspect_ratio']),
                    'rectangularity': float(shape['rectangularity']),
                    'center': [int(shape['center'][0]), int(shape['center'][1])],
                    'bbox': [int(x) for x in shape['bbox']]
                }
                cluster_info['shapes'].append(shape_info)
            
            report['clusters'].append(cluster_info)
        
        # Calculate overall statistics
        areas = [shape['area'] for shape in shapes]
        aspect_ratios = [shape['aspect_ratio'] for shape in shapes]
        rectangularities = [shape['rectangularity'] for shape in shapes]
        
        report['statistics'] = {
            'total_area': sum(areas),
            'average_area': np.mean(areas),
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'average_aspect_ratio': np.mean(aspect_ratios),
            'average_rectangularity': np.mean(rectangularities),
            'area_std': np.std(areas) if areas else 0
        }
        
        return report
    
    def save_shape_coordinates(self, shapes, output_file):
        """Save shape coordinates to JSON file"""
        coordinates = []
        
        for i, shape in enumerate(shapes):
            x, y, w, h = shape['bbox']
            coordinates.append({
                'shape_id': i + 1,
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'center_x': int(shape['center'][0]),
                'center_y': int(shape['center'][1]),
                'area': float(shape['area']),
                'aspect_ratio': float(shape['aspect_ratio']),
                'rectangularity': float(shape['rectangularity']),
                'perimeter': float(shape['perimeter']),
                'vertices': int(shape['vertices']),
                'corners': [[int(corner[0]), int(corner[1])] for corner in shape['corners']]
            })
        
        with open(output_file, 'w') as f:
            json.dump(coordinates, f, indent=2)

def main():
    """Main function for adaptive shape detection"""
    detector = AdaptiveDetector()
    
    print("🎯 Adaptive Shape Detection System")
    print("=" * 60)
    print("Automatically adapts to any image content")
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
    
    # Use the first image
    image_path = image_files[0]
    print(f"\nProcessing: {image_path}")
    
    # Detect shapes adaptively
    result = detector.create_adaptive_visualization(image_path)
    if result is None:
        return
    
    fig, shapes, clusters, analysis, params = result
    
    # Generate comprehensive report
    report = detector.generate_adaptive_report(shapes, clusters, analysis, params)
    
    # Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f'adaptive_detection_{base_name}.png', dpi=300, bbox_inches='tight')
    detector.save_shape_coordinates(shapes, f'shape_coordinates_{base_name}.json')
    
    # Print comprehensive results
    print(f"\n📊 Adaptive Detection Results:")
    print(f"Total shapes detected: {report['total_shapes']}")
    print(f"Total clusters: {report['total_clusters']}")
    
    print(f"\n🔍 Image Analysis:")
    print(f"Image size: {analysis['image_size']}")
    print(f"Mean intensity: {analysis['mean_intensity']:.1f}")
    print(f"Edge density: {analysis['edge_density']:.3f}")
    print(f"Estimated median area: {analysis['median_area']:.1f}")
    
    print(f"\n⚙️ Adaptive Parameters:")
    print(f"Min area: {params['min_area']}")
    print(f"Max area: {params['max_area']}")
    print(f"Aspect ratio range: {params['aspect_ratio_range']}")
    print(f"Rectangularity threshold: {params['rectangularity_threshold']:.2f}")
    
    print(f"\n📈 Statistical Analysis:")
    print(f"Average area: {report['statistics']['average_area']:.1f} pixels")
    print(f"Average aspect ratio: {report['statistics']['average_aspect_ratio']:.2f}")
    print(f"Average rectangularity: {report['statistics']['average_rectangularity']:.3f}")
    
    print(f"\n📁 Files saved:")
    print(f"- adaptive_detection_{base_name}.png")
    print(f"- shape_coordinates_{base_name}.json")
    
    # Show the plot
    plt.show()
    
    print("\n🎉 Adaptive detection completed!")
    print("The system automatically adapted to your image content.")

if __name__ == "__main__":
    main()
