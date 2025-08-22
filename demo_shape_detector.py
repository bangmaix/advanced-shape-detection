import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def simple_shape_detection(image_path):
    """Simple but effective shape detection for any image"""
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 500  # Adjust this value based on your image
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Threshold image
    ax2.imshow(thresh, cmap='gray')
    ax2.set_title('Threshold Image')
    ax2.axis('off')
    
    # Detected shapes with colors
    ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Define colors for shapes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9', 
              '#F8C471', '#BDC3C7']
    
    # Draw each contour with a different color
    for i, contour in enumerate(filtered_contours):
        color = colors[i % len(colors)]
        
        # Fill the contour
        ax3.fill(contour[:, 0, 0], contour[:, 0, 1], 
                color=color, alpha=0.6, edgecolor='black', linewidth=2)
        
        # Add shape number
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            ax3.text(cx, cy, str(i+1), ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax3.set_title(f'Detected Shapes ({len(filtered_contours)} shapes)')
    ax3.axis('off')
    
    plt.tight_layout()
    return fig, filtered_contours

def analyze_shapes(contours):
    """Analyze the detected shapes"""
    analysis = []
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Approximate shape
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Determine shape type
        if vertices == 3:
            shape_type = "Triangle"
        elif vertices == 4:
            if 0.9 < aspect_ratio < 1.1:
                shape_type = "Square"
            else:
                shape_type = "Rectangle"
        elif vertices == 5:
            shape_type = "Pentagon"
        elif vertices == 6:
            shape_type = "Hexagon"
        elif vertices > 6:
            shape_type = "Polygon"
        else:
            shape_type = "Unknown"
        
        analysis.append({
            'shape_id': i + 1,
            'type': shape_type,
            'area': int(area),
            'perimeter': int(perimeter),
            'vertices': vertices,
            'aspect_ratio': round(aspect_ratio, 2),
            'bbox': (x, y, w, h)
        })
    
    return analysis

def main():
    """Main function with command line interface"""
    print("🎯 Simple Shape Detection System")
    print("=" * 40)
    
    # Check if image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for common image files in current directory
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_files = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        if image_files:
            print("Found image files in current directory:")
            for i, file in enumerate(image_files):
                print(f"{i+1}. {file}")
            
            try:
                choice = int(input("\nSelect image number (or press Enter for first image): ")) - 1
                if 0 <= choice < len(image_files):
                    image_path = image_files[choice]
                else:
                    image_path = image_files[0]
            except (ValueError, IndexError):
                image_path = image_files[0]
        else:
            print("No image files found in current directory.")
            print("Usage: python demo_shape_detector.py <image_path>")
            return
    
    print(f"\nProcessing image: {image_path}")
    
    # Detect shapes
    result = simple_shape_detection(image_path)
    if result is None:
        return
    
    fig, contours = result
    
    # Analyze shapes
    analysis = analyze_shapes(contours)
    
    # Print analysis
    print(f"\n📊 Shape Analysis Results:")
    print(f"Total shapes detected: {len(contours)}")
    print("\nDetailed analysis:")
    print("-" * 60)
    print(f"{'ID':<3} {'Type':<12} {'Area':<8} {'Vertices':<8} {'Aspect Ratio':<12}")
    print("-" * 60)
    
    for shape in analysis:
        print(f"{shape['shape_id']:<3} {shape['type']:<12} {shape['area']:<8} "
              f"{shape['vertices']:<8} {shape['aspect_ratio']:<12}")
    
    # Save results
    output_filename = f"shape_detection_results_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Results saved as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    print("\n🎉 Shape detection completed!")
    print("The system detected and colored each shape with a unique color.")

if __name__ == "__main__":
    main()

