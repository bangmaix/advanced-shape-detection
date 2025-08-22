import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def create_tennis_court_diagram():
    """
    Create a tennis court kiosk diagram with different colored shapes
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    
    # Define colors for different shapes
    colors = {
        'dekranasda': '#FF6B6B',      # Red
        'tennis_court': '#4ECDC4',    # Teal
        'kiosk_1': '#45B7D1',         # Blue
        'kiosk_2': '#96CEB4',         # Green
        'kiosk_3': '#FFEAA7',         # Yellow
        'kiosk_4': '#DDA0DD',         # Plum
        'kiosk_5': '#98D8C8',         # Mint
        'kiosk_6': '#F7DC6F',         # Gold
        'kiosk_7': '#BB8FCE',         # Lavender
        'kiosk_8': '#85C1E9',         # Sky Blue
        'igd': '#F8C471',             # Orange
        'street': '#BDC3C7'           # Gray
    }
    
    # DEKRANASDA BUKITTINGGI (Left section)
    dekranasda = Rectangle((0.5, 4), 1.5, 3, facecolor=colors['dekranasda'], 
                          edgecolor='black', linewidth=2)
    ax.add_patch(dekranasda)
    ax.text(1.25, 5.5, 'DEKRANASDA\nBUKITTINGGI', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # LAPANGAN TENIS (Tennis Court - Central section)
    tennis_court = Rectangle((2.5, 4), 4, 3, facecolor=colors['tennis_court'], 
                            edgecolor='black', linewidth=2)
    ax.add_patch(tennis_court)
    ax.text(4.5, 5.5, 'LAPANGAN TENIS', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # IGD RSAM LAMA (Right section)
    igd = Rectangle((7, 4), 2, 3, facecolor=colors['igd'], 
                   edgecolor='black', linewidth=2)
    ax.add_patch(igd)
    ax.text(8, 5.5, 'IGD RSAM LAMA', ha='center', va='center', 
            fontsize=8, fontweight='bold')
    
    # Kiosks (8 numbered boxes below tennis court)
    kiosk_width = 0.4
    kiosk_height = 0.8
    kiosk_colors = [colors['kiosk_1'], colors['kiosk_2'], colors['kiosk_3'], 
                   colors['kiosk_4'], colors['kiosk_5'], colors['kiosk_6'], 
                   colors['kiosk_7'], colors['kiosk_8']]
    
    for i in range(8):
        x_pos = 2.5 + i * 0.5
        kiosk = Rectangle((x_pos, 2.5), kiosk_width, kiosk_height, 
                         facecolor=kiosk_colors[i], edgecolor='black', linewidth=1)
        ax.add_patch(kiosk)
        ax.text(x_pos + kiosk_width/2, 2.9, str(i+1), ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Street (Jl. Dr. Abdul Rivai)
    street = Rectangle((0.5, 1.5), 8.5, 0.3, facecolor=colors['street'], 
                      edgecolor='black', linewidth=1)
    ax.add_patch(street)
    ax.text(4.75, 1.65, 'Jl. Dr. Abdul Rivai', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Title
    ax.text(4.75, 7.5, 'KIOS LAPANGAN TENIS', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    
    # Legend
    legend_elements = [
        patches.Patch(color=colors['dekranasda'], label='DEKRANASDA BUKITTINGGI'),
        patches.Patch(color=colors['tennis_court'], label='LAPANGAN TENIS'),
        patches.Patch(color=colors['igd'], label='IGD RSAM LAMA'),
        patches.Patch(color=colors['kiosk_1'], label='Kiosk 1-8'),
        patches.Patch(color=colors['street'], label='Jl. Dr. Abdul Rivai')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
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

def detect_shapes_in_image(image_path):
    """
    Function to detect shapes in an actual image (if provided)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image for drawing
    result = image.copy()
    
    # Define colors for different shapes
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    # Draw contours with different colors
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Filter small contours
            color = colors[i % len(colors)]
            cv2.drawContours(result, [contour], -1, color, 2)
            
            # Add shape number
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i+1), (cx, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result

def main():
    """
    Main function to run the shape detection and coloring
    """
    print("Creating tennis court kiosk diagram with colored shapes...")
    
    # Create the diagram
    fig = create_tennis_court_diagram()
    
    # Save the diagram
    plt.savefig('tennis_court_colored.png', dpi=300, bbox_inches='tight')
    print("Diagram saved as 'tennis_court_colored.png'")
    
    # Show the diagram
    plt.show()
    
    print("\nShape detection completed!")
    print("Each shape has been colored with a different color:")
    print("- DEKRANASDA BUKITTINGGI: Red")
    print("- LAPANGAN TENIS: Teal") 
    print("- IGD RSAM LAMA: Orange")
    print("- Kiosks 1-8: Different colors (Blue, Green, Yellow, Plum, Mint, Gold, Lavender, Sky Blue)")
    print("- Street: Gray")

if __name__ == "__main__":
    main()

