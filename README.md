# Advanced Shape Detection and Coloring System

This project provides multiple approaches for detecting and coloring shapes in images, with a focus on tennis court kiosk diagrams.

## 🚀 Features

### Basic System (`shape_detection.py`)
- **Manual Diagram Creation**: Creates colored diagrams based on layout descriptions
- **Color Coding**: Each shape is colored with a unique color for easy identification
- **Legend**: Includes a color legend for easy reference

### Advanced System (`advanced_shape_detection.py`)
- **Computer Vision**: Uses OpenCV for automatic shape detection
- **Contour Analysis**: Advanced contour detection and analysis
- **Shape Classification**: Intelligent shape type classification
- **Layout Pattern Analysis**: Analyzes layout patterns to identify specific sections

### Intelligent System (`intelligent_shape_detector.py`)
- **Multi-Technique Processing**: Combines multiple image processing techniques
- **Machine Learning**: Uses clustering algorithms for shape grouping
- **Hierarchical Analysis**: Analyzes shape hierarchy and relationships
- **Text Region Detection**: Identifies text regions in diagrams
- **Detailed Reporting**: Generates comprehensive shape analysis reports

### Demo System (`demo_shape_detector.py`)
- **Universal Detection**: Works with any image file
- **Simple Interface**: Easy-to-use command line interface
- **Real-time Analysis**: Provides immediate shape analysis results
- **Multiple Views**: Shows original, processed, and detected images

## 🎨 Detected Shapes

Based on the tennis court diagram, the following shapes are detected and colored:

1. **DEKRANASDA BUKITTINGGI** - 🔴 Red (#FF6B6B)
2. **LAPANGAN TENIS** (Tennis Court) - 🟢 Teal (#4ECDC4)
3. **IGD RSAM LAMA** - 🟠 Orange (#F8C471)
4. **Kiosks 1-8** - Each with different colors:
   - Kiosk 1: 🔵 Blue (#45B7D1)
   - Kiosk 2: 🟢 Green (#96CEB4)
   - Kiosk 3: 🟡 Yellow (#FFEAA7)
   - Kiosk 4: 🟣 Plum (#DDA0DD)
   - Kiosk 5: 🟢 Mint (#98D8C8)
   - Kiosk 6: 🟡 Gold (#F7DC6F)
   - Kiosk 7: 🟣 Lavender (#BB8FCE)
   - Kiosk 8: 🔵 Sky Blue (#85C1E9)
5. **Jl. Dr. Abdul Rivai** (Street) - ⚪ Gray (#BDC3C7)

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/bangmaix/advanced-shape-detection.git
cd advanced-shape-detection
```

2. **Install the required dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install in development mode** (optional):
```bash
pip install -e .
```

## 🛠️ Usage

### For Manual Diagram Creation:
```bash
python shape_detection.py
```

### For Advanced Shape Detection:
```bash
python advanced_shape_detection.py
```

### For Intelligent Shape Detection:
```bash
python intelligent_shape_detector.py
```

### For Universal Shape Detection (Works with any image):
```bash
python demo_shape_detector.py [image_path]
```

## 📊 Output Examples

### Manual System:
- `tennis_court_colored.png` - Colored diagram with legend
- `manual_diagram_colored.png` - Manual layout recreation

### Advanced System:
- Multiple visualization views
- Shape hierarchy analysis
- Position-based clustering

### Intelligent System:
- 4-panel analysis (original, processed, detected, analyzed)
- Detailed shape reports in JSON format
- Layout pattern recognition

### Demo System:
- 3-panel visualization (original, threshold, detected)
- Shape analysis table with statistics
- Universal compatibility with any image

## 🔧 Technical Features

### Image Processing Techniques:
- **Gaussian Blur**: Noise reduction
- **Adaptive Thresholding**: Multiple thresholding methods
- **Morphological Operations**: Shape cleaning and enhancement
- **Contour Detection**: Advanced contour finding algorithms
- **Shape Approximation**: Polygon approximation for shape classification

### Machine Learning:
- **K-means Clustering**: Shape grouping by position
- **DBSCAN**: Density-based clustering for kiosks
- **Hierarchical Analysis**: Shape relationship analysis

### Computer Vision:
- **MSER Detection**: Text region identification
- **Geometric Analysis**: Aspect ratio, circularity, area calculations
- **Layout Pattern Recognition**: Tennis court specific pattern matching

## 📋 Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+
- Scikit-learn 1.3+
- SciPy 1.11+

## 🎯 Use Cases

1. **Architectural Diagrams**: Analyze building layouts and floor plans
2. **Technical Drawings**: Process engineering diagrams and schematics
3. **Map Analysis**: Detect and classify geographic features
4. **Document Processing**: Extract shapes from technical documents
5. **Quality Control**: Verify shape consistency in manufacturing

## 🔗 Links

- **Repository**: https://github.com/bangmaix/advanced-shape-detection
- **Releases**: https://github.com/bangmaix/advanced-shape-detection/releases
- **Issues**: https://github.com/bangmaix/advanced-shape-detection/issues
- **Actions**: https://github.com/bangmaix/advanced-shape-detection/actions
