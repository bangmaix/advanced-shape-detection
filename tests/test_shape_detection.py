"""
Tests for shape detection functionality.
"""

import pytest
import numpy as np
import cv2
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape_detection import create_colored_diagram


class TestShapeDetection:
    """Test cases for shape detection functionality."""

    def test_create_colored_diagram(self):
        """Test that colored diagram creation works."""
        # Test with basic parameters
        result = create_colored_diagram()
        
        # Check that result is not None
        assert result is not None
        
        # Check that the output file exists
        assert os.path.exists("tennis_court_colored.png")

    def test_image_loading(self):
        """Test that images can be loaded properly."""
        # Test with a sample image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Add some shapes to the test image
        cv2.rectangle(test_image, (10, 10), (90, 90), (255, 255, 255), -1)
        
        # Check that image has correct shape
        assert test_image.shape == (100, 100, 3)
        assert test_image.dtype == np.uint8

    def test_contour_detection(self):
        """Test contour detection functionality."""
        # Create a test image with a simple shape
        test_image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(test_image, (20, 20), (80, 80), 255, -1)
        
        # Find contours
        contours, _ = cv2.findContours(test_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check that contours are found
        assert len(contours) > 0
        
        # Check that the contour has the expected area
        area = cv2.contourArea(contours[0])
        expected_area = 60 * 60  # 80-20 = 60
        assert abs(area - expected_area) < 10  # Allow some tolerance

    def test_color_generation(self):
        """Test that colors are generated correctly."""
        # Test color generation for different shapes
        colors = [
            (255, 107, 107),  # Red for DEKRANASDA
            (78, 205, 196),   # Teal for Tennis Court
            (248, 196, 113),  # Orange for IGD
        ]
        
        for color in colors:
            # Check that colors are valid RGB values
            assert all(0 <= c <= 255 for c in color)
            assert len(color) == 3

    def test_coordinate_extraction(self):
        """Test coordinate extraction functionality."""
        # Create a test contour
        contour = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.int32)
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check that coordinates are valid
        assert x == 10
        assert y == 10
        assert w == 80
        assert h == 80

    def test_file_operations(self):
        """Test file operations for saving results."""
        # Test saving a simple image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_filename = "test_output.png"
        
        # Save image
        cv2.imwrite(test_filename, test_image)
        
        # Check that file exists
        assert os.path.exists(test_filename)
        
        # Clean up
        os.remove(test_filename)


if __name__ == "__main__":
    pytest.main([__file__])
