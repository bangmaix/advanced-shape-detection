from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-shape-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Shape Detection and Coloring System for Computer Vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/advanced-shape-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shape-detection=demo_shape_detector:main",
            "advanced-detection=advanced_shape_detection:main",
            "intelligent-detection=intelligent_shape_detector:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.png", "*.jpg", "*.json"],
    },
    keywords="computer-vision, shape-detection, opencv, machine-learning, image-processing",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/advanced-shape-detection/issues",
        "Source": "https://github.com/yourusername/advanced-shape-detection",
        "Documentation": "https://github.com/yourusername/advanced-shape-detection#readme",
    },
)
