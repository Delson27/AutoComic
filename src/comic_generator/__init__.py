"""
Comic Generator Package
A package for generating comic-style frames from videos.

The package consists of two main modules:
1. keyframes - For extracting key frames from videos
2. styling - For converting frames to comic-style images
"""

__version__ = '1.0.0'

from .keyframes import generate_keyframes
from .styling import cartoonify, read_image 