"""
Style transfer module for creating cartoon-like effects with enhanced color preservation.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def read_image(image_path_or_data):
    """
    Read an image from file path or binary data and convert it to RGB format.
    
    Args:
        image_path_or_data: Path to image file or binary image data
    
    Returns:
        numpy.ndarray: RGB image array
    """
    try:
        if isinstance(image_path_or_data, str):
            img = cv2.imread(image_path_or_data)
        else:
            img = cv2.imdecode(np.frombuffer(image_path_or_data, np.uint8), cv2.IMREAD_COLOR)
            
        if img is None:
            raise ValueError("Failed to load image")
            
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
            
    except Exception as e:
        logger.error(f"Error reading image: {str(e)}")
        raise

def enhance_edges(img, line_size=7, blur_value=7):
    """
    Create enhanced edges that preserve more detail.
    
    Args:
        img: Input image
        line_size: Size of the adaptive threshold window
        blur_value: Gaussian blur kernel size
    
    Returns:
        numpy.ndarray: Enhanced edge mask
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply median blur to reduce noise while preserving edges
        gray_blur = cv2.medianBlur(gray, blur_value)
        
        # Apply adaptive thresholding for better edge detection
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        
        # Dilate edges slightly to ensure connectivity
        kernel = np.ones((line_size, line_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
        
    except Exception as e:
        logger.error(f"Error creating edge mask: {str(e)}")
        raise

def color_quantization(img, k=9):
    """
    Reduce the number of colors while preserving dominant colors.
    
    Args:
        img: Input image
        k: Number of colors to quantize to
    
    Returns:
        numpy.ndarray: Color quantized image
    """
    try:
        # Convert to float32 for k-means
        data = np.float32(img).reshape((-1, 3))
        
        # Define criteria for k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        
        # Apply k-means clustering
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, 10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert back to uint8
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        
        # Reshape back to original image shape
        quantized = quantized.reshape(img.shape)
        
        return quantized
        
    except Exception as e:
        logger.error(f"Error in color quantization: {str(e)}")
        raise

def smooth_image(img):
    """
    Apply multiple bilateral filters for enhanced smoothing while preserving edges.
    
    Args:
        img: Input image
    
    Returns:
        numpy.ndarray: Smoothed image
    """
    try:
        # First pass with larger sigma values for overall smoothing
        smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Second pass with smaller sigma values for detail preservation
        smooth = cv2.bilateralFilter(smooth, d=7, sigmaColor=50, sigmaSpace=50)
        
        return smooth
    except Exception as e:
        logger.error(f"Error in bilateral smoothing: {str(e)}")
        raise

def cartoonify(img):
    """
    Convert an image to cartoon-style with enhanced color preservation.
    
    Args:
        img: Input image in RGB format
    
    Returns:
        numpy.ndarray: Cartoonified image with preserved colors
    """
    try:
        # Preserve original colors by working on a copy
        img_copy = img.copy()
        
        # Apply color quantization with more colors
        color_img = color_quantization(img_copy, k=9)
        
        # Smooth the quantized image
        smooth = smooth_image(color_img)
        
        # Get enhanced edges
        edges = enhance_edges(img_copy)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Create edge mask with reduced intensity
        edge_mask = cv2.addWeighted(
            edges_colored, 0.1,  # Reduce edge intensity
            np.ones_like(edges_colored) * 255, 0.9,
            0
        )
        
        # Combine smooth colors with soft edges
        cartoon = cv2.addWeighted(
            smooth, 0.9,  # More weight to colors
            cv2.bitwise_and(smooth, edge_mask), 0.1,  # Less weight to edges
            0
        )
        
        # Ensure no pure black or white regions
        cartoon = np.clip(cartoon, 10, 245)  # Limit the range to avoid extremes
        
        return cartoon
        
    except Exception as e:
        logger.error(f"Error creating cartoon effect: {str(e)}")
        raise 