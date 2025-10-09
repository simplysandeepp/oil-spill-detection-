# ============================================================================
# PREPROCESSING.PY - Image Preprocessing Utilities
# ============================================================================

import cv2
import numpy as np
from PIL import Image
import config.config as cfg

def load_and_preprocess_image(image_input, target_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH)):
    """
    Load and preprocess image for model inference
    
    Args:
        image_input: Either file path (str) or PIL Image or numpy array
        target_size: Tuple (height, width) for resizing
        
    Returns:
        preprocessed_img: Normalized image ready for model (batch, H, W, C)
        original_img: Original image in RGB format for visualization
    """
    # Handle different input types
    if isinstance(image_input, str):
        # Load from file path
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, Image.Image):
        # PIL Image
        img = np.array(image_input)
    elif isinstance(image_input, np.ndarray):
        # Already numpy array
        img = image_input
    else:
        raise ValueError(f"Unsupported input type: {type(image_input)}")
    
    # Store original for visualization
    original_img = img.copy()
    
    # Resize to model input size
    resized_img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    preprocessed_img = resized_img.astype(np.float32) / 255.0
    
    # Add batch dimension
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    
    return preprocessed_img, original_img


def postprocess_mask(pred_mask, threshold=cfg.CONFIDENCE_THRESHOLD, target_size=None):
    """
    Convert probability mask to binary mask
    
    Args:
        pred_mask: Model prediction (H, W) with values [0, 1]
        threshold: Confidence threshold for binary conversion
        target_size: Optional (width, height) to resize mask to original image size
        
    Returns:
        binary_mask: Binary mask (0 or 255)
        confidence_map: Original probability values
    """
    # Squeeze extra dimensions
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask.squeeze()
    
    # Store confidence map
    confidence_map = pred_mask.copy()
    
    # Create binary mask
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Resize if needed
    if target_size is not None:
        binary_mask = cv2.resize(binary_mask, target_size)
        confidence_map = cv2.resize(confidence_map, target_size)
    
    return binary_mask, confidence_map


def validate_image(image_input):
    """
    Validate uploaded image
    
    Returns:
        (is_valid, error_message)
    """
    try:
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            img = image_input
            
        if img is None:
            return False, "Unable to read image file"
        
        if len(img.shape) not in [2, 3]:
            return False, "Invalid image dimensions"
        
        if len(img.shape) == 3 and img.shape[2] not in [3, 4]:
            return False, "Image must be RGB or RGBA format"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"
