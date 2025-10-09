# ============================================================================
# VISUALIZATION.PY - Visualization Utilities
# ============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import config.config as cfg


def create_overlay(original_image, binary_mask, alpha=cfg.OVERLAY_ALPHA):
    """
    Create overlay visualization with detected oil spills highlighted
    
    Args:
        original_image: RGB image (H, W, 3)
        binary_mask: Binary mask (H, W) with values 0 or 255
        alpha: Transparency factor for overlay
        
    Returns:
        overlay_image: Image with red overlay on detected regions
    """
    # Ensure same dimensions
    if original_image.shape[:2] != binary_mask.shape:
        binary_mask = cv2.resize(binary_mask, 
                                (original_image.shape[1], original_image.shape[0]))
    
    # Create overlay
    overlay = original_image.copy()
    overlay[binary_mask > 127] = cfg.SPILL_COLOR
    
    # Blend with original
    blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    
    return blended


def create_confidence_heatmap(confidence_map, original_image=None):
    """
    Create colored heatmap showing model confidence
    
    Args:
        confidence_map: Probability values (H, W) with range [0, 1]
        original_image: Optional original image to blend with
        
    Returns:
        heatmap_image: RGB heatmap visualization
    """
    # Normalize to 0-255
    heatmap = (confidence_map * 255).astype(np.uint8)
    
    # Apply colormap (yellow-orange-red for oil theme)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Optionally blend with original image
    if original_image is not None:
        if original_image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored,
                                        (original_image.shape[1], 
                                         original_image.shape[0]))
        heatmap_colored = cv2.addWeighted(original_image, 0.5, 
                                         heatmap_colored, 0.5, 0)
    
    return heatmap_colored


def create_comparison_view(original, binary_mask, confidence_map, overlay):
    """
    Create a 2x2 grid comparison view
    
    Args:
        original: Original image
        binary_mask: Binary segmentation
        confidence_map: Confidence heatmap
        overlay: Overlay visualization
        
    Returns:
        comparison_image: Combined visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    
    # Binary Mask
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title('Binary Segmentation', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    
    # Confidence Heatmap
    heatmap = create_confidence_heatmap(confidence_map)
    axes[1, 0].imshow(heatmap)
    axes[1, 0].set_title('Confidence Heatmap', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    
    # Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Detection Overlay', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    fig.canvas.draw()
    comparison_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison_img = comparison_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return comparison_img


def add_metrics_overlay(image, metrics, font_scale=0.7):
    """
    Add text overlay with detection metrics
    
    Args:
        image: Input image
        metrics: Dictionary with detection statistics
        font_scale: Font size scale factor
        
    Returns:
        image_with_text: Image with metrics overlay
    """
    img_copy = image.copy()
    
    # Prepare text
    lines = [
        f"Oil Spill Detected: {'YES' if metrics['has_spill'] else 'NO'}",
        f"Coverage: {metrics['coverage_percentage']:.2f}%",
        f"Avg Confidence: {metrics['avg_confidence']:.2%}",
        f"Max Confidence: {metrics['max_confidence']:.2%}"
    ]
    
    # Background rectangle
    y_offset = 30
    for i, line in enumerate(lines):
        y_pos = y_offset + i * 40
        
        # Background
        cv2.rectangle(img_copy, (10, y_pos - 25), (400, y_pos + 5), 
                     (0, 0, 0), -1)
        cv2.rectangle(img_copy, (10, y_pos - 25), (400, y_pos + 5), 
                     (255, 255, 255), 2)
        
        # Text color based on detection
        if i == 0:
            color = (0, 255, 0) if metrics['has_spill'] else (255, 255, 255)
        else:
            color = (255, 255, 255)
        
        cv2.putText(img_copy, line, (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    
    return img_copy
