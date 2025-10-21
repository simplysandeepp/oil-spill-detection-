# ============================================================================
# VISUALIZATION.PY - Visualization Utilities (Revamped UI Version)
# ============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import config.config as cfg


def create_overlay(original_image, binary_mask, alpha=cfg.OVERLAY_ALPHA):
    """
    Create overlay visualization with detected oil spills highlighted
    
    IMPROVED: Enhanced color contrast and clarity for better visibility
    
    Args:
        original_image: RGB image (H, W, 3)
        binary_mask: Binary mask (H, W) with values 0 or 255
        alpha: Transparency factor for overlay
        
    Returns:
        overlay_image: Image with high-contrast red overlay on detected regions (uint8)
    """
    # Ensure images are uint8
    if original_image.dtype != np.uint8:
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)
    
    # Ensure binary mask is uint8
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    
    # Ensure same dimensions
    if original_image.shape[:2] != binary_mask.shape:
        binary_mask = cv2.resize(binary_mask, 
                                (original_image.shape[1], original_image.shape[0]))
    
    # IMPROVED: Create high-contrast overlay with brighter red color
    overlay = original_image.copy()
    
    # Use a more vibrant red color for better visibility (BGR format in cv2)
    # Changed from config color to a brighter, more visible red
    bright_red = [255, 0, 0]  # Pure bright red in RGB
    overlay[binary_mask > 127] = bright_red
    
    # IMPROVED: Add a slight edge detection for clearer boundaries
    # This makes the overlay edges more visible
    edges = cv2.Canny(binary_mask, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Blend with original
    blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    
    # IMPROVED: Add white borders around detected regions for clarity
    blended[edges_dilated > 0] = [255, 255, 255]  # White edges
    
    # Ensure output is uint8
    blended = blended.astype(np.uint8)
    
    return blended


def create_confidence_heatmap(confidence_map, original_image=None):
    """
    Create colored heatmap showing model confidence
    
    IMPROVED: Enhanced color mapping and contrast for better interpretation
    
    Args:
        confidence_map: Probability values (H, W) with range [0, 1]
        original_image: Optional original image to blend with
        
    Returns:
        heatmap_image: RGB heatmap visualization with improved contrast (uint8)
    """
    # Normalize to 0-255
    heatmap = (confidence_map * 255).astype(np.uint8)
    
    # IMPROVED: Use COLORMAP_JET for better color distinction
    # JET provides better visual separation of confidence levels
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Optionally blend with original image
    if original_image is not None:
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            if original_image.max() <= 1.0:
                original_image = (original_image * 255).astype(np.uint8)
            else:
                original_image = original_image.astype(np.uint8)
            
        if original_image.shape[:2] != heatmap_colored.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored,
                                        (original_image.shape[1], 
                                         original_image.shape[0]))
        
        # IMPROVED: Adjusted blending ratio for better visibility
        # 60% heatmap, 40% original gives better contrast
        heatmap_colored = cv2.addWeighted(original_image, 0.4, 
                                         heatmap_colored, 0.6, 0)
    
    # IMPROVED: Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This makes low and high confidence areas more distinguishable
    lab = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    heatmap_colored = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Ensure output is uint8
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    return heatmap_colored


def create_comparison_view(original, binary_mask, confidence_map, overlay):
    """
    Create a 2x2 grid comparison view
    
    IMPROVED: Enhanced layout with better labels and contrast
    
    Args:
        original: Original image
        binary_mask: Binary segmentation
        confidence_map: Confidence heatmap
        overlay: Overlay visualization
        
    Returns:
        comparison_image: Combined visualization with clear labels (uint8)
    """
    # Ensure all images are uint8
    if original.dtype != np.uint8:
        original = (original * 255).astype(np.uint8) if original.max() <= 1.0 else original.astype(np.uint8)
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    if overlay.dtype != np.uint8:
        overlay = overlay.astype(np.uint8)
    
    # IMPROVED: Use white background and better styling
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), facecolor='white')
    fig.suptitle('Oil Spill Detection Analysis', fontsize=20, fontweight='bold', 
                 color='#0D1B2A', y=0.98)
    
    # IMPROVED: Better title styling with background boxes
    title_style = {
        'fontweight': 'bold',
        'fontsize': 14,
        'color': '#0D1B2A',
        'bbox': dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', 
                    edgecolor='#1976D2', linewidth=2)
    }
    
    # Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', **title_style)
    axes[0, 0].axis('off')
    
    # Binary Mask
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title('Binary Segmentation', **title_style)
    axes[0, 1].axis('off')
    
    # Confidence Heatmap
    heatmap = create_confidence_heatmap(confidence_map)
    axes[1, 0].imshow(heatmap)
    axes[1, 0].set_title('Confidence Heatmap', **title_style)
    axes[1, 0].axis('off')
    
    # Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Detection Overlay', **title_style)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    fig.canvas.draw()
    comparison_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    comparison_img = comparison_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return comparison_img


def add_metrics_overlay(image, metrics, font_scale=0.8):
    """
    Add text overlay with detection metrics
    
    IMPROVED: High-contrast text with clear backgrounds and borders
    
    Args:
        image: Input image
        metrics: Dictionary with detection statistics
        font_scale: Font size scale factor
        
    Returns:
        image_with_text: Image with clear, readable metrics overlay (uint8)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    img_copy = image.copy()
    
    # Prepare text
    lines = [
        f"Oil Spill Detected: {'YES' if metrics['has_spill'] else 'NO'}",
        f"Coverage: {metrics['coverage_percentage']:.2f}%",
        f"Avg Confidence: {metrics['avg_confidence']:.2%}",
        f"Max Confidence: {metrics['max_confidence']:.2%}"
    ]
    
    # IMPROVED: Calculate dynamic positioning based on image size
    img_height, img_width = img_copy.shape[:2]
    box_width = int(img_width * 0.35)  # 35% of image width
    box_height = 50
    y_offset = 30
    x_offset = 20
    
    for i, line in enumerate(lines):
        y_pos = y_offset + i * (box_height + 10)
        
        # IMPROVED: White background with dark border for maximum contrast
        # Draw shadow for depth
        cv2.rectangle(img_copy, 
                     (x_offset + 3, y_pos - 30 + 3), 
                     (x_offset + box_width + 3, y_pos + 10 + 3), 
                     (0, 0, 0), -1)  # Shadow
        
        # Main background - solid white
        cv2.rectangle(img_copy, 
                     (x_offset, y_pos - 30), 
                     (x_offset + box_width, y_pos + 10), 
                     (255, 255, 255), -1)  # White background
        
        # Border - dark blue for contrast
        cv2.rectangle(img_copy, 
                     (x_offset, y_pos - 30), 
                     (x_offset + box_width, y_pos + 10), 
                     (13, 27, 42), 3)  # Dark blue border (#0D1B2A)
        
        # IMPROVED: Text color - dark for readability on white background
        if i == 0:
            # First line: Red if spill detected, Green if clean
            if metrics['has_spill']:
                text_color = (0, 0, 255)  # Red (BGR)
            else:
                text_color = (0, 200, 0)  # Green (BGR)
        else:
            text_color = (13, 27, 42)  # Dark blue (BGR) for other lines
        
        # IMPROVED: Add text with outline for extra clarity
        # White outline
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cv2.putText(img_copy, line, 
                       (x_offset + 15 + dx, y_pos + dy), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (255, 255, 255), 3)
        
        # Main text
        cv2.putText(img_copy, line, 
                   (x_offset + 15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   text_color, 2)
    
    return img_copy


def create_legend_overlay(image, show_legend=True):
    """
    Add a color legend to heatmap images
    
    NEW FUNCTION: Helps users interpret confidence heatmaps
    
    Args:
        image: Heatmap image
        show_legend: Whether to show the legend
        
    Returns:
        image_with_legend: Image with color legend (uint8)
    """
    if not show_legend:
        return image
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    img_copy = image.copy()
    img_height, img_width = img_copy.shape[:2]
    
    # Legend parameters
    legend_height = 30
    legend_width = int(img_width * 0.3)
    legend_x = img_width - legend_width - 20
    legend_y = img_height - legend_height - 20
    
    # Create gradient bar
    gradient = np.linspace(0, 255, legend_width).astype(np.uint8)
    gradient = np.tile(gradient, (legend_height, 1))
    gradient_colored = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
    gradient_colored = cv2.cvtColor(gradient_colored, cv2.COLOR_BGR2RGB)
    
    # Add white background with border
    cv2.rectangle(img_copy, 
                 (legend_x - 5, legend_y - 5), 
                 (legend_x + legend_width + 5, legend_y + legend_height + 35), 
                 (255, 255, 255), -1)
    cv2.rectangle(img_copy, 
                 (legend_x - 5, legend_y - 5), 
                 (legend_x + legend_width + 5, legend_y + legend_height + 35), 
                 (13, 27, 42), 2)
    
    # Place gradient
    img_copy[legend_y:legend_y + legend_height, 
             legend_x:legend_x + legend_width] = gradient_colored
    
    # Add labels
    cv2.putText(img_copy, "Low", (legend_x, legend_y + legend_height + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (13, 27, 42), 2)
    cv2.putText(img_copy, "High", 
               (legend_x + legend_width - 40, legend_y + legend_height + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (13, 27, 42), 2)
    cv2.putText(img_copy, "Confidence",
               (legend_x + legend_width // 2 - 40, legend_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (13, 27, 42), 2)
    
    return img_copy