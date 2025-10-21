# ============================================================================
# CONFIG.PY - Configuration Settings
# ============================================================================

import os

# Model Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
CONFIDENCE_THRESHOLD = 0.5  # Threshold for binary mask

# File Paths
MODEL_PATH = 'models/best_model.h5'
UPLOAD_FOLDER = 'temp_uploads'

# Visualization Settings
OVERLAY_ALPHA = 0.4  # Transparency for overlay
SPILL_COLOR = [255, 0, 0]  # Red color for detected oil spills

# Streamlit Settings
PAGE_TITLE = "HydroVexel - Oil Spill Detection System"
PAGE_ICON = "🛢️"
MAX_FILE_SIZE = 10  # MB

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
