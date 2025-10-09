# ============================================================================
# INFERENCE.PY - Model Inference with Google Drive Download
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(50000)

# Disable GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import config.config as cfg
from utils.preprocessing import load_and_preprocess_image, postprocess_mask

try:
    import gdown
except ImportError:
    print("⚠ gdown not installed. Install with: pip install gdown")


def download_model_if_needed(model_path, gdrive_file_id):
    """Download model from Google Drive if not present"""
    if not os.path.exists(model_path):
        print("📥 Downloading model from Google Drive...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        gdown.download(url, model_path, quiet=False)
        print("✓ Model downloaded successfully!")
    else:
        print("✓ Model already exists locally")


class OilSpillDetector:
    """Oil Spill Detection Model Wrapper"""
    
    # Your Google Drive File ID
    GDRIVE_FILE_ID = "11PQQ0zWCFoWnJz30fvcveDEloUu-VDcf"
    
    def __init__(self, model_path=cfg.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        
        # Download model if needed
        download_model_if_needed(self.model_path, self.GDRIVE_FILE_ID)
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load model by rebuilding architecture and loading weights"""
        try:
            print("🔄 Loading model...")
            
            from models.model_architecture import build_enhanced_unet
            
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(50000)
            
            print("  → Building architecture...")
            self.model = build_enhanced_unet()
            
            print(f"  → Loading weights from {self.model_path}...")
            self.model.load_weights(self.model_path)
            
            sys.setrecursionlimit(old_limit)
            
            print(f"✓ Model loaded successfully!")
            
        except Exception as e:
            sys.setrecursionlimit(1000)
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, image_input):
        """Perform inference on a single image"""
        preprocessed, original = load_and_preprocess_image(image_input)
        pred_mask = self.model.predict(preprocessed, verbose=0)[0].squeeze()
        
        original_size = (original.shape[1], original.shape[0])
        binary_mask, confidence_map = postprocess_mask(
            pred_mask, 
            threshold=cfg.CONFIDENCE_THRESHOLD,
            target_size=original_size
        )
        
        metrics = self._calculate_metrics(binary_mask, confidence_map)
        
        return {
            'binary_mask': binary_mask,
            'confidence_map': confidence_map,
            'original_image': original,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, binary_mask, confidence_map):
        """Calculate detection statistics"""
        total_pixels = binary_mask.size
        detected_pixels = np.sum(binary_mask > 0)
        coverage_percentage = (detected_pixels / total_pixels) * 100
        
        if detected_pixels > 0:
            avg_confidence = np.mean(confidence_map[binary_mask > 0])
        else:
            avg_confidence = 0.0
        
        max_confidence = np.max(confidence_map)
        
        return {
            'coverage_percentage': coverage_percentage,
            'detected_pixels': detected_pixels,
            'total_pixels': total_pixels,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'has_spill': detected_pixels > 0
        }


_detector_instance = None

def get_detector(model_path=cfg.MODEL_PATH):
    """Get or create detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = OilSpillDetector(model_path)
    return _detector_instance