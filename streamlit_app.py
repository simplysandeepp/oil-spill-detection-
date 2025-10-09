# ============================================================================
# APP.PY - Streamlit Web Application for Oil Spill Detection
# ============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.inference import get_detector
from utils.visualization import (create_overlay, create_confidence_heatmap, 
                                 add_metrics_overlay)
from utils.preprocessing import validate_image
import config.config as cfg


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=cfg.PAGE_TITLE,
    page_icon=cfg.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Apply it
local_css("styles/style.css")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ff4444;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ff4444;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-transform: uppercase;
    }
    
    .status-detected {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .status-clean {
        color: #00cc66;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load model with caching"""
    try:
        detector = get_detector(cfg.MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()


def process_image(detector, uploaded_file):
    """Process uploaded image and return results"""
    try:
        # Convert uploaded file to PIL Image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Validate image
        is_valid, message = validate_image(image)
        if not is_valid:
            st.error(f"❌ Invalid image: {message}")
            return None
        
        # Run inference
        with st.spinner('🔍 Analyzing image...'):
            results = detector.predict(image)
        
        return results
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown(
        f'<div class="main-header">{cfg.PAGE_ICON} Oil Spill Detection System</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses **Deep Learning** to detect oil spills in satellite/aerial imagery.
        
        **How it works:**
        1. Upload an image
        2. AI model analyzes it
        3. Get instant detection results
        
        **Model:** Enhanced U-Net with Attention Gates
        
        **Accuracy:** ~95%
        """)
        
        st.divider()
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=cfg.CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Adjust sensitivity of detection"
        )
        
        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=cfg.OVERLAY_ALPHA,
            step=0.1,
            help="Adjust visibility of detected regions"
        )
        
        st.divider()
        
        st.header("📊 Statistics")
        if 'total_processed' not in st.session_state:
            st.session_state.total_processed = 0
        if 'total_detections' not in st.session_state:
            st.session_state.total_detections = 0
            
        st.metric("Images Processed", st.session_state.total_processed)
        st.metric("Spills Detected", st.session_state.total_detections)
    
    # Main content
    st.markdown('<div class="info-box">📤 Upload an image to begin analysis</div>', 
                unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help=f"Maximum file size: {cfg.MAX_FILE_SIZE}MB"
    )
    
    if uploaded_file is not None:
        # Load model
        detector = load_model()
        
        # Update confidence threshold if changed
        cfg.CONFIDENCE_THRESHOLD = confidence_threshold
        cfg.OVERLAY_ALPHA = overlay_alpha
        
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
        
        # Process image
        results = process_image(detector, uploaded_file)
        
        if results is not None:
            # Update statistics
            st.session_state.total_processed += 1
            if results['metrics']['has_spill']:
                st.session_state.total_detections += 1
            
            # Create visualizations
            overlay = create_overlay(
                results['original_image'],
                results['binary_mask'],
                alpha=overlay_alpha
            )
            
            heatmap = create_confidence_heatmap(results['confidence_map'])
            
            # Display results
            with col2:
                st.subheader("🎯 Detection Result")
                
                # Detection status
                if results['metrics']['has_spill']:
                    st.markdown(
                        '<p class="status-detected">⚠️ OIL SPILL DETECTED</p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<p class="status-clean">✅ NO OIL SPILL DETECTED</p>',
                        unsafe_allow_html=True
                    )
                
                st.image(overlay, use_column_width=True, 
                        caption="Red overlay shows detected oil spill regions")
            
            # Metrics display
            st.divider()
            st.subheader("📊 Detection Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Coverage</div>
                    <div class="metric-value">{results['metrics']['coverage_percentage']:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-value">{results['metrics']['avg_confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Max Confidence</div>
                    <div class="metric-value">{results['metrics']['max_confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Pixels</div>
                    <div class="metric-value">{results['metrics']['detected_pixels']:,}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional visualizations
            st.divider()
            st.subheader("🔍 Detailed Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Binary Mask", "Confidence Heatmap", "Raw Data"])
            
            with tab1:
                st.image(results['binary_mask'], 
                        caption="Binary segmentation mask (white = detected)",
                        use_column_width=True, clamp=True)
            
            with tab2:
                st.image(heatmap,
                        caption="Model confidence heatmap (red = high confidence)",
                        use_column_width=True)
            
            with tab3:
                st.json({
                    'has_spill': bool(results['metrics']['has_spill']),
                    'coverage_percentage': float(results['metrics']['coverage_percentage']),
                    'detected_pixels': int(results['metrics']['detected_pixels']),
                    'total_pixels': int(results['metrics']['total_pixels']),
                    'avg_confidence': float(results['metrics']['avg_confidence']),
                    'max_confidence': float(results['metrics']['max_confidence']),
                    'threshold_used': float(confidence_threshold)
                })
            
            # Download results
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert overlay to bytes for download
                overlay_pil = Image.fromarray(overlay)
                import io
                buf = io.BytesIO()
                overlay_pil.save(buf, format='PNG')
                
                st.download_button(
                    label="📥 Download Detection Overlay",
                    data=buf.getvalue(),
                    file_name="oil_spill_detection_overlay.png",
                    mime="image/png"
                )
            
            with col2:
                # Download heatmap
                heatmap_pil = Image.fromarray(heatmap)
                buf2 = io.BytesIO()
                heatmap_pil.save(buf2, format='PNG')
                
                st.download_button(
                    label="📥 Download Confidence Heatmap",
                    data=buf2.getvalue(),
                    file_name="oil_spill_confidence_heatmap.png",
                    mime="image/png"
                )


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()
