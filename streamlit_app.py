# ============================================================================
# APP_PREMIUM.PY - Streamlit Web Application (Premium Light Bluish UI)
# This file preserves your model logic and replaces only the web UI/UX parts.
# Do NOT change the inference utilities - they are imported and used as-is.
# ============================================================================

import streamlit as st
import numpy as np
from PIL import Image
import sys
import os
import io
import base64
import pandas as pd
from datetime import datetime
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.inference import get_detector
from utils.visualization import (create_overlay, create_confidence_heatmap,
                                 add_metrics_overlay)
from utils.preprocessing import validate_image
import config.config as cfg

# ------------------------ PAGE CONFIG --------------------------------------
st.set_page_config(
    page_title=cfg.PAGE_TITLE or "AI Oil Spill Detection",
    page_icon=cfg.PAGE_ICON or "🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------ PREMIUM LIGHT BLUISH STYLES ----------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    scroll-behavior: smooth;
}

/* Main container background - soft watery gradient */
.stApp {
    background: linear-gradient(135deg, #E3F4F4 0%, #F0F9FF 25%, #E0F2FE 50%, #DBEAFE 75%, #E3F4F4 100%);
    background-attachment: fixed;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ================ HERO SECTION ================ */
.hero-section {
    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(240,249,255,0.95) 100%);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 60px 40px;
    margin-bottom: 40px;
    box-shadow: 0 20px 60px rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.1);
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.8s ease-out;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(14,165,233,0.03) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-section h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #0369A1 0%, #0EA5E9 50%, #06B6D4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 20px;
    letter-spacing: -1px;
    position: relative;
    z-index: 1;
}

.hero-section .subtitle {
    font-size: 1.3rem;
    color: #0C4A6E;
    margin-bottom: 15px;
    font-weight: 400;
    line-height: 1.6;
    position: relative;
    z-index: 1;
}

.hero-section .author {
    font-size: 1rem;
    color: #0891B2;
    font-weight: 500;
    margin-bottom: 30px;
    position: relative;
    z-index: 1;
}

.cta-button {
    display: inline-block;
    background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%);
    color: white;
    padding: 16px 40px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 10px 30px rgba(14, 165, 233, 0.3);
    transition: all 0.3s ease;
    position: relative;
    z-index: 1;
    border: none;
    cursor: pointer;
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(14, 165, 233, 0.4);
}

/* ================ ABOUT SECTION ================ */
.about-section {
    margin: 50px 0;
}

.about-title {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #0369A1;
    margin-bottom: 50px;
    animation: fadeInUp 0.8s ease-out 0.2s both;
}

.cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 30px;
    margin-bottom: 40px;
}

.info-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,249,255,0.9) 100%);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 35px 30px;
    box-shadow: 0 15px 40px rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.1);
    transition: all 0.4s ease;
    animation: fadeInUp 0.8s ease-out both;
}

.info-card:nth-child(1) { animation-delay: 0.3s; }
.info-card:nth-child(2) { animation-delay: 0.4s; }
.info-card:nth-child(3) { animation-delay: 0.5s; }

.info-card:hover {
    transform: translateY(-10px);
    box-shadow: 0 20px 50px rgba(14, 165, 233, 0.15);
    border-color: rgba(14, 165, 233, 0.3);
}

.info-card .icon {
    font-size: 3rem;
    margin-bottom: 20px;
    display: block;
}

.info-card h3 {
    font-family: 'Poppins', sans-serif;
    font-size: 1.5rem;
    color: #0369A1;
    margin-bottom: 15px;
    font-weight: 600;
}

.info-card p {
    color: #0C4A6E;
    line-height: 1.7;
    font-size: 1rem;
}

/* ================ UPLOAD SECTION ================ */
.upload-section {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,249,255,0.9) 100%);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 40px;
    margin: 40px 0;
    box-shadow: 0 20px 60px rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.15);
    animation: fadeInUp 0.8s ease-out 0.6s both;
}

.section-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    color: #0369A1;
    margin-bottom: 10px;
    font-weight: 600;
}

.section-subtitle {
    color: #0891B2;
    margin-bottom: 30px;
    font-size: 1rem;
}

/* Streamlit file uploader styling */
[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(240,249,255,0.8) 0%, rgba(224,242,254,0.8) 100%);
    border-radius: 16px;
    padding: 20px;
    border: 2px dashed rgba(14, 165, 233, 0.3);
    transition: all 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(14, 165, 233, 0.6);
    background: linear-gradient(135deg, rgba(240,249,255,1) 0%, rgba(224,242,254,1) 100%);
}

/* Sliders */
.stSlider {
    padding: 10px 0;
}

.stSlider > label {
    color: #0369A1 !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-weight: 600;
    font-size: 1rem;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.25);
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 30px rgba(14, 165, 233, 0.35);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #0891B2 0%, #0EA5E9 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: 500;
    box-shadow: 0 6px 15px rgba(8, 145, 178, 0.2);
    transition: all 0.3s ease;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(8, 145, 178, 0.3);
}

/* ================ RESULTS SECTION ================ */
.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 25px;
    margin: 30px 0;
}

.result-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,249,255,0.9) 100%);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 15px 40px rgba(14, 165, 233, 0.08);
    border: 1px solid rgba(14, 165, 233, 0.1);
    transition: all 0.3s ease;
}

.result-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 50px rgba(14, 165, 233, 0.15);
}

.result-card h3 {
    font-family: 'Poppins', sans-serif;
    color: #0369A1;
    margin-bottom: 20px;
    font-size: 1.4rem;
    font-weight: 600;
}

.result-card img {
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}

.metric-box {
    background: linear-gradient(135deg, rgba(224,242,254,0.5) 0%, rgba(240,249,255,0.5) 100%);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
    border: 1px solid rgba(14, 165, 233, 0.1);
}

.metric-value {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0369A1;
    margin-bottom: 5px;
}

.metric-label {
    color: #0891B2;
    font-size: 0.9rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ================ TABS ================ */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: rgba(240,249,255,0.5);
    padding: 8px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #0891B2;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%);
    color: white;
}

/* ================ STATS BANNER ================ */
.stats-banner {
    background: linear-gradient(135deg, rgba(14,165,233,0.1) 0%, rgba(6,182,212,0.1) 100%);
    border-radius: 16px;
    padding: 20px;
    margin: 30px 0;
    text-align: center;
    border: 1px solid rgba(14, 165, 233, 0.2);
}

.stats-banner .stat {
    display: inline-block;
    margin: 0 30px;
}

.stats-banner .stat-number {
    font-family: 'Poppins', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #0369A1;
}

.stats-banner .stat-label {
    color: #0891B2;
    font-size: 0.95rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ================ FOOTER ================ */
.footer {
    text-align: center;
    padding: 40px 20px;
    margin-top: 60px;
    border-top: 1px solid rgba(14, 165, 233, 0.1);
    color: #0891B2;
    font-size: 1rem;
}

.footer strong {
    color: #0369A1;
    font-weight: 600;
}

/* ================ ANIMATIONS ================ */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* ================ DATABASE SECTION ================ */
.database-section {
    background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,249,255,0.9) 100%);
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 40px;
    margin: 40px 0;
    box-shadow: 0 20px 60px rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.15);
}

.database-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.db-stats {
    display: flex;
    gap: 20px;
    margin-bottom: 25px;
}

.db-stat-box {
    background: linear-gradient(135deg, rgba(224,242,254,0.5) 0%, rgba(240,249,255,0.5) 100%);
    border-radius: 12px;
    padding: 15px 20px;
    border: 1px solid rgba(14, 165, 233, 0.1);
    flex: 1;
    text-align: center;
}

.db-stat-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #0369A1;
}

.db-stat-label {
    color: #0891B2;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 5px;
}

/* Streamlit dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 8px 20px rgba(14, 165, 233, 0.08);
}

/* ================ RESPONSIVE ================ */
@media (max-width: 768px) {
    .hero-section h1 {
        font-size: 2.5rem;
    }
    
    .hero-section .subtitle {
        font-size: 1.1rem;
    }
    
    .about-title {
        font-size: 2rem;
    }
    
    .cards-container {
        grid-template-columns: 1fr;
    }
    
    .results-grid {
        grid-template-columns: 1fr;
    }
}

/* ================ LOADING SPINNER ================ */
.stSpinner > div {
    border-top-color: #0EA5E9 !important;
}

/* ================ SUCCESS/ERROR MESSAGES ================ */
.stAlert {
    border-radius: 12px;
    border: none;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ------------------------ DATABASE HELPERS ---------------------------------
def init_database():
    """Initialize the detection records database in session state"""
    if 'detection_records' not in st.session_state:
        st.session_state.detection_records = []

def add_detection_record(filename, has_spill, coverage_pct, avg_confidence, max_confidence, detected_pixels):
    """Add a new detection record to the database"""
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'filename': filename,
        'result': 'Spill Detected ✅' if has_spill else 'No Spill ❌',
        'coverage_%': round(coverage_pct, 2),
        'avg_confidence': round(avg_confidence * 100, 1),
        'max_confidence': round(max_confidence * 100, 1),
        'detected_pixels': detected_pixels
    }
    st.session_state.detection_records.insert(0, record)  # Add to beginning
    
    # Keep only last 50 records to avoid memory issues
    if len(st.session_state.detection_records) > 50:
        st.session_state.detection_records = st.session_state.detection_records[:50]

def get_records_dataframe():
    """Convert records to pandas DataFrame"""
    if not st.session_state.detection_records:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.detection_records)


# ------------------------ HELPERS & MODEL (UNCHANGED LOGIC) ----------------
@st.cache_resource
def load_model():
    """Load the model - UNCHANGED"""
    try:
        detector = get_detector(cfg.MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()


def process_image(detector, uploaded_file):
    """Process uploaded image and return results - UNCHANGED"""
    try:
        image = Image.open(uploaded_file).convert('RGB')

        is_valid, message = validate_image(image)
        if not is_valid:
            st.error(f"❌ Invalid image: {message}")
            return None

        with st.spinner('🔍 Analyzing image with AI...'):
            results = detector.predict(image)

        return results
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None


def image_to_bytes(img: Image.Image, fmt="PNG"):
    """Convert PIL image to bytes for downloads - UNCHANGED"""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ------------------------ MAIN UI -----------------------------------------
def main():
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'total_processed' not in st.session_state:
        st.session_state.total_processed = 0
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    if 'scroll_to_upload' not in st.session_state:
        st.session_state.scroll_to_upload = False

    # ==================== HERO SECTION ====================
    st.markdown("""
    <div class="hero-section">
        <h1 style="text-align:center;">🌊 AI-Powered Oil Spill Detection System</h1>
        <p class="subtitle" style="text-align:justify;">
            Our system leverages cutting-edge Deep Learning and AI technologies to detect and analyze oil spills from satellite and aerial imagery with high speed and accuracy. Designed for environmental monitoring agencies, researchers, and response teams, it transforms raw imagery into actionable insights, helping protect marine ecosystems and coastal communities.
        </p>
        <p class="author">Developed by <strong>Sandeep Prajapati ❣️</strong></p>
    """, unsafe_allow_html=True)

    # ==================== ABOUT SECTION ====================
    st.markdown('<h2 class="about-title">Understanding Oil Spill Detection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="cards-container">
        <div class="info-card">
            <span class="icon">🌊</span>
            <h3>What is an Oil Spill?</h3>
            <p>An oil spill is the release of liquid petroleum hydrocarbons into the environment, especially marine areas. These incidents can devastate marine ecosystems, kill wildlife, contaminate water sources, and cause long-lasting environmental damage that affects coastal communities and economies.</p>
        </div>
        <div class="info-card">
            <span class="icon">🛰️</span>
            <h3>Why Early Detection Matters</h3>
            <p>Early detection is critical for effective response. The faster we identify oil spills, the quicker cleanup crews can be deployed, containment strategies can be implemented, and environmental damage can be minimized. Time is the most crucial factor in spill response operations.</p>
        </div>
        <div class="info-card">
            <span class="icon">⚙️</span>
            <h3>How AI Helps</h3>
            <p>Deep learning models analyze satellite and aerial imagery at scale, identifying potential oil spills with high accuracy. AI can process thousands of images in minutes, detecting patterns invisible to the human eye, and enabling rapid response to environmental threats worldwide.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    # ==================== SCROLL TO UPLOAD ====================

    # ==================== UPLOAD SECTION ====================
    st.markdown('<h2 class="section-title">📤 Upload & Analyze Imagery</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">Upload satellite or aerial imagery to detect potential oil spills using our AI model</p>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="For best results, use high-resolution satellite or aerial imagery"
    )

    # Controls in columns
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=cfg.CONFIDENCE_THRESHOLD,
            step=0.01,
            help="Minimum confidence level for detection"
        )
    with col2:
        overlay_alpha = st.slider(
            "Overlay Transparency",
            min_value=0.0,
            max_value=1.0,
            value=cfg.OVERLAY_ALPHA,
            step=0.05,
            help="Transparency of the detection overlay"
        )

    # Action buttons
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        detect_button = st.button("🔍 Detect Oil Spill", type="primary", use_container_width=True)
    with col_btn2:
        clear_button = st.button("♻️ Clear Results", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Handle clear button
    if clear_button:
        st.session_state.total_processed = 0
        st.session_state.total_detections = 0
        st.rerun()

    # Update config values
    cfg.CONFIDENCE_THRESHOLD = confidence_threshold
    cfg.OVERLAY_ALPHA = overlay_alpha

    # ==================== RESULTS SECTION ====================
    if uploaded_file is not None and detect_button:
        detector = load_model()
        results = process_image(detector, uploaded_file)

        if results is not None:
            # Update session stats
            st.session_state.total_processed += 1
            if results['metrics']['has_spill']:
                st.session_state.total_detections += 1

            # Add record to database
            add_detection_record(
                filename=uploaded_file.name,
                has_spill=results['metrics']['has_spill'],
                coverage_pct=results['metrics']['coverage_percentage'],
                avg_confidence=results['metrics']['avg_confidence'],
                max_confidence=results['metrics']['max_confidence'],
                detected_pixels=results['metrics']['detected_pixels']
            )

            # Create visualizations
            overlay = create_overlay(
                results['original_image'],
                results['binary_mask'],
                alpha=overlay_alpha
            )
            heatmap = create_confidence_heatmap(results['confidence_map'])

            # Success message
            if results['metrics']['has_spill']:
                st.success(f"✅ Oil spill detected! Coverage: {results['metrics']['coverage_percentage']:.2f}%")
            else:
                st.info("ℹ️ No oil spill detected in this image")

            # Results grid
            st.markdown('<div class="results-grid">', unsafe_allow_html=True)

            # Column 1: Detection Overlay
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3>Detection Overlay</h3>', unsafe_allow_html=True)
                st.image(overlay, use_container_width=True)
                ov_pil = Image.fromarray(overlay) if isinstance(overlay, np.ndarray) else overlay
                st.download_button(
                    "📥 Download Overlay",
                    data=image_to_bytes(ov_pil),
                    file_name='oil_spill_overlay.png',
                    mime='image/png',
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Column 2: Confidence Heatmap
            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3>Confidence Heatmap</h3>', unsafe_allow_html=True)
                st.image(heatmap, use_container_width=True)
                hm_pil = Image.fromarray(heatmap) if isinstance(heatmap, np.ndarray) else heatmap
                st.download_button(
                    "📥 Download Heatmap",
                    data=image_to_bytes(hm_pil),
                    file_name='oil_spill_heatmap.png',
                    mime='image/png',
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # Column 3: Metrics
            with col3:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<h3>Detection Metrics</h3>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['coverage_percentage']:.2f}%</div>
                    <div class="metric-label">Coverage Area</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['avg_confidence']:.1%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['max_confidence']:.1%}</div>
                    <div class="metric-label">Max Confidence</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{results['metrics']['detected_pixels']:,}</div>
                    <div class="metric-label">Detected Pixels</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Additional details in tabs
            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["📊 Binary Mask", "📋 Raw JSON Data", "📈 Analysis Summary"])

            with tab1:
                st.image(
                    results['binary_mask'],
                    caption='Binary segmentation mask (white = oil spill detected)',
                    use_container_width=True
                )

            with tab2:
                st.json({
                    'detection_status': 'Spill Detected' if results['metrics']['has_spill'] else 'No Spill',
                    'coverage_percentage': float(results['metrics']['coverage_percentage']),
                    'detected_pixels': int(results['metrics']['detected_pixels']),
                    'total_pixels': int(results['metrics']['total_pixels']),
                    'average_confidence': float(results['metrics']['avg_confidence']),
                    'maximum_confidence': float(results['metrics']['max_confidence']),
                    'threshold_used': float(confidence_threshold),
                    'overlay_alpha': float(overlay_alpha)
                })

            with tab3:
                st.markdown(f"""
                ### Analysis Summary
                
                **Detection Result:** {'✅ Oil Spill Detected' if results['metrics']['has_spill'] else '❌ No Oil Spill Detected'}
                
                **Coverage Analysis:**
                - Total area analyzed: {results['metrics']['total_pixels']:,} pixels
                - Contaminated area: {results['metrics']['detected_pixels']:,} pixels
                - Coverage percentage: {results['metrics']['coverage_percentage']:.2f}%
                
                **Confidence Metrics:**
                - Average confidence: {results['metrics']['avg_confidence']:.1%}
                - Maximum confidence: {results['metrics']['max_confidence']:.1%}
                - Detection threshold: {confidence_threshold:.1%}
                
                **Recommendations:**
                {('- Immediate response required for cleanup operations' if results['metrics']['coverage_percentage'] > 5 else '- Monitor the area for potential expansion') if results['metrics']['has_spill'] else '- Continue routine monitoring'}
                """)

    # ==================== STATISTICS BANNER ====================
    st.markdown(f"""
    <div class="stats-banner">
        <div class="stat">
            <div class="stat-number">{st.session_state.total_processed}</div>
            <div class="stat-label">Images Analyzed</div>
        </div>
        <div class="stat">
            <div class="stat-number">{st.session_state.total_detections}</div>
            <div class="stat-label">Spills Detected</div>
        </div>
        <div class="stat">
            <div class="stat-number">{(st.session_state.total_detections / max(st.session_state.total_processed, 1) * 100):.1f}%</div>
            <div class="stat-label">Detection Rate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== LIVE DATABASE SECTION ====================
    if st.session_state.detection_records:
        st.markdown('<div class="database-section">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="database-header">
            <h2 class="section-title">📊 Live Detection Database</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Database stats
        df = get_records_dataframe()
        total_records = len(df)
        spills_found = len(df[df['result'].str.contains('✅')])
        avg_coverage = df['coverage_%'].mean() if not df.empty else 0
        
        st.markdown(f"""
        <div class="db-stats">
            <div class="db-stat-box">
                <div class="db-stat-value">{total_records}</div>
                <div class="db-stat-label">Total Records</div>
            </div>
            <div class="db-stat-box">
                <div class="db-stat-value">{spills_found}</div>
                <div class="db-stat-label">Spills Found</div>
            </div>
            <div class="db-stat-box">
                <div class="db-stat-value">{avg_coverage:.1f}%</div>
                <div class="db-stat-label">Avg Coverage</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "filename": st.column_config.TextColumn("Image File", width="medium"),
                "result": st.column_config.TextColumn("Result", width="small"),
                "coverage_%": st.column_config.NumberColumn("Coverage %", format="%.2f"),
                "avg_confidence": st.column_config.NumberColumn("Avg Conf %", format="%.1f"),
                "max_confidence": st.column_config.NumberColumn("Max Conf %", format="%.1f"),
                "detected_pixels": st.column_config.NumberColumn("Pixels", format="%d")
            }
        )
        
        # Export options
        col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
        with col_export1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f'detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                use_container_width=True
            )
        with col_export2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                "📥 Export JSON",
                data=json_data,
                file_name=f'detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json',
                use_container_width=True
            )
        with col_export3:
            if st.button("🗑️ Clear All Records", use_container_width=True):
                st.session_state.detection_records = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="database-section">
            <h2 class="section-title">📊 Live Detection Database</h2>
            <p style="text-align: center; color: #0891B2; padding: 40px 0;">
                No records yet. Upload and analyze images to see detection history here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ==================== FOOTER ====================
    st.markdown("""
    <div class="footer">
        Built with ❤️ by <strong>Sandeep Prajapati</strong> | Powered by AI & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()