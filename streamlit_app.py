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
import requests

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

# ---------- small helper to load lottie json ----------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None

# ------------------------ STYLES (PREMIUM LIGHT BLUISH) --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Page background: soft watery gradient */
.block-container { padding-top: 1.5rem; padding-bottom: 2.5rem; }
.reportview-container, .main, .block-container{
    background: linear-gradient(180deg,#f7fbff 0%, #eaf7ff 40%, #f0fbff 100%);
    color: #04263a;
}

/* Hero */
.hero {
    background: linear-gradient(90deg, rgba(226,249,255,0.9) 0%, rgba(227,242,255,0.9) 100%);
    border-radius: 16px;
    padding: 28px 26px;
    margin-bottom: 18px;
    border: 1px solid rgba(3,105,161,0.06);
}
.hero h1 { margin:0; font-size:34px; color:#023e58; letter-spacing: -0.5px }
.hero p { margin:8px 0 0 0; color:#08536b }
.hero .cta { margin-top:12px }

/* Info cards */
.info-row { display:flex; gap:14px; margin-top:16px }
.info-card { flex:1; background: white; border-radius:12px; padding:12px; box-shadow: 0 8px 20px rgba(3,60,90,0.04); border: 1px solid rgba(3,60,90,0.03) }
.info-card h4 { margin:0; color:#024a66 }
.info-card p { margin:6px 0 0 0; color:#256076; font-size:14px }

/* Uploader card */
.uploader-card { background: linear-gradient(180deg, rgba(255,255,255,1), rgba(245,255,ff,1)); border-radius:14px; padding:18px; border: 1px solid rgba(3,105,161,0.06); box-shadow: 0 12px 30px rgba(3,105,161,0.06) }
.uploader-cta { display:flex; gap:10px; align-items:center }
.btn-primary { background: linear-gradient(90deg,#0ea5a7,#06b6d4); color:white; padding:10px 14px; border-radius:10px; font-weight:700 }
.btn-ghost { background:transparent; color:#04506a; padding:8px 12px; border-radius:10px; border:1px solid rgba(4,80,106,0.08) }

/* Result cards */
.result-grid { display:flex; gap:14px; margin-top:14px }
.result-card { flex:1; background:white; border-radius:12px; padding:12px; border:1px solid rgba(3,60,90,0.03); box-shadow: 0 10px 24px rgba(3,60,90,0.04) }
.metric { font-weight:700; font-size:18px; color:#024a66 }
.metric-label { color:#2b6b7e; font-size:12px; text-transform:uppercase }

/* Tabs and small styles */
.small-muted { color:#2b6b7e; font-size:13px }
.footer { text-align:center; color:#0b4b5f; margin-top:22px; font-size:13px }

/* Make native buttons rounded */
div.stButton > button { border-radius: 10px; padding:8px 10px }

/* Responsive handling for narrow screens */
@media (max-width: 760px) {
    .info-row { flex-direction:column }
    .result-grid { flex-direction:column }
}
</style>
""", unsafe_allow_html=True)

# ------------------------ LOTTIE ASSETS -----------------------------------
# Using friendly public Lottie files for ocean/ai animations
LOTTIE_WAVE = load_lottie_url("https://assets6.lottiefiles.com/packages/lf20_jmgekfqg.json")
LOTTIE_RADAR = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_jz8bwz7r.json")

# ------------------------ HELPERS & MODEL (UNCHANGED LOGIC) ----------------
@st.cache_resource
def load_model():
    try:
        detector = get_detector(cfg.MODEL_PATH)
        return detector
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()


def process_image(detector, uploaded_file):
    """Process uploaded image and return results (keeps your original behavior)"""
    try:
        image = Image.open(uploaded_file).convert('RGB')

        is_valid, message = validate_image(image)
        if not is_valid:
            st.error(f"❌ Invalid image: {message}")
            return None

        with st.spinner('🔍 Analyzing image...'):
            results = detector.predict(image)

        return results
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None

# convert PIL image to bytes for downloads
def image_to_bytes(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

# ------------------------ MAIN UI -----------------------------------------
def main():
    # Hero section with Lottie on the right
    left, right = st.columns([2, 1])
    with left:
        st.markdown(
            """
            <div class='hero'>
                <div style='display:flex; align-items:center; justify-content:space-between'>
                    <div style='flex:1'>
                        <h1>AI-Powered Oil Spill Detection</h1>
                        <p>Detect and analyze oil spills from satellite & aerial imagery instantly —
                        explainable overlays, confidence heatmaps, and usable metrics for responders.</p>
                        <div class='hero cta'>
                            <button class='btn-primary' onclick="window.streamlitRun('show_uploader')">Try Demo</button>
                            <button class='btn-ghost' onclick="window.streamlitRun('learn_more')">Learn More</button>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with right:
        # show Lottie animation if available using components.html fallback
        if LOTTIE_WAVE:
            import streamlit.components.v1 as components
            components.html(f"<div style='width:100%;height:240px'></div>", height=240)
            st_lottie_placeholder = st.empty()
            try:
                from streamlit_lottie import st_lottie
                st_lottie(LOTTIE_WAVE, height=220)
            except Exception:
                # If streamlit_lottie not installed, show static img fallback
                st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800&q=80", use_column_width=True)

    # Small spacer
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Info / What is an oil spill section
    st.markdown("""
    <div class='info-row'>
        <div class='info-card'>
            <h4>What is an Oil Spill?</h4>
            <p>An oil spill is the release of liquid petroleum hydrocarbons into the environment, especially marine areas.
            It harms ecosystems, wildlife, and coastal economies.</p>
        </div>
        <div class='info-card'>
            <h4>Why Detect Early?</h4>
            <p>Early detection enables faster response, reduces environmental impact, and helps route cleanup resources.</p>
        </div>
        <div class='info-card'>
            <h4>How AI Helps</h4>
            <p>Deep learning models analyze satellite imagery at scale to find likely spill regions quickly and reliably.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Uploader card - keeps model inference intact and rich UI controls
    st.markdown("<div class='uploader-card'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload Image & Detect")
    st.markdown("<div class='small-muted'>Supported: JPG, PNG — for best results provide high-res satellite imagery.</div>", unsafe_allow_html=True)

    # Controls row
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    with c2:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, cfg.CONFIDENCE_THRESHOLD, 0.01)
    with c3:
        overlay_alpha = st.slider("Overlay Transparency", 0.0, 1.0, cfg.OVERLAY_ALPHA, 0.05)

    # Action buttons
    action_col1, action_col2 = st.columns([1,1])
    with action_col1:
        detect_btn = st.button("🔎 Detect Oil Spill", key='detect')
    with action_col2:
        clear_btn = st.button("♻️ Reset", key='reset')

    st.markdown("</div>", unsafe_allow_html=True)

    # handle reset
    if clear_btn:
        if 'total_processed' in st.session_state:
            st.session_state.total_processed = 0
        if 'total_detections' in st.session_state:
            st.session_state.total_detections = 0
        st.experimental_rerun()

    # When user presses detect or uploads and wants automatic detection
    run_inference = False
    if detect_btn and uploaded_file is not None:
        run_inference = True

    # Also allow auto-run if user uploaded & wants immediate (optional)
    if uploaded_file is not None and 'auto_run' in cfg.__dict__ and cfg.auto_run:
        run_inference = True

    # Update config values but keep model logic untouched
    cfg.CONFIDENCE_THRESHOLD = confidence_threshold
    cfg.OVERLAY_ALPHA = overlay_alpha

    # Show sample / placeholder when no file
    if uploaded_file is None:
        st.markdown("<div class='result-grid'>", unsafe_allow_html=True)
        st.markdown("<div class='result-card'><h3 style='margin-top:6px'>Try a sample</h3><p class='small-muted'>You can test with any aerial/satellite image. Use the Detect button once uploaded.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='result-card'><h3 style='margin-top:6px'>Explainability</h3><p class='small-muted'>The app provides both segmentation overlays and confidence heatmaps for analysis.</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='result-card'><h3 style='margin-top:6px'>Export</h3><p class='small-muted'>Download overlay and heatmap PNGs suitable for reports.</p></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # If file present & user requested detection
    if uploaded_file is not None and run_inference:
        detector = load_model()

        results = process_image(detector, uploaded_file)
        if results is not None:
            # session stats
            if 'total_processed' not in st.session_state:
                st.session_state.total_processed = 0
            if 'total_detections' not in st.session_state:
                st.session_state.total_detections = 0

            st.session_state.total_processed += 1
            if results['metrics']['has_spill']:
                st.session_state.total_detections += 1

            # visuals
            overlay = create_overlay(results['original_image'], results['binary_mask'], alpha=overlay_alpha)
            heatmap = create_confidence_heatmap(results['confidence_map'])

            # Results layout
            st.markdown("<div class='result-grid'>", unsafe_allow_html=True)

            # left: overlay image
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Detection Overlay</h3>", unsafe_allow_html=True)
            st.image(overlay, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # middle: heatmap
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Confidence Heatmap</h3>", unsafe_allow_html=True)
            st.image(heatmap, use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # right: metrics & downloads
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<h3>Metrics</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>{results['metrics']['coverage_percentage']:.2f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Coverage</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>{results['metrics']['avg_confidence']:.1%}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Average Confidence</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric'>{results['metrics']['max_confidence']:.1%}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-label'>Max Confidence</div>", unsafe_allow_html=True)

            # downloads
            ov_pil = Image.fromarray(overlay) if isinstance(overlay, np.ndarray) else overlay
            hm_pil = Image.fromarray(heatmap) if isinstance(heatmap, np.ndarray) else heatmap
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("📥 Download Overlay", data=image_to_bytes(ov_pil), file_name='overlay.png', mime='image/png')
            with col_b:
                st.download_button("📥 Download Heatmap", data=image_to_bytes(hm_pil), file_name='heatmap.png', mime='image/png')

            st.markdown("</div>", unsafe_allow_html=True)

            # detailed tabs below
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(["Binary Mask", "Raw JSON", "Confidence Summary"])
            with tab1:
                st.image(results['binary_mask'], caption='Binary mask (white = detected)', use_column_width=True, clamp=True)
            with tab2:
                st.json({
                    'has_spill': bool(results['metrics']['has_spill']),
                    'coverage_percentage': float(results['metrics']['coverage_percentage']),
                    'detected_pixels': int(results['metrics']['detected_pixels']),
                    'total_pixels': int(results['metrics']['total_pixels']),
                    'avg_confidence': float(results['metrics']['avg_confidence']),
                    'max_confidence': float(results['metrics']['max_confidence']),
                    'threshold_used': float(confidence_threshold)
                })
            with tab1:
                pass

    # show live stats in a bottom strip
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>Images processed: {st.session_state.get('total_processed',0)} • Spills detected: {st.session_state.get('total_detections',0)}</div>", unsafe_allow_html=True)

    # footer
    st.markdown("<div class='footer'>Built with ❤️ by SimplySandeepp • Powered by AI & Streamlit</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
