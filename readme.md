# 🛢️ Oil Spill Detection System - Deployment Guide

A deep learning-powered web application for detecting oil spills in satellite/aerial imagery using Enhanced U-Net architecture with attention gates.

---

## 📋 Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Deployment Options](#deployment-options)
  - [Option 1: Local Deployment (VS Code)](#option-1-local-deployment-vs-code)
  - [Option 2: Google Colab Deployment](#option-2-google-colab-deployment)
  - [Option 3: Cloud Deployment](#option-3-cloud-deployment)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- **Real-time Oil Spill Detection**: Upload images and get instant analysis
- **Multiple Visualizations**: Binary masks, confidence heatmaps, overlays
- **Adjustable Sensitivity**: Configure detection threshold
- **Comprehensive Metrics**: Coverage percentage, confidence scores, pixel counts
- **Download Results**: Export detection overlays and heatmaps
- **Professional UI**: Clean, intuitive Streamlit interface

---

## 🔧 Prerequisites

- Python 3.8+ (3.10 recommended)
- At least 4GB RAM
- 2GB free disk space
- Your trained model file: `best_model.h5`

---

## 📦 Installation

### Step 1: Clone or Setup Project

```bash
# Create project directory
mkdir oil-spill-detection
cd oil-spill-detection

# Create folder structure
mkdir -p models utils config assets/sample_images notebooks
```

### Step 2: Copy Files

Place these files in the correct locations:
```
oil-spill-detection/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── README.md              # This file
├── config/
│   └── config.py          # Configuration settings
├── utils/
│   ├── __init__.py        # Empty file (create manually)
│   ├── preprocessing.py   # Image preprocessing
│   ├── inference.py       # Model inference
│   └── visualization.py   # Visualization utilities
└── models/
    └── best_model.h5      # Your trained model (COPY FROM COLAB)
```

**IMPORTANT**: Create empty `__init__.py` file in `utils/` directory:
```bash
touch utils/__init__.py
```

### Step 3: Copy Model from Colab

**From Google Colab:**
```python
# In your Colab notebook, run:
from google.colab import files

# Download the model
files.download('models/best_model.h5')
```

Then place the downloaded `best_model.h5` in your local `models/` folder.

### Step 4: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Deployment Options

---

## Option 1: Local Deployment (VS Code) - **RECOMMENDED**

### Advantages ✅
- Full control over environment
- Fast iteration and debugging
- No session timeouts
- Can run indefinitely
- Best for development and testing

### Setup Instructions

1. **Verify Installation**
   ```bash
   # Check if all files are in place
   ls -R
   
   # Should show:
   # app.py, requirements.txt, config/, utils/, models/best_model.h5
   ```

2. **Test Model Loading**
   ```bash
   # Create a quick test script
   python -c "from utils.inference import get_detector; print('Model loaded successfully!')"
   ```

3. **Run Streamlit App**
   ```bash
   streamlit run app.py
   ```

4. **Access Application**
   - Open browser and go to: `http://localhost:8501`
   - The app should load with the upload interface

5. **Customize Port (Optional)**
   ```bash
   streamlit run app.py --server.port 8080
   ```

### Troubleshooting Local Deployment

**Issue: "Module not found" errors**
```bash
# Solution: Ensure utils/__init__.py exists
touch utils/__init__.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue: Model fails to load**
```bash
# Check model file exists
ls -lh models/best_model.h5

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Issue: Memory errors**
```bash
# Reduce batch size in config.py
# Or use CPU-only TensorFlow (slower but uses less RAM)
pip install tensorflow-cpu
```

---

## Option 2: Google Colab Deployment

### Advantages ✅
- Free GPU access
- No local setup required
- Good for testing

### Limitations ⚠️
- Sessions timeout after ~12 hours
- Need to expose app via tunneling service
- Not suitable for production

### Setup Instructions

1. **Create New Colab Notebook**

2. **Install Streamlit and ngrok**
   ```python
   !pip install streamlit pyngrok
   ```

3. **Upload Project Files**
   ```python
   from google.colab import files
   
   # Upload all project files (app.py, config/, utils/, models/)
   # Or clone from GitHub if you've pushed your code
   ```

4. **Configure ngrok** (Get token from https://ngrok.com)
   ```python
   from pyngrok import ngrok
   
   # Set your ngrok auth token
   !ngrok authtoken YOUR_AUTH_TOKEN_HERE
   
   # Start ngrok tunnel
   public_url = ngrok.connect(8501)
   print(f'Streamlit app URL: {public_url}')
   ```

5. **Run Streamlit in Background**
   ```python
   # Run app in background
   !streamlit run app.py &>/dev/null &
   
   # Wait for it to start
   import time
   time.sleep(5)
   
   print(f'Access your app at: {public_url}')
   ```

6. **Keep Session Alive**
   ```python
   # Run this to prevent timeout (runs for 12 hours)
   import time
   for i in range(720):  # 720 * 60 seconds = 12 hours
       time.sleep(60)
       print(f"Running... {i+1}/720 minutes")
   ```

### Alternative: Colab with LocalTunnel
```python
# Install localtunnel
!npm install -g localtunnel

# Run streamlit in background
!streamlit run app.py &>/dev/null &

# Start tunnel
!npx localtunnel --port 8501
```

---

## Option 3: Cloud Deployment - **PRODUCTION**

For permanent, public deployment, use cloud platforms:

### 3A. Streamlit Cloud (Easiest) ⭐

**Advantages:**
- Free tier available
- Automatic deployments from GitHub
- Built-in SSL and scaling
- Zero configuration

**Steps:**

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin YOUR_REPO_URL
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Click "New app"
   - Select your GitHub repo
   - Set main file: `app.py`
   - Click "Deploy"

3. **Upload Model File**
   - Option 1: Use GitHub LFS for large files
   - Option 2: Load model from Google Drive URL
   - Option 3: Host model on Hugging Face Hub

**Note:** GitHub has 100MB file size limit. For large models (>100MB):

```python
# In inference.py, modify load_model to download from URL:

import requests
import os

def download_model_from_drive(gdrive_url, save_path):
    """Download model from Google Drive"""
    if not os.path.exists(save_path):
        print("Downloading model from Google Drive...")
        response = requests.get(gdrive_url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded!")

# Your Google Drive shareable link (make sure it's public)
MODEL_URL = "YOUR_GDRIVE_DIRECT_DOWNLOAD_LINK"
download_model_from_drive(MODEL_URL, cfg.MODEL_PATH)
```

### 3B. Heroku Deployment

1. **Create Heroku account** at https://heroku.com

2. **Create files:**

   `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```

   `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### 3C. AWS EC2 / Google Cloud / Azure

For full control, deploy on VPS:

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip nginx

# Clone your repo
git clone YOUR_REPO_URL
cd oil-spill-detection

# Install packages
pip3 install -r requirements.txt

# Run with screen (keeps running after logout)
screen -S streamlit
streamlit run app.py --server.port 8501

# Configure nginx as reverse proxy (optional)
```

---

## 🎯 Usage

### Basic Workflow

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Upload Image**
   - Click "Browse files" or drag & drop
   - Supported formats: JPG, JPEG, PNG
   - Max file size: 10MB (configurable)

3. **View Results**
   - Original image vs. Detection overlay
   - Coverage percentage
   - Confidence scores
   - Binary mask and heatmap

4. **Adjust Settings**
   - Use sidebar to adjust confidence threshold
   - Modify overlay transparency
   - View detection statistics

5. **Download Results**
   - Click download buttons to save:
     - Detection overlay (PNG)
     - Confidence heatmap (PNG)

---

## 🔧 Configuration

Edit `config/config.py` to customize:

```python
# Model parameters
IMG_HEIGHT = 256  # Change if your model uses different size
CONFIDENCE_THRESHOLD = 0.5  # Detection sensitivity

# Visualization
OVERLAY_ALPHA = 0.4  # Overlay transparency
SPILL_COLOR = [255, 0, 0]  # Detection color (R, G, B)

# Application
MAX_FILE_SIZE = 10  # Maximum upload size (MB)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Model file not found"**
```bash
# Solution: Verify model path
ls models/best_model.h5

# If missing, copy from Colab:
# In Colab: files.download('models/best_model.h5')
```

**2. "Import Error: No module named 'utils'"**
```bash
# Solution: Create __init__.py
touch utils/__init__.py
```

**3. "TensorFlow not found" or GPU errors**
```bash
# For CPU-only (no GPU):
pip uninstall tensorflow
pip install tensorflow-cpu

# For GPU support:
pip install tensorflow[and-cuda]  # TensorFlow 2.15+
```

**4. Port already in use**
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use different port
streamlit run app.py --server.port 8502
```

**5. Out of memory errors**
```python
# In config.py, add:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Or use CPU inference:
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

**6. Slow predictions**
- First prediction is always slower (model loading)
- Subsequent predictions are fast (cached)
- Consider using `SavedModel` format instead of `.h5` for faster loading

### Performance Optimization

**Convert .h5 to SavedModel (faster loading):**
```python
import tensorflow as tf

# Load .h5 model
model = tf.keras.models.load_model('models/best_model.h5', compile=False)

# Save as SavedModel
model.save('models/saved_model', save_format='tf')

# Update config.py:
# MODEL_PATH = 'models/saved_model'
```

---

## 📊 Model Information

- **Architecture**: Enhanced U-Net with Attention Gates
- **Input Size**: 256x256x3 (RGB images)
- **Output**: 256x256x1 (Binary segmentation mask)
- **Accuracy**: ~95% (as per training)
- **Framework**: TensorFlow 2.15 / Keras

---

## 🤝 Contributing

To improve the deployment:

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

---

## 📄 License

This project is licensed under MIT License.

---

## 🆘 Support

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section
2. Verify all files are in correct locations
3. Ensure dependencies are installed correctly
4. Check TensorFlow compatibility with your system

---

## 🎉 Success Checklist

Before considering deployment complete, verify:

- [ ] ✅ All files in correct folder structure
- [ ] ✅ `best_model.h5` present in `models/` directory
- [ ] ✅ Virtual environment created and activated
- [ ] ✅ All dependencies installed (`pip install -r requirements.txt`)
- [ ] ✅ `utils/__init__.py` file exists
- [ ] ✅ Test model loading: `python -c "from utils.inference import get_detector"`
- [ ] ✅ Streamlit launches without errors: `streamlit run app.py`
- [ ] ✅ Can upload image and see predictions
- [ ] ✅ Download buttons work correctly

---

**Happy Detecting! 🛢️🔍**
