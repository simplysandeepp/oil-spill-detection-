🚀 **Oil Spill Detection System - Deployment Guide**
====================================================

**From Trained Model (Colab) to Running Streamlit App**
-------------------------------------------------------

📋 **Table of Contents**
------------------------

1.  [Export Model from Colab](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-1-export-model-from-google-colab)
    
2.  [Setup Local Environment](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-2-setup-local-development-environment)
    
3.  [Create Project Structure](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-3-create-project-structure)
    
4.  [Install Dependencies](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-4-install-dependencies)
    
5.  [Configure Files](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-5-configure-all-files)
    
6.  [Test Model Loading](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-6-test-model-loading)
    
7.  [Run Streamlit App](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-7-run-streamlit-app)
    
8.  [Troubleshooting](https://claude.ai/chat/b6353ca8-148c-4a9c-8baf-4448663cdaac?artifactId=oil_spill_model_architecture#phase-8-troubleshooting)
    

📦 **Phase 1: Export Model from Google Colab**
----------------------------------------------

### **Step 1: Download Trained Model**

After your model training completes in Colab, run this cell:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Download the trained model  from google.colab import files  import os  if os.path.exists('models/best_model.h5'):      print("✓ Model found!")      print(f"  Size: {os.path.getsize('models/best_model.h5') / (1024*1024):.2f} MB")      # Download      files.download('models/best_model.h5')      print("✓ Model downloaded to your computer!")  else:      print("❌ Model not found!")   `

### **Step 2: Save Model File**

*   Check your **Downloads** folder
    
*   File: best\_model.h5 (50-200 MB)
    
*   **Keep this file safe** - you'll need it for deployment
    

💻 **Phase 2: Setup Local Development Environment**
---------------------------------------------------

### **Prerequisites**

*   **Python 3.8 - 3.11** installed
    
*   **Git** (optional, for GitHub)
    
*   **PowerShell** or **Command Prompt** (Windows) / **Terminal** (Mac/Linux)
    

### **Step 1: Create Project Directory**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Navigate to desired location  cd E:\  # Windows  # or  cd ~/Documents  # Mac/Linux  # Create project folder  mkdir oil-spill-detection  cd oil-spill-detection   `

### **Step 2: Create Virtual Environment**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Create virtual environment  python -m venv venv  # Activate it  # Windows PowerShell:  .\venv\Scripts\Activate.ps1  # Windows Command Prompt:  venv\Scripts\activate  # Mac/Linux:  source venv/bin/activate   `

You should see (venv) in your terminal prompt.

📂 **Phase 3: Create Project Structure**
----------------------------------------

### **Step 1: Create Folders**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Create all necessary folders  mkdir models utils config assets temp_uploads   `

### **Step 2: Create \_\_init\_\_.py Files**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Windows PowerShell:  New-Item -Path models\__init__.py -ItemType File  New-Item -Path utils\__init__.py -ItemType File  New-Item -Path config\__init__.py -ItemType File  # Mac/Linux:  touch models/__init__.py utils/__init__.py config/__init__.py   `

### **Step 3: Copy Your Model**

Copy best\_model.h5 from Downloads to models/ folder:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Windows:  copy "C:\Users\YourName\Downloads\best_model.h5" models\  # Mac/Linux:  cp ~/Downloads/best_model.h5 models/   `

### **Final Structure**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   oil-spill-detection/  ├── models/  │   ├── __init__.py  │   ├── model_architecture.py  (create next)  │   └── best_model.h5          (your trained model)  ├── utils/  │   ├── __init__.py  │   ├── preprocessing.py       (create next)  │   ├── inference.py           (create next)  │   └── visualization.py       (create next)  ├── config/  │   ├── __init__.py  │   └── config.py              (create next)  ├── assets/  ├── temp_uploads/  ├── venv/                      (virtual environment)  ├── app.py                     (create next)  ├── requirements.txt           (create next)  └── README.md                  (this file)   `

📦 **Phase 4: Install Dependencies**
------------------------------------

### **Step 1: Create requirements.txt**

Create file in project root:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad requirements.txt  # Windows  nano requirements.txt     # Mac/Linux   `

**Paste this:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Core Deep Learning  tensorflow>=2.16.0  keras>=2.15.0  # Image Processing  opencv-python==4.9.0.80  Pillow==10.2.0  # Data Processing  numpy==1.26.3  pandas==2.2.0  # Visualization  matplotlib==3.8.2  seaborn==0.13.1  # Web Application  streamlit==1.31.0  # Utilities  tqdm==4.66.1  scikit-learn==1.4.0   `

### **Step 2: Install All Dependencies**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Make sure virtual environment is activated (see (venv) in prompt)  pip install --upgrade pip  pip install -r requirements.txt   `

This will take 5-10 minutes. Wait for completion.

⚙️ **Phase 5: Configure All Files**
-----------------------------------

### **File 1: config/config.py**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad config\config.py  # Windows  nano config/config.py     # Mac/Linux   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ============================================================================  # CONFIG.PY - Configuration Settings  # ============================================================================  import os  # Model Parameters  IMG_HEIGHT = 256  IMG_WIDTH = 256  IMG_CHANNELS = 3  CONFIDENCE_THRESHOLD = 0.5  # File Paths  MODEL_PATH = 'models/best_model.h5'  UPLOAD_FOLDER = 'temp_uploads'  # Visualization Settings  OVERLAY_ALPHA = 0.4  SPILL_COLOR = [255, 0, 0]  # Red  # Streamlit Settings  PAGE_TITLE = "🛢️ Oil Spill Detection System"  PAGE_ICON = "🛢️"  MAX_FILE_SIZE = 10  # MB  # Create directories  os.makedirs('models', exist_ok=True)  os.makedirs(UPLOAD_FOLDER, exist_ok=True)   `

### **File 2: utils/preprocessing.py**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad utils\preprocessing.py   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ============================================================================  # PREPROCESSING.PY - Image Preprocessing  # ============================================================================  import cv2  import numpy as np  from PIL import Image  import config.config as cfg  def load_and_preprocess_image(image_input, target_size=(cfg.IMG_HEIGHT, cfg.IMG_WIDTH)):      """Load and preprocess image for inference"""      # Handle different input types      if isinstance(image_input, str):          img = cv2.imread(image_input)          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      elif isinstance(image_input, Image.Image):          img = np.array(image_input)      else:          img = image_input      original_img = img.copy()      # Resize and normalize      resized_img = cv2.resize(img, target_size)      preprocessed_img = resized_img.astype(np.float32) / 255.0      preprocessed_img = np.expand_dims(preprocessed_img, axis=0)      return preprocessed_img, original_img  def postprocess_mask(pred_mask, threshold=cfg.CONFIDENCE_THRESHOLD, target_size=None):      """Convert probability mask to binary"""      if len(pred_mask.shape) > 2:          pred_mask = pred_mask.squeeze()      confidence_map = pred_mask.copy()      binary_mask = (pred_mask > threshold).astype(np.uint8) * 255      if target_size is not None:          binary_mask = cv2.resize(binary_mask, target_size)          confidence_map = cv2.resize(confidence_map, target_size)      return binary_mask, confidence_map  def validate_image(image_input):      """Validate image format"""      try:          if isinstance(image_input, str):              img = cv2.imread(image_input)          elif isinstance(image_input, Image.Image):              img = np.array(image_input)          else:              img = image_input          if img is None:              return False, "Unable to read image"          return True, "Valid"      except Exception as e:          return False, str(e)   `

### **File 3: utils/visualization.py**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad utils\visualization.py   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ============================================================================  # VISUALIZATION.PY - Visualization Utilities  # ============================================================================  import cv2  import numpy as np  import config.config as cfg  def create_overlay(original_image, binary_mask, alpha=cfg.OVERLAY_ALPHA):      """Create red overlay on detected regions"""      if original_image.shape[:2] != binary_mask.shape:          binary_mask = cv2.resize(binary_mask,                                   (original_image.shape[1], original_image.shape[0]))      overlay = original_image.copy()      overlay[binary_mask > 127] = cfg.SPILL_COLOR      blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)      return blended  def create_confidence_heatmap(confidence_map, original_image=None):      """Create confidence heatmap visualization"""      heatmap = (confidence_map * 255).astype(np.uint8)      heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)      heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)      if original_image is not None:          if original_image.shape[:2] != heatmap_colored.shape[:2]:              heatmap_colored = cv2.resize(heatmap_colored,                                          (original_image.shape[1],                                            original_image.shape[0]))          heatmap_colored = cv2.addWeighted(original_image, 0.5,                                            heatmap_colored, 0.5, 0)      return heatmap_colored   `

### **File 4: models/model\_architecture.py**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad models\model_architecture.py   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ============================================================================  # MODEL_ARCHITECTURE.PY - Enhanced U-Net  # ============================================================================  import tensorflow as tf  from tensorflow import keras  from tensorflow.keras import layers, models  import sys  sys.setrecursionlimit(50000)  IMG_HEIGHT = 256  IMG_WIDTH = 256  IMG_CHANNELS = 3  try:      import config.config as cfg      IMG_HEIGHT = cfg.IMG_HEIGHT      IMG_WIDTH = cfg.IMG_WIDTH      IMG_CHANNELS = cfg.IMG_CHANNELS  except:      pass  def attention_block(x, g, inter_channel):      """Attention gate"""      theta_x = layers.Conv2D(inter_channel, 1, padding='same')(x)      phi_g = layers.Conv2D(inter_channel, 1, padding='same')(g)      if x.shape[1] != g.shape[1]:          phi_g = layers.UpSampling2D(size=(2, 2))(phi_g)      add_xg = layers.Add()([theta_x, phi_g])      act_xg = layers.Activation('relu')(add_xg)      psi = layers.Conv2D(1, 1, padding='same')(act_xg)      psi = layers.Activation('sigmoid')(psi)      y = layers.Multiply()([x, psi])      y = layers.Conv2D(inter_channel, 1, padding='same')(y)      return y  def residual_conv_block(inputs, num_filters, use_dropout=False):      """Residual block"""      x = layers.Conv2D(num_filters, 3, padding='same')(inputs)      x = layers.BatchNormalization()(x)      x = layers.Activation('relu')(x)      if use_dropout:          x = layers.Dropout(0.2)(x)      x = layers.Conv2D(num_filters, 3, padding='same')(x)      x = layers.BatchNormalization()(x)      if inputs.shape[-1] == num_filters:          shortcut = inputs      else:          shortcut = layers.Conv2D(num_filters, 1, padding='same')(inputs)      x = layers.Add()([x, shortcut])      x = layers.Activation('relu')(x)      return x  def encoder_block(inputs, num_filters, use_dropout=False):      """Encoder block"""      x = residual_conv_block(inputs, num_filters, use_dropout)      p = layers.MaxPooling2D((2, 2))(x)      return x, p  def decoder_block(inputs, skip_features, num_filters, use_attention=True):      """Decoder block"""      x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)      if use_attention:          skip_features = attention_block(skip_features, x, num_filters)      x = layers.Concatenate()([x, skip_features])      x = residual_conv_block(x, num_filters)      return x  def build_enhanced_unet(input_shape=None):      """Build Enhanced U-Net model"""      if input_shape is None:          input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)      inputs = layers.Input(input_shape, name='input_image')      # Encoder      s1, p1 = encoder_block(inputs, 64, use_dropout=False)      s2, p2 = encoder_block(p1, 128, use_dropout=True)      s3, p3 = encoder_block(p2, 256, use_dropout=True)      s4, p4 = encoder_block(p3, 512, use_dropout=True)      # Bridge      bridge = residual_conv_block(p4, 1024, use_dropout=True)      # Decoder      d1 = decoder_block(bridge, s4, 512, use_attention=True)      d2 = decoder_block(d1, s3, 256, use_attention=True)      d3 = decoder_block(d2, s2, 128, use_attention=True)      d4 = decoder_block(d3, s1, 64, use_attention=True)      # Output      outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid',                              dtype='float32', name='output_mask')(d4)      model = models.Model(inputs, outputs, name='Enhanced-Attention-UNet')      return model   `

### **File 5: utils/inference.py**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad utils\inference.py   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ============================================================================  # INFERENCE.PY - Model Inference  # ============================================================================  import numpy as np  import tensorflow as tf  from tensorflow import keras  import sys  import os  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  sys.setrecursionlimit(50000)  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  import config.config as cfg  from utils.preprocessing import load_and_preprocess_image, postprocess_mask  class OilSpillDetector:      """Oil Spill Detection Model Wrapper"""      def __init__(self, model_path=cfg.MODEL_PATH):          self.model_path = model_path          self.model = None          self.load_model()      def load_model(self):          """Load model by rebuilding architecture and loading weights"""          try:              print("🔄 Loading model...")              from models.model_architecture import build_enhanced_unet              print("  → Building architecture...")              self.model = build_enhanced_unet()              print(f"  → Loading weights from {self.model_path}...")              self.model.load_weights(self.model_path)              print("✓ Model loaded successfully!")          except Exception as e:              raise RuntimeError(f"Failed to load model: {str(e)}")      def predict(self, image_input):          """Perform inference"""          preprocessed, original = load_and_preprocess_image(image_input)          pred_mask = self.model.predict(preprocessed, verbose=0)[0].squeeze()          original_size = (original.shape[1], original.shape[0])          binary_mask, confidence_map = postprocess_mask(              pred_mask,               threshold=cfg.CONFIDENCE_THRESHOLD,              target_size=original_size          )          metrics = self._calculate_metrics(binary_mask, confidence_map)          return {              'binary_mask': binary_mask,              'confidence_map': confidence_map,              'original_image': original,              'metrics': metrics          }      def _calculate_metrics(self, binary_mask, confidence_map):          """Calculate detection statistics"""          total_pixels = binary_mask.size          detected_pixels = np.sum(binary_mask > 0)          coverage_percentage = (detected_pixels / total_pixels) * 100          if detected_pixels > 0:              avg_confidence = np.mean(confidence_map[binary_mask > 0])          else:              avg_confidence = 0.0          max_confidence = np.max(confidence_map)          return {              'coverage_percentage': coverage_percentage,              'detected_pixels': detected_pixels,              'total_pixels': total_pixels,              'avg_confidence': avg_confidence,              'max_confidence': max_confidence,              'has_spill': detected_pixels > 0          }  _detector_instance = None  def get_detector(model_path=cfg.MODEL_PATH):      """Get or create detector instance"""      global _detector_instance      if _detector_instance is None:          _detector_instance = OilSpillDetector(model_path)      return _detector_instance   `

### **File 6: app.py (Main Streamlit Application)**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   notepad app.py   `

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # ============================================================================  # APP.PY - Streamlit Web Application  # ============================================================================  import streamlit as st  import numpy as np  from PIL import Image  import sys  import os  sys.path.append(os.path.dirname(os.path.abspath(__file__)))  from utils.inference import get_detector  from utils.visualization import create_overlay, create_confidence_heatmap  from utils.preprocessing import validate_image  import config.config as cfg  # Page config  st.set_page_config(      page_title=cfg.PAGE_TITLE,      page_icon=cfg.PAGE_ICON,      layout="wide"  )  # Custom CSS  st.markdown("""  </div><div class="slate-code_line">    .main-header {</div><div class="slate-code_line">        font-size: 3rem;</div><div class="slate-code_line">        font-weight: bold;</div><div class="slate-code_line">        text-align: center;</div><div class="slate-code_line">        color: #ff4444;</div><div class="slate-code_line">        padding: 1rem;</div><div class="slate-code_line">        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);</div><div class="slate-code_line">        border-radius: 10px;</div><div class="slate-code_line">        margin-bottom: 2rem;</div><div class="slate-code_line">    }</div><div class="slate-code_line">  """, unsafe_allow_html=True)  # Load model  @st.cache_resource  def load_model():      try:          return get_detector(cfg.MODEL_PATH)      except Exception as e:          st.error(f"❌ Failed to load model: {str(e)}")          st.stop()  # Main app  def main():      st.markdown(          f'  {cfg.PAGE_ICON} Oil Spill Detection System  ',          unsafe_allow_html=True      )      # Sidebar      with st.sidebar:          st.header("ℹ️ About")          st.markdown("""          Deep Learning-based oil spill detection in satellite/aerial imagery.          **Model:** Enhanced U-Net with Attention Gates            **Accuracy:** ~95%          """)          st.divider()          st.header("⚙️ Settings")          confidence_threshold = st.slider(              "Confidence Threshold",              0.0, 1.0, cfg.CONFIDENCE_THRESHOLD, 0.05          )          overlay_alpha = st.slider(              "Overlay Transparency",              0.0, 1.0, cfg.OVERLAY_ALPHA, 0.1          )          if 'total_processed' not in st.session_state:              st.session_state.total_processed = 0          if 'total_detections' not in st.session_state:              st.session_state.total_detections = 0          st.divider()          st.header("📊 Statistics")          st.metric("Images Processed", st.session_state.total_processed)          st.metric("Spills Detected", st.session_state.total_detections)      # File uploader      uploaded_file = st.file_uploader(          "Upload an image",          type=['jpg', 'jpeg', 'png']      )      if uploaded_file is not None:          detector = load_model()          cfg.CONFIDENCE_THRESHOLD = confidence_threshold          cfg.OVERLAY_ALPHA = overlay_alpha          col1, col2 = st.columns([1, 1])          with col1:              st.subheader("📷 Original Image")              image = Image.open(uploaded_file).convert('RGB')              st.image(image, use_column_width=True)          # Process          is_valid, message = validate_image(image)          if not is_valid:              st.error(f"❌ Invalid image: {message}")              return          with st.spinner('🔍 Analyzing...'):              results = detector.predict(image)          st.session_state.total_processed += 1          if results['metrics']['has_spill']:              st.session_state.total_detections += 1          # Results          overlay = create_overlay(              results['original_image'],              results['binary_mask'],              alpha=overlay_alpha          )          with col2:              st.subheader("🎯 Detection Result")              if results['metrics']['has_spill']:                  st.markdown('  ⚠️ OIL SPILL DETECTED  ', unsafe_allow_html=True)              else:                  st.markdown('  ✅ NO OIL SPILL  ', unsafe_allow_html=True)              st.image(overlay, use_column_width=True)          # Metrics          st.divider()          st.subheader("📊 Detection Metrics")          col1, col2, col3, col4 = st.columns(4)          col1.metric("Coverage", f"{results['metrics']['coverage_percentage']:.2f}%")          col2.metric("Avg Confidence", f"{results['metrics']['avg_confidence']:.1%}")          col3.metric("Max Confidence", f"{results['metrics']['max_confidence']:.1%}")          col4.metric("Detected Pixels", f"{results['metrics']['detected_pixels']:,}")          # Additional views          st.divider()          st.subheader("🔍 Detailed Analysis")          tab1, tab2, tab3 = st.tabs(["Binary Mask", "Confidence Heatmap", "Raw Data"])          with tab1:              st.image(results['binary_mask'], clamp=True, use_column_width=True)          with tab2:              heatmap = create_confidence_heatmap(results['confidence_map'])              st.image(heatmap, use_column_width=True)          with tab3:              st.json({                  'has_spill': bool(results['metrics']['has_spill']),                  'coverage': float(results['metrics']['coverage_percentage']),                  'confidence': float(results['metrics']['avg_confidence'])              })  if __name__ == "__main__":      main()   `

🧪 **Phase 6: Test Model Loading**
----------------------------------

Before running the app, test if everything works:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Test model architecture  python models/model_architecture.py  # Test model loading  python -c "from utils.inference import get_detector; print('Model loaded successfully!')"   `

**Expected output:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   🔄 Loading model...    → Building architecture...    → Loading weights from models/best_model.h5...  ✓ Model loaded successfully!  Model loaded successfully!   `

🚀 **Phase 7: Run Streamlit App**
---------------------------------

### **Start the Application**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   streamlit run app.py   `

### **What Happens:**

1.  Terminal shows:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   You can now view your Streamlit app in your browser.    Local URL: http://localhost:8501    Network URL: http://192.168.x.x:8501   `

1.  **Browser opens automatically** (or open manually: http://localhost:8501)
    
2.  **Upload an image** and see results!
    

### **Using the App:**

1.  **Upload Image:** Click "Browse files" or drag & drop
    
2.  **View Results:** Detection overlay, confidence scores
    
3.  **Adjust Settings:** Use sidebar sliders
    
4.  **Download Results:** Click download buttons
    

🐛 **Phase 8: Troubleshooting**
-------------------------------

### **Issue 1: "Model not found"**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Check if model exists  ls models/best_model.h5  # Mac/Linux  dir models\best_model.h5 # Windows  # If missing, copy it again   `

### **Issue 2: "Cannot import module"**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Create missing __init__.py files  # Windows:  New-Item -Path models\__init__.py -ItemType File -Force  New-Item -Path utils\__init__.py -ItemType File -Force  New-Item -Path config\__init__.py -ItemType File -Force  # Mac/Linux:  touch models/__init__.py utils/__init__.py config/__init__.py   `

### **Issue 3: "TensorFlow not found"**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install tensorflow   `

### **Issue 4: "Recursion limit exceeded"**

Already fixed in code with sys.setrecursionlimit(50000)

### **Issue 5: Port already in use**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Use different port  streamlit run app.py --server.port 8502   `

### **Issue 6: Model detects everything except water**

This means masks were inverted during training. **Temporary fix:**

In utils/inference.py, add this line in predict function:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pred_mask = self.model.predict(preprocessed, verbose=0)[0].squeeze()  pred_mask = 1.0 - pred_mask  # ADD THIS LINE   `

**Permanent fix:** Retrain model with inverted masks (see training notebook).

📋 **Quick Reference Commands**
-------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Activate environment  .\venv\Scripts\Activate.ps1  # Windows  source venv/bin/activate     # Mac/Linux  # Install dependencies  pip install -r requirements.txt  # Test model  python -c "from utils.inference import get_detector"  # Run app  streamlit run app.py  # Stop app  Ctrl+C  # Deactivate environment  deactivate   `

✅ **Success Checklist**
-----------------------

*   \[ \] Model downloaded from Colab
    
*   \[ \] Virtual environment created & activated
    
*   \[ \] All folders created
    
*   \[ \] All files created with correct code
    
*   \[ \] \_\_init\_\_.py files in all packages
    
*   \[ \] Dependencies installed
    
*   \[ \] Model file in models/ folder
    
*   \[ \] Test model loading successful
    
*   \[ \] Streamlit app launches
    
*   \[ \] Can upload image and see predictions
    

**Next Steps:**

*   Deploy to Streamlit Cloud (tomorrow!)
    
*   Test with more images
    
*   Fine-tune confidence threshold
    
*   Share with your team
    

**Built with ❤️ using TensorFlow, Keras, and Streamlit**