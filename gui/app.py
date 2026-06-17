import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from datetime import datetime

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Pneumonia Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .pneumonia-card {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .normal-card {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():

    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = BASE_DIR / "model" / "pneumonia_classifier.pth"
    
    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

    model.eval()

    return model

model = load_model()

classes = ["Normal", "Pneumonia"]

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Initialize session state for prediction history
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Model information
    st.subheader("📊 Model Information")
    st.write("**Model Architecture:** EfficientNet-B0")
    st.write("**Task:** Pneumonia Classification")
    st.write("**Classes:** Normal, Pneumonia")
    st.write("**Input Size:** 224 × 224")
    
    st.markdown("---")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence required to make a prediction"
    )
    
    st.markdown("---")
    
    # Show Grad-CAM toggle
    show_gradcam = st.checkbox(
        "Show Grad-CAM Visualization",
        value=True,
        help="Display attention heatmap showing model focus areas"
    )
    
    st.markdown("---")
    
    # Prediction history
    st.subheader("📋 Prediction History")
    if st.session_state.predictions_history:
        for i, pred in enumerate(reversed(st.session_state.predictions_history[-5:]), 1):
            st.write(f"{i}. {pred['prediction']} ({pred['confidence']:.1f}%) - {pred['timestamp']}")
        if st.button("Clear History", key="clear_history"):
            st.session_state.predictions_history = []
            st.rerun()
    else:
        st.write("No predictions yet")
    
    st.markdown("---")
    st.markdown("""
    **⚠️ Medical Disclaimer**
    
    This model is for educational and research purposes only. 
    Do not use for clinical diagnosis without professional medical review.
    """)

# =============================
# MAIN CONTENT
# =============================
st.title("🫁 Pneumonia Detection System")
st.markdown("Upload a chest X-ray image to detect pneumonia using deep learning.")

# =============================
# FILE UPLOAD
# =============================
st.markdown("### 📤 Upload Image")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload a Chest X-ray Image",
        type=["jpg", "jpeg", "png"],
        help="Select a chest X-ray image in JPG, JPEG, or PNG format"
    )

with col2:
    st.write("")  # Spacing
    if st.button("🔄 Clear", help="Clear the current prediction"):
        st.rerun()

# =============================
# PREDICTION
# =============================
if uploaded_file:
    try:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        img_tensor = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            prediction = classes[pred_idx]
            confidence = probs[0][pred_idx].item()
            normal_prob = probs[0][0].item()
            pneumonia_prob = probs[0][1].item()

        # Add to history
        st.session_state.predictions_history.append({
            'prediction': prediction,
            'confidence': confidence * 100,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })

        # =============================
        # RESULTS LAYOUT
        # =============================
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        # Main prediction card
        if prediction == "Pneumonia":
            col_pred = st.container()
            with col_pred:
                st.error(f"""
                    ### ⚠️ PNEUMONIA DETECTED
                    **Prediction:** {prediction}  
                    **Confidence:** {confidence*100:.2f}%
                """)
        else:
            col_pred = st.container()
            with col_pred:
                st.success(f"""
                    ### ✅ NORMAL
                    **Prediction:** {prediction}  
                    **Confidence:** {confidence*100:.2f}%
                """)

        # Image display
        st.markdown("---")
        st.markdown("### 🖼️ Images")
        
        col_img1, col_img2 = st.columns(2)

        with col_img1:
            st.subheader("Original X-ray")
            st.image(image, use_container_width=True, caption="Uploaded chest X-ray")

        with col_img2:
            if show_gradcam:
                st.subheader("Grad-CAM Heatmap")
                try:
                    # Grad-CAM visualization
                    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
                    target_layers = [model.features[-1]]
                    cam = GradCAM(model=model, target_layers=target_layers)
                    grayscale_cam = cam(input_tensor=img_tensor)[0]
                    heatmap = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                    st.image(heatmap, use_container_width=True, caption="Model attention areas")
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM: {str(e)}")
            else:
                st.subheader("Resized Image (Model Input)")
                st.image(image.resize((224, 224)), use_container_width=True, caption="Image used for model (224×224)")

        # Detailed probabilities
        st.markdown("---")
        st.markdown("### 📈 Class Probabilities")

        prob_col1, prob_col2 = st.columns(2)

        with prob_col1:
            st.metric(
                "Normal",
                f"{normal_prob*100:.2f}%",
                delta=None,
                delta_color="off"
            )
            st.progress(float(normal_prob), text="")

        with prob_col2:
            st.metric(
                "Pneumonia",
                f"{pneumonia_prob*100:.2f}%",
                delta=None,
                delta_color="off"
            )
            st.progress(float(pneumonia_prob), text="")

        # Detailed information
        st.markdown("---")
        st.markdown("### 📝 Detailed Information")

        info_col1, info_col2, info_col3 = st.columns(3)

        with info_col1:
            st.write("**Filename:**")
            st.write(uploaded_file.name)

        with info_col2:
            st.write("**Image Size:**")
            st.write(f"{image.size[0]} × {image.size[1]} px")

        with info_col3:
            st.write("**Upload Time:**")
            st.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Clinical note
        st.markdown("---")
        st.info("""
        **🔬 Model Information:**
        - This model was trained on the ChexPert/NIH chest X-ray dataset
        - Sensitivity and specificity vary; always consult medical professionals
        - For research and educational purposes only
        """)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    # No image uploaded yet - show instructions
    st.markdown("---")
    st.info("""
    ### 📋 How to Use:
    1. Click "Browse files" to upload a chest X-ray image (JPG, JPEG, or PNG)
    2. The model will analyze the image and predict if pneumonia is present
    3. View the Grad-CAM heatmap to see which areas influenced the prediction
    4. Check the confidence score and class probabilities
    
    **Supported Formats:** JPG, JPEG, PNG  
    **Recommended Resolution:** 224×224 pixels or higher
    """)