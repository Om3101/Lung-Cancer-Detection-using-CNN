import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
from PIL import Image
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# Load the trained DenseNet-169 model
MODEL_PATH = "lung_cancer_densenet169.h5"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ["Benign", "Malignant", "Normal"]
LABEL_COLORS = {"Benign": "#3498db", "Malignant": "#e74c3c", "Normal": "#2ecc71"}

target_layer = model.get_layer("conv5_block16_concat")  # Adjust if necessary

# Function to preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize to match model input
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to compute Grad-CAM++
def get_gradcam(image, model, target_layer, class_idx=0):
    # Convert image to batch format
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize

    # Initialize Grad-CAM++
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)
    
    # Define score function for target class
    score = CategoricalScore(class_idx)

    # Generate heatmap
    cam = gradcam(score, image, penultimate_layer=target_layer.name)[0]

    # Convert heatmap to color
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap


# Function to overlay heatmap
def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# Streamlit UI Styling
st.set_page_config(page_title="Lung Cancer Classification", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #34495e;'>🔬 Lung Cancer Classification Using DenseNet-169</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #7f8c8d;'>Upload a CT scan image to classify it as Benign, Malignant, or Normal.</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], help="Upload a CT scan for classification.")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Layout: Two columns for displaying images
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='text-align: center; color: #34495e;'>📌 Uploaded Image</h4>", unsafe_allow_html=True)
        st.image(img_rgb, caption="Original Image", use_column_width=True, channels="RGB")

    # Preprocess and predict
    img_array = preprocess_image(img_rgb)

    with st.spinner("🔍 Analyzing the image..."):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100

    prediction_label = CLASS_LABELS[predicted_class]
    label_color = LABEL_COLORS[prediction_label]

    # Display Prediction with Color Highlight
    st.markdown(
        f"<h3 style='text-align: center; color: {label_color};'>🩺 Prediction: {prediction_label}</h3>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='text-align: center; font-size:18px; color: {label_color};'>Confidence: {confidence:.2f}%</p>",
        unsafe_allow_html=True
    )

    # Progress bar for confidence level
    st.progress(int(confidence))

    # Generate Grad-CAM++ heatmap
    heatmap = grad_cam_plus_plus(model, img_array, predicted_class)
    heatmap_overlay = overlay_heatmap(img_rgb, heatmap)

    with col2:
        st.markdown("<h4 style='text-align: center; color: #34495e;'>🔥 Grad-CAM++ Heatmap</h4>", unsafe_allow_html=True)
        st.image(heatmap_overlay, caption="Grad-CAM++ Heatmap", use_column_width=True)

    st.success("✅ Classification Complete!")
    st.balloons()
