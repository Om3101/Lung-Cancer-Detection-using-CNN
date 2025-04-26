import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained DenseNet-169 model
MODEL_PATH = "lung_cancer_densenet169.h5"
model = load_model(MODEL_PATH)

# Define class labels and styling
CLASS_LABELS = ["Benign", "Malignant", "Normal"]
LABEL_COLORS = {"Benign": "#3498db", "Malignant": "#e74c3c", "Normal": "#2ecc71"}
last_conv_layer_name = "conv5_block32_concat"

# Function to preprocess image
def preprocess_image(img):
    """Convert image to RGB, resize, normalize, and expand dimensions for model input."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to 3 channels
    img_resized = cv2.resize(img_rgb, (224, 224))  # Resize to model's expected input size
    img_normalized = img_resized.astype("float32") / 255.0  # Normalize pixel values

    # Expand dimensions to match model input shape (batch_size, height, width, channels)
    return np.expand_dims(img_normalized, axis=0)  # Final shape: (1, 224, 224, 3)




# Function to generate Grad-CAM++ heatmap
def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
        grads = tape1.gradient(class_output, conv_output)
        grads_2 = tape2.gradient(grads, conv_output)

    grads = grads[0]
    grads_2 = grads_2[0]
    conv_output = conv_output[0]

    alpha = grads_2 / (2 * grads_2 + grads + 1e-10)
    weights = np.sum(alpha * grads, axis=(0, 1))

    heatmap = np.zeros(conv_output.shape[:2], dtype=np.float32)
    for i in range(weights.shape[0]):
        heatmap += weights[i] * conv_output[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1.0
    heatmap = np.power(heatmap, 1.2)
    return heatmap

# Function to overlay heatmap on original image
def overlay_heatmap(img, heatmap):
    """Resize and blend heatmap with grayscale image."""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2RGB)
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(img_colored, 0.5, heatmap_resized, 0.5, 0)

# Streamlit UI Configuration
st.set_page_config(page_title="Lung Cancer Classification", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #34495e;'>🔬 Lung Cancer Classification with Grad-CAM++</h1>
    <h4 style='text-align: center; color: #7f8c8d;'>Upload a CT scan to classify and visualize model decisions</h4>
""", unsafe_allow_html=True)
st.markdown("---")

# File Upload Section
uploaded_file = st.file_uploader("📤 Upload CT Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess image
    img_array = preprocess_image(img)

    # Model prediction
    with st.spinner("🔍 Analyzing image..."):
        predictions = model.predict(img_array)
        pred_class = np.argmax(predictions)
        confidence = predictions[0][pred_class] * 100
        label = CLASS_LABELS[pred_class]

    # Generate Grad-CAM++ heatmap
    heatmap = make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_class)
    overlay = overlay_heatmap(img_gray, heatmap)

    # Display Results
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Original Image")
        st.image(img_rgb, use_column_width=True)
    with col2:
        st.markdown("### 🔥 Model Attention Map")
        st.image(overlay, use_column_width=True)
    
    st.markdown(f"""
        <h3 style='text-align: center; color: {LABEL_COLORS[label]};'>🩺 Prediction: {label} ({confidence:.2f}%)</h3>
    """, unsafe_allow_html=True)
    st.success("✅ Analysis Complete!")
