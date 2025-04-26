import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

# Load the trained DenseNet-169 model
model_path = "lung_cancer_densenet169.h5"
model = load_model(model_path)

# Find the last convolutional layer
target_layer = model.get_layer("conv5_block16_concat")  # Adjust if necessary

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

# Streamlit UI
st.title("Grad-CAM++ for Lung Cancer Detection")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to NumPy
    image_np = np.array(image)
    image_np = cv2.resize(image_np, (224, 224))  # Resize for DenseNet
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Get Grad-CAM++ heatmap
    heatmap = get_gradcam(image_np, model, target_layer)

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(image_np.astype(np.float32) / 255, 0.5, heatmap, 0.5, 0)

    st.image(overlay, caption="Grad-CAM++ Overlay", use_column_width=True)
