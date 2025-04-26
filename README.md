
# 🔬 Lung Cancer Detection using DenseNet-169 and Grad-CAM++

## 📄 Project Overview
This project presents an AI-based system for **automated lung cancer classification** from CT scan images using a **DenseNet-169** deep learning model combined with **Grad-CAM++** visualization for interpretability.  
The model classifies CT scans into three categories:
- **Benign**
- **Malignant**
- **Normal**

A **Streamlit web application** is developed for **real-time image upload, classification, and heatmap visualization** to support radiologists in early and accurate diagnosis.

🚀 **Live App:** [Om3101-Lung-App](https://a-lung-detection-09.streamlit.app/)

---

## 📚 Key Components
- **Model:** DenseNet-169 architecture, fine-tuned for 3-class lung cancer detection.
- **Explainable AI (XAI):** Grad-CAM++ heatmaps highlight important areas influencing model predictions.
- **Deployment:** Interactive Streamlit app for real-time usage.

---

## 🛠️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- tf-keras-vis (for Grad-CAM++)
- Scikit-learn
- Matplotlib, Seaborn

---

## 🖥️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/lung-cancer-detection.git
   cd lung-cancer-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Upload a CT scan image** in JPG/PNG format and view:
   - Predicted category (Benign, Malignant, Normal)
   - Confidence score
   - Grad-CAM++ attention heatmap

---

## 📂 Project Structure
```
├── app.py                  # Streamlit web application
├── lung_cancer_densenet169.h5 # Trained DenseNet-169 model
├── Updated_densenet.ipynb   # Model training notebook
├── Preprocessed datasets (x_train.npy, y_train.npy, etc.)
├── cleaned_dataset/         # Directory containing processed CT scan images
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
```

---

## 📈 Results
- **Overall Test Accuracy:** **99%**
- **High Precision, Recall, F1-Scores** for all three categories.
- **Clear Grad-CAM++ Visualizations** showing model attention on CT scan abnormalities.

---

## 🔮 Future Work
- Expanding dataset diversity for better generalization.
- Incorporating PET/MRI scans for multi-modal diagnosis.
- Integrating full clinical deployment with electronic health records (EHRs).
- Exploring attention-based and transformer architectures for enhanced performance.

---

## 🙏 Acknowledgments
- Dataset Source: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)
- Streamlit and tf-keras-vis open-source communities.

---

# 🎯 Try It Live
👉 [**Om3101-Lung-App**](https://a-lung-detection-09.streamlit.app/)  

Upload your CT scan, get instant results, and visualize the important regions with Grad-CAM++ heatmaps!

---
