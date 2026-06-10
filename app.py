import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import sys
import os
import glob

# =====================================================
# ADD SRC FOLDER TO PYTHON PATH
# =====================================================

sys.path.append(os.path.abspath("src"))

from model_CNN import CNNModel
from model_ANN import ANNModel
from dataset import test_loader  # Make sure this path is correct

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Hazelnut Defect Detection",
    layout="centered"
)

st.title("🌰 Hazelnut Defect Detection")

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# FIND ALL SAVED MODELS
# =====================================================

model_files = glob.glob("models/*.pth")

if len(model_files) == 0:
    st.error("No model files found in models/")
    st.stop()

# =====================================================
# MODEL SELECTION
# =====================================================

selected_model = st.selectbox(
    "Select Model",
    model_files,
    key="model_selector"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model(model_path):
    # Detect model type from filename
    if "cnn" in model_path.lower():
        model = CNNModel(dropout_rate=0.3)
        model_type = "cnn"
    elif "ann" in model_path.lower():
        model = ANNModel()
        model_type = "ann"
    else:
        raise ValueError("Could not determine model type from filename.")
    
    # Load weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )
    
    model.to(device)
    model.eval()
    
    return model, model_type

# Load the model
model, model_type = load_model(selected_model)

# =====================================================
# IMAGE TRANSFORM - MUST MATCH TRAINING
# =====================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# =====================================================
# TABS
# =====================================================

tab1, tab2 = st.tabs([
    "Single Image Prediction",
    "Evaluate Test Dataset"
])

# =====================================================
# TAB 1: SINGLE IMAGE PREDICTION
# =====================================================

with tab1:
    st.header("Single Image Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload Hazelnut Image",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert("RGB")
        
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )
        
        # Preprocess Image
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        # ANN Requires Flattened Input
        if model_type == "ann":
            img_tensor = img_tensor.view(
                img_tensor.size(0),
                -1
            )
        
        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        # Class Names
        classes = ["not_ok", "ok"]
        prediction = classes[predicted.item()]
        confidence = confidence.item() * 100
        
        # Prediction Result
        st.divider()
        st.subheader("Prediction Result")
        
        if prediction == "ok":
            st.success(f"GOOD HAZELNUT\n\nConfidence: {confidence:.2f}%")
        else:
            st.error(f"DEFECTIVE HAZELNUT\n\nConfidence: {confidence:.2f}%")
        
        # Probability Scores
        st.subheader("Class Probabilities")
        
        not_ok_prob = probabilities[0][0].item()
        ok_prob = probabilities[0][1].item()
        
        st.write(f"NOT OK: {not_ok_prob * 100:.2f}%")
        st.progress(float(not_ok_prob))
        
        st.write(f"OK: {ok_prob * 100:.2f}%")
        st.progress(float(ok_prob))
        
        # Model Information
        st.divider()
        st.subheader("Model Information")
        st.write(f"Model Type: {model_type.upper()}")
        st.write(f"Loaded Model: {os.path.basename(selected_model)}")

# =====================================================
# TAB 2: MODEL EVALUATION
# =====================================================

with tab2:
    st.header("Model Evaluation")
    
    # Check if test_loader is available
    try:
        if test_loader is None:
            st.error("Test dataset not loaded. Please check dataset.py")
            st.stop()
    except:
        st.error("Test dataset not available. Please check dataset.py")
        st.stop()
    
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating model on test dataset..."):
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    if model_type == "ann":
                        images = images.view(
                            images.size(0),
                            -1
                        )
                    
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            # Display metrics
            st.subheader("Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
                st.metric("Precision", f"{precision*100:.2f}%")
            
            with col2:
                st.metric("Recall", f"{recall*100:.2f}%")
                st.metric("F1 Score", f"{f1*100:.2f}%")
            
            # Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            
            st.subheader("Confusion Matrix")
            
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["not_ok", "ok"],
                yticklabels=["not_ok", "ok"],
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            
            st.pyplot(fig)
            
            # Additional info
            st.info(f"Total test samples: {len(all_labels)}")

# =====================================================
# SIDEBAR INFO
# =====================================================

# with st.sidebar:
#     st.header("📊 Model Info")
#     st.write(f"**Model Type:** {model_type.upper()}")
#     st.write(f"**Device:** {device}")
#     st.write(f"**Input Size:** 128x128")
#     st.write(f"**Classes:** OK / NOT OK")