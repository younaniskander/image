import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Function to load a pre-trained model from a file path
def load_pretrained_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

# Function to perform image segmentation using the loaded model
def perform_segmentation(model, image):
    if model is not None:
        # Preprocess the image
        img_np = np.array(image.resize((256, 256)))
        img_np = img_np / 255.0
        img_np = np.expand_dims(img_np, axis=0)

        # Perform segmentation
        segmented_mask = model.predict(img_np)
        segmented_mask = np.argmax(segmented_mask, axis=-1)
        return segmented_mask[0]
    return None

# Function to calculate evaluation metrics (if ground truth mask is available)
def calculate_metrics(segmented_mask, ground_truth_mask):
    # Convert ground truth mask to binary
    ground_truth_mask_binary = cv2.cvtColor(ground_truth_mask, cv2.COLOR_BGR2GRAY)
    
    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = np.sum(np.logical_and(segmented_mask == 1, ground_truth_mask_binary == 255))
    TN = np.sum(np.logical_and(segmented_mask == 0, ground_truth_mask_binary == 0))
    FP = np.sum(np.logical_and(segmented_mask == 1, ground_truth_mask_binary == 0))
    FN = np.sum(np.logical_and(segmented_mask == 0, ground_truth_mask_binary == 255))
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    
    return accuracy, precision, sensitivity

# Streamlit app
def main():
    st.title("Brain Tumor Image Segmentation")

    # Load pre-trained model from a file path
    model_path = "model_UNet.h5"  # Update this to the path where your model is stored
    model = load_pretrained_model(model_path)

    # Upload ground truth mask (if available)
    ground_truth_mask_file = st.file_uploader("Upload ground truth mask (if available)", type=["png", "jpg"], key="mask")
    if ground_truth_mask_file is not None:
        ground_truth_mask = np.array(Image.open(ground_truth_mask_file))
    else:
        ground_truth_mask = None

    # Upload image
    uploaded_file = st.file_uploader("Upload brain MRI image", type=["jpg", "jpeg", "png"], key="image")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        # Perform image segmentation
        segmented_mask = perform_segmentation(model, image)
        
        if segmented_mask is not None:
            # Display segmented image
            st.image(segmented_mask, caption='Segmented Image.', use_column_width=True)

            # Calculate evaluation metrics if ground truth mask is provided
            if ground_truth_mask is not None:
                accuracy, precision, sensitivity = calculate_metrics(segmented_mask, ground_truth_mask)
                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Sensitivity: {sensitivity:.2f}")

if __name__ == '__main__':
    main()
