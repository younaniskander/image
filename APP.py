import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to perform image segmentation using GrabCut algorithm
def perform_segmentation(image):
    # Convert image to numpy array
    img_np = np.array(image)
    
    # Initialize mask with zeros
    mask = np.zeros(img_np.shape[:2], np.uint8)
    
    # Define the rectangle for initialization
    rect = (50, 50, img_np.shape[1]-50, img_np.shape[0]-50)
    
    # Run GrabCut algorithm
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    cv2.grabCut(img_np,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    # Create mask where all probable foreground and definite foreground are set to 1
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    # Apply the mask to the original image
    segmented_image = img_np * mask2[:, :, np.newaxis]
    
    return segmented_image, mask2

# Function to calculate evaluation metrics (accuracy, precision, sensitivity)
def calculate_metrics(segmented_mask, ground_truth_mask):
    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = np.sum(np.logical_and(segmented_mask == 1, ground_truth_mask == 1))
    TN = np.sum(np.logical_and(segmented_mask == 0, ground_truth_mask == 0))
    FP = np.sum(np.logical_and(segmented_mask == 1, ground_truth_mask == 0))
    FN = np.sum(np.logical_and(segmented_mask == 0, ground_truth_mask == 1))
    
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    
    return accuracy, precision, sensitivity

# Streamlit app
def main():
    st.title("Brain Tumor Image Segmentation")

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
        segmented_image, segmented_mask = perform_segmentation(image)
        
        # Calculate evaluation metrics if ground truth mask is provided
        if ground_truth_mask is not None:
            accuracy, precision, sensitivity = calculate_metrics(segmented_mask, ground_truth_mask)
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Sensitivity: {sensitivity:.2f}")

        # Display segmented image
        st.image(segmented_image, caption='Segmented Image.', use_column_width=True)

if __name__ == '__main__':
    main()
