import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os

from tensorflow.keras.metrics import dice_coef  

# Define a function to download the model
def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        response = requests.get(model_url)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            f.write(response.content)
        st.write("Model downloaded successfully.")
    else:
        st.write("Model already exists.")

# Specify the model's URL and local path
model_url = 'https://drive.google.com/file/d/13VwABEDa1Wchyg0n_RmZUMKsVcrDMWM9/view?usp=sharing'
model_path = 'model_2021_2D_UNet.h5'

# Download the model
download_model(model_url, model_path)

# Load your trained model
model = tf.keras.models.load_model(model_path, 
                                    custom_objects={'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                    "precision": precision,
                                                    "sensitivity":sensitivity,
                                                    "specificity":specificity,
                                                    }, compile=False)

# Function to preprocess the image
def preprocess_image(image, image_width=128, image_height=128): # Adjust image_width and image_height as needed
    image = image.resize((image_width, image_height))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def make_prediction(image):
    preprocessed_img = preprocess_image(image)
    prediction = model.predict(preprocessed_img)
    return prediction

# Streamlit app
def main():
    st.title("Image Classification with your Model")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Make Prediction'):
            prediction = make_prediction(image)
            st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
