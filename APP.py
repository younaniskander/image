import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Function to load a model uploaded by the user
def load_model(uploaded_model):
    if uploaded_model is not None:
        # To read the model file, we need to write it to a temporary file then load it
        with open("model_UNet.h5", "wb") as f:
            f.write(uploaded_model.getbuffer())
        return tf.keras.models.load_model("model_UNet.h5", 
                                          custom_objects={'accuracy': tf.keras.metrics.MeanIoU(num_classes=4),
                                                          "dice_coef": dice_coef,
                                                          "precision": precision,
                                                          "sensitivity": sensitivity,
                                                          "specificity": specificity,
                                                          # "dice_coef_necrotic": dice_coef_necrotic,
                                                          # "dice_coef_edema": dice_coef_edema,
                                                          # "dice_coef_enhancing": dice_coef_enhancing
                                                         }, compile=False)
    return None

# Function to preprocess the image
def preprocess_image(image, image_width=128, image_height=128): # Adjust image_width and image_height as needed
    image = image.resize((image_width, image_height))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def make_prediction(model, image):
    if model and image is not None:
        preprocessed_img = preprocess_image(image)
        prediction = model.predict(preprocessed_img)
        return prediction
    return None

# Streamlit app
def main():
    st.title("Image Classification with your Model")

    # Upload model
    uploaded_model = st.file_uploader("Upload a model", type=["h5"], key="model")
    model = load_model(uploaded_model)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Make Prediction'):
            if model is not None:
                prediction = make_prediction(model, image)
                st.write("Prediction:", prediction)
            else:
                st.error("Please upload a model to make predictions.")

if __name__ == '__main__':
    main()
