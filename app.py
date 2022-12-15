import streamlit as st
import os
import tensorflow as tf
from PIL import Image
import numpy as np

def load_model():
    name = r"models\resnet152v2_without_data_augumentation.h5"

    model_path = os.path.join(os.getcwd(), name)

    model = tf.keras.models.load_model(model_path)

    return model

def render_image_for_display(image):
    img = Image.open(image)
    img = img.resize((256,256))
    return img

def render_image_for_model(image):
    image_array = np.array(image)
    image_array = np.reshape(image_array, newshape=(1,256,256,3))
    image_array = image_array / 255.0
    return image_array

def predict(model, image_array):
    prediction = model.predict(image_array)
    return prediction

if __name__ == "__main__":
    st.header("Brain Tumor Classification By Using Transfer Learning (Resnet152V2)")

    model = load_model()

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    image = st.file_uploader(
        "Upload Image for which Prediction is to be made-", 
        type=None,
    )

    if (image != None):
        image = render_image_for_display(image)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image(image, "Inference Image", width=200)
        with col3:
            st.write(' ')

        image_array = render_image_for_model(image)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 , col4, col5 = st.columns(5)
        with col1:
            pass
        with col2:
            pass
        with col4:
            pass
        with col5:
            pass
        with col3 :
            center_button = st.button('Predict')

        if center_button:
            prediction = predict(model, image_array)

            prediction = np.argmax(prediction, axis=1)[0]

            if (prediction == 0):
                st.markdown(
                    "<h1 style='text-align:center;'>Prediction is - Glioma tumor</h1>",
                    unsafe_allow_html=True
                )
            elif (prediction == 1):
                st.markdown(
                    "<h1 style='text-align:center;'>Prediction is - Meningioma tumor</h1>",
                    unsafe_allow_html=True
                )
            elif (prediction == 2):
                st.markdown(
                    "<h1 style='text-align:center;'>Prediction is - No tumor</h1>",
                    unsafe_allow_html=True
                )
            elif (prediction == 3):
                st.markdown(
                    "<h1 style='text-align:center;'>Prediction is - Pituitary tumor</h1>",
                    unsafe_allow_html=True
                )
        