# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19dFdNLohov4KNzcgf_ZXTofn85AnbDcV
"""

# prompt: install from streamlit_option_menu import option_menu

!pip install streamlit

!pip install streamlit-option-menu

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns


# loading the saved models

model = pickle.load(open('/content/skin_disease_model.sav', 'rb'))
class_labels = ['cellulitis', 'impetigo', 'athlete-foot', 'nail-fungus', 'ringworm',
                'cutaneous-larva-migrans', 'chickenpox', 'shingles']
def preprocess_image(image):
    img = cv2.resize(np.array(image), (224, 224))  # Resize to match training input
    img = img / 255.0 # Normalize
    return np.expand_dims(img, axis=0)

def predict_disease(image):
  processed_image = preprocess_image(image)
  prediction = model.predict(processed_image)
  predicted_class_index = np.argmax(prediction)
  predicted_class = class_labels[predicted_class_index]
  probability = prediction[0][predicted_class_index]

  return predicted_class, probability


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    predicted_class, probability = predict_disease(image)

    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {probability:.2%}")

