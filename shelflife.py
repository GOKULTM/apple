import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Verify the model file path
model_path = r"C:\Users\gokul\Desktop\college\ip\shelf life project\appleshelflife.h5"

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    model = None  # Ensure model variable exists
else:
    try:
        # Load the trained model
        model = load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None  # Prevent crashes if model fails to load

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    # Resize the image to match the model's expected input shape
    uploaded_image = uploaded_image.resize((300, 300))  # Adjust to your model's input size
    image_array = np.array(uploaded_image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app UI
st.title("🍎 Apple Shelf Life Classification App")
st.write("Upload an image of an apple, and the model will classify its shelf life.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded image
        uploaded_image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)  # Fixed parameter name
        st.write("Processing...")

        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_image)

        # Make predictions if the model is loaded
        if model is not None:
            predictions = model.predict(preprocessed_image)[0]  # Get prediction probabilities

            # Class labels (adjust these to match your model's training labels)
            class_labels = ["Apple(1-3)", "Apple(4-8)", "Apple(9-13)", "AppleExpired"]

            # Get the predicted class and confidence score
            class_idx = np.argmax(predictions)
            predicted_class = class_labels[class_idx]
            confidence = predictions[class_idx] * 100

            # Create a DataFrame for visualization
            df = pd.DataFrame({
                "Class": class_labels,
                "Confidence (%)": predictions * 100  # Convert probabilities to percentages
            })

            # Sort by confidence for better visualization
            df = df.sort_values(by="Confidence (%)", ascending=False)

            # Plot the confidence levels
            st.write("### Confidence Levels:")
            fig, ax = plt.subplots()
            ax.bar(df["Class"], df["Confidence (%)"], color="red")
            ax.set_xlabel("Class")
            ax.set_ylabel("Confidence (%)")
            ax.set_title("Model Confidence by Class")
            st.pyplot(fig)

            # Display the predicted class and confidence
            st.write(f"### 🏆 Predicted Class: {predicted_class}")
            st.write(f"### 🔍 Confidence: {confidence:.2f}%")
        else:
            st.error("Model is not loaded. Please check the model file path.")

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a valid JPG, JPEG, or PNG file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.write("📤 Please upload an image to proceed.")