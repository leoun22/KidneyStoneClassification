import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import torch

from torchvision import transforms

def preprocess_image(image):
    """
    Preprocess the input image to match the model requirements.
    - Resize to 224x224 (or the appropriate size for your model).
    - Convert to tensor.
    - Normalize with ImageNet mean and std.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    Classifies an input image using the given model.
    """
    # Preprocess the image
    data = preprocess_image(image)

    # Ensure the model is in evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = model(data)

    # Convert model outputs to probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Get the predicted class index and confidence score
    conf_score, class_idx = torch.max(probabilities, dim=1)
    class_name = class_names[class_idx.item()]

    return class_name, conf_score.item()
