import streamlit as st
from PIL import Image
import numpy as np
import torch
from util import classify, set_background
import torchvision

set_background('/Users/leoun22/Desktop/Medvation web2.png')

# set title
st.title('Kidney Stone classification')

# set header
st.header('Please upload an image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
from torchvision.models import efficientnet_b0

# Load the model (Ensure the architecture matches the training process)
model = efficientnet_b0(pretrained=False)
model = torch.load('/Users/leoun22/Desktop/kidney_stone_model.pth', map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode


# load class names
class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

from PIL import Image

# Load the uploaded image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))