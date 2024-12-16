import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

# Setup device and transformations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std)
])

# Initialize the model architecture (ResNet18)
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, 12)  # Assuming 12 classes for plant species
    return model

# Load the model state dictionary from a file
def load_model(model, model_file):
    try:
        # Load the pre-trained weights (excluding the 'fc' layer)
        state_dict = torch.load(model_file, map_location=DEVICE)
        
        # Remove 'fc' layer weights from the state_dict before loading
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        
        model.load_state_dict(state_dict, strict=False)  # Load weights without 'fc'
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Prediction function
def predict_image(image, model):
    # Open the image
    image = Image.open(image)
    
    # Convert image to RGB if it is not in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply the transformation
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(DEVICE)

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_label = labels[predicted.item()]

    return predicted_label


# Streamlit App UI
st.title("Plant Seedlings Classification")
st.write("Upload an image of a plant seedling to classify its species.")

# Map of labels (make sure this matches your dataset's classes)
labels = ['Species1', 'Species2', 'Species3', 'Species4', 'Species5', 'Species6', 'Species7', 'Species8', 'Species9', 'Species10', 'Species11', 'Species12']

# Model uploader section
uploaded_model = st.file_uploader("Upload your trained model", type=["pth"])

if uploaded_model is not None:
    # Create model architecture and load the uploaded model
    model = create_model()
    model = load_model(model, uploaded_model)

    if model is not None:
        st.write("Model loaded successfully!")
else:
    st.write("Please upload a model file to continue.")

# Image upload section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Ensure the model is loaded before making a prediction
    if uploaded_model is not None and model is not None:
        species = predict_image(uploaded_image, model)
        st.write(f"Prediction: {species}")
    else:
        st.error("Please upload a model first to classify the image.")

# Optional: Display a description of the model
st.sidebar.title("About the Model")
st.sidebar.write("""
This model is a ResNet18-based Convolutional Neural Network (CNN) trained on a dataset of plant seedlings images. 
The model predicts the species of a given seedling image.
""")
