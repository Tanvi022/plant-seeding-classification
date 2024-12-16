########Plant Seedling Classification Using Streamlit#####
This project demonstrates how to build a machine learning model to classify plant seedlings and deploy it using Streamlit. The model takes images of plant seedlings as input and predicts the class of the plant. It is a great way to learn how to integrate machine learning with a user-friendly web interface.

Project Overview
In this project, we useResNet-18, a deep Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset to classify images of plant seedlings. The model is trained on a dataset of plant seedlings, and we use Streamlit  for users to upload images of seedlings and receive predictions.

Features
Image Classification: Users can upload images of seedlings, and predict's which plant species it is.
Interactive UI: The app provides a simple user interface where users can upload images and view the prediction.
Model Performance: The model provides predictions, giving users insights on its classification.

Requirements
To run the project locally, you need the following dependencies:
Python 3.7 or above
Streamlit
TensorFlow or PyTorch (depending on which framework you used to train the model)
Pillow (for image processing)
Numpy
Matplotlib (for visualizations)
Scikit-learn (optional, for model evaluation)

Install Dependencies
You can install the necessary dependencies using pip:
pip install streamlit tensorflow numpy pillow matplotlib scikit-learn
Alternatively, you can install all dependencies using a requirements.txt file:
pip install -r requirements.txt
requirements.txt
streamlit==1.12.0
tensorflow==2.8.0
numpy==1.21.5
pillow==8.4.0
matplotlib==3.5.0
scikit-learn==1.0.2


Project Structure
The project follows the following directory structure:
How to Run the App
Clone this repository:
git clone https://github.com/yourusername/plant-seedling-classification.git
cd plant-seedling-classification


Run the Streamlit app:
streamlit run app.py
Open your web browser and go to http://localhost:8501 to view the app.
Usage
Once the app is running, follow these steps:

Upload an image of a plant seedling by clicking the "Browse files" button.
After uploading, the model will process the image and predict which plant species it belongs to.
The prediction will be displayed below the uploaded image.
Example Output
Prediction: :Species5

Model Details
Model Architecture : You are using ResNet-18, a deep Convolutional Neural Network (CNN) pre-trained on the ImageNet dataset. This is good for transfer learning, where you fine-tune the pre-trained model to classify your plant seedlings dataset. However, I see that you are replacing the final fully connected (FC) layer to adjust it to the number of classes (12 in your case)

Model Loading: The model weights file is expected to have been saved in .pth format (PyTorch model). The code properly loads the model weights while excluding the final layer (fc.weight, fc.bias) to allow for the new number of output classes.

Image Prediction: After an image is uploaded, it is processed, transformed (resized, normalized), and passed to the model for classification.




