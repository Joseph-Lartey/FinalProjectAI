# FinalProjectAI
# Spam Filter Web Application

This project is a web application that uses a machine learning model to predict whether a given text message is likely spam or ham (not spam). The model is trained on a dataset of spam and non-spam messages using a Long Short-Term Memory (LSTM) neural network.

## Overview

The web application is built using Streamlit, a Python library for creating interactive web applications.
Users can input a text message, and the model will predict whether it is spam or ham.

## Usage

1. **Install Dependencies:**
   Make sure you have the required dependencies installed. You can install them using:

pip install streamlit
pip install pandas
pip install pickle
pip install  tensorflow
pip install scikit-learn
pip install lime
pip install  matplotlib

streamlit: The main library for creating the web application.
pandas: Used for data manipulation and handling DataFrames.
pickle: Required for loading and saving the tokenizer.
tensorflow: The machine learning library for building and training neural networks.
scikit-learn: Used for preprocessing and standardizing the data.
lime: The Local Interpretable Model-agnostic Explanations (LIME) library for model interpretability.
matplotlib: Used for creating visualizations, particularly in the explanation part.

**2. Run the Application:**

Execute the Streamlit application with the following command:

streamlit run Spam_Deployment.py

This will start a local development server, and you can access the application in your web browser at http://localhost:8501.

**3. Interact with the Application**

Input a message in the provided text box.
Click the "Predict" button to view the model's prediction.

**4 .Files and Structure**

Spam_Deployment.py: The primary Streamlit application file.
spam_model.h5: The LSTM model saved in HDF5 format.

tokenizer.pkl: A Pickle file containing the tokenizer used for text preprocessing.
Model Training

The machine learning model is trained using TensorFlow and Keras. 
The training process encompasses preprocessing the text data, defining and training the LSTM model, and evaluating its performance.

Additional Information
For more details, watch the project's YouTube video (https://www.youtube.com/watch?v=zA2Ht4xd3gQ) explaining the application's functionality.

Note: This application is for demonstration purposes, and the model's predictions may not be perfect.





