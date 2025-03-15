# Customer Churn Prediction - ANN

This project implements a **Customer Churn Prediction** model using an **Artificial Neural Network (ANN)** built with TensorFlow and Keras. The goal is to predict whether a customer will leave a service based on various features.

## Project Overview

In this project, we train an artificial neural network (ANN) using a dataset of customer information. The model predicts the likelihood of customer churn (whether a customer will leave or not) based on input features such as credit score, gender, geography, and more.

The project consists of two parts:
1. **Training the ANN model** using customer data.
2. **Deploying a Streamlit app** that allows users to input customer data and receive a prediction on whether the customer will churn.


## Installation

To run this project locally or deploy it, follow the steps below.

### 1. Clone the repository

```bash
git clone https://github.com/DhanushGD/ANN-CLASSIFICATION-CHURN.git
cd ANN-CLASSIFICATION-CHURN
```

### 2. Install the required dependencies
Make sure you have Python 3.6+ installed. Then, use pip to install the necessary libraries. You can install them using the requirements.txt file.

```bash
pip install -r DAY - 4 ANN project/requirements.txt
```

### 3. Run the training script
Run the training.py script to train the model if it hasn't been trained yet. This will create the model.h5, scaler.pkl, label_encoder_gender.pkl, and onehot_encoder_geo.pkl files necessary for the Streamlit app.

```bash
python DAY - 4 ANN project/training.py
```
This script will:
Load and preprocess the dataset.
Train an ANN model.
Save the trained model and preprocessing objects.

### 4. Start the Streamlit app
After training the model, you can run the Streamlit app to predict customer churn.
```bash
streamlit run DAY - 4 ANN project/app.py
```
Once the app is running, open your browser and navigate to the local Streamlit app URL (usually http://localhost:8501).

### 5. View TensorBoard Logs (Optional)
If you want to view the training progress and visualize metrics like accuracy and loss, you can run TensorBoard by navigating to the logs directory and starting the server:
```bash
tensorboard --logdir="DAY - 4 ANN project/logs/fit"
```
Visit http://localhost:6006 in your browser to view TensorBoard logs.
