# disease_detection
disease detection using ANN model 
🧠 Diabetes Prediction using Artificial Neural Network (ANN)
📋 Project Overview

This project implements a Computational Intelligence model using an Artificial Neural Network (ANN) to predict whether a patient is likely to have diabetes based on various medical diagnostic features such as glucose level, BMI, blood pressure, and age.
The model is built using TensorFlow (Keras) and Scikit-learn, trained on a subset of the Pima Indians Diabetes Dataset from the UCI Machine Learning Repository.

🎯 Objective
To identify the likelihood of diabetes in patients by analyzing medical parameters using an ANN model that learns patterns from data.
🧩 Dataset
Source: UCI Machine Learning Repository – Pima Indians Diabetes Dataset

Attributes:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

Outcome (0 = No Diabetes, 1 = Diabetes)

⚙️ Tech Stack

Language: Python 🐍

Libraries:

TensorFlow / Keras

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

🚀 Implementation Steps

Load Dataset – Data loaded directly into a Pandas DataFrame.

Preprocessing – Split into training and test sets, scaled using StandardScaler.

Model Building – Multi-Layer Perceptron with:

Input layer: 8 neurons

Hidden layers: 12 and 8 neurons (ReLU activation)

Output layer: 1 neuron (Sigmoid activation)

Training – Model trained for 5 epochs with batch size of 10 using adam optimizer.

Evaluation – Accuracy, confusion matrix, and classification report generated.

Visualization – Accuracy and loss curves plotted using Matplotlib.

Model Saving – Saved trained model as diabetes_ann_model.h5.

📊 Results

Accuracy: ~80–85% (varies slightly per run)

Model Output:

Confusion Matrix

Classification Report

Accuracy and Loss graphs

🧠 Key Learnings

Understanding of Artificial Neural Network architecture.

Importance of data scaling for convergence.

Use of activation functions, loss functions, and optimizers.

Practical experience with TensorFlow Keras sequential models.

🧾 How to Run
# Clone the repository
git clone https://github.com/<your-username>/diabetes-ann-prediction.git
cd diabetes-ann-prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook or Python script
python diabetes_ann.py

📦 Output Files

diabetes_ann_model.h5 → Saved trained ANN model

Accuracy and loss graphs displayed during execution

💡 Future Enhancements

Add more hidden layers or neurons for higher accuracy

Implement dropout regularization to prevent overfitting

Deploy the model using a simple web interface (Flask/Streamlit)
