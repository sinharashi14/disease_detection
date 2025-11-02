# 🧠 Diabetes Prediction using Artificial Neural Network (ANN)

## 📋 Project Overview
This project implements a **Computational Intelligence model** using an **Artificial Neural Network (ANN)** to predict whether a patient is likely to have diabetes based on various medical diagnostic features such as glucose level, BMI, blood pressure, and age.  
The model is built using **TensorFlow (Keras)** and **Scikit-learn**, trained on the **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository.

---

## 🎯 Objective
To identify the likelihood of diabetes in patients by analyzing medical parameters using an ANN model that learns patterns from data.

---

## 🧩 Dataset
- **Source:** [UCI Machine Learning Repository – Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- **Attributes:**
  - Pregnancies  
  - Glucose  
  - BloodPressure  
  - SkinThickness  
  - Insulin  
  - BMI  
  - DiabetesPedigreeFunction  
  - Age  
  - Outcome (0 = No Diabetes, 1 = Diabetes)

---

## ⚙️ Tech Stack
- **Language:** Python 🐍  
- **Libraries Used:**
  - TensorFlow / Keras  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - Matplotlib  
  - Seaborn  

---

## 🚀 Implementation Steps
1. **Load Dataset** – Imported using Pandas DataFrame.  
2. **Preprocessing** – Splitting and scaling data with `StandardScaler`.  
3. **Model Building** – Multi-Layer Perceptron:
   - Input layer: 8 neurons  
   - Hidden layers: 12 and 8 neurons (ReLU)  
   - Output layer: 1 neuron (Sigmoid)  
4. **Training** – 5 epochs, batch size = 10, optimizer = `adam`.  
5. **Evaluation** – Accuracy, confusion matrix, and classification report.  
6. **Visualization** – Accuracy and loss plotted using Matplotlib.  
7. **Model Saving** – Model saved as `diabetes_ann_model.h5`.

---

## 📊 Results
- **Accuracy:** ~80–85% (varies slightly per run)  
- **Outputs:**
  - Confusion Matrix  
  - Classification Report  
  - Accuracy and Loss Graphs  

---

## 🧠 Key Learnings
- Designing and training **Artificial Neural Networks**.  
- Data scaling for faster convergence.  
- Role of activation and loss functions in classification.  
- Using **TensorFlow Keras Sequential Models** effectively.

---

## 🧾 How to Run
```bash
# Clone the repository
git clone https://github.com/<your-username>/diabetes-ann-prediction.git
cd diabetes-ann-prediction

# Install dependencies
pip install -r requirements.txt

# Run the model
python diabetes_ann.py

📦 Output Files
diabetes_ann_model.h5 → Trained ANN model
Accuracy and loss graphs displayed during execution

💡 Future Enhancements
Add dropout regularization to avoid overfitting
Increase hidden layers for improved accuracy
Deploy model via Flask or Streamlit for real-time predictions
