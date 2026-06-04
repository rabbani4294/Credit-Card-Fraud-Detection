# Credit-Card-Fraud-Detection
AI-powered Credit Card Fraud Detection System using XGBoost, SMOTE, and Python with a professional desktop application for real-time fraud prediction.
.

📌 Overview

CreditGuard AI is a Machine Learning-powered fraud detection system designed to identify potentially fraudulent credit card transactions with high accuracy.

The project leverages XGBoost, SMOTE, and advanced preprocessing techniques to tackle the highly imbalanced nature of fraud datasets. A user-friendly desktop application enables real-time transaction analysis and fraud prediction.

This project demonstrates the complete Machine Learning lifecycle:

Data Preprocessing
Exploratory Data Analysis
Feature Engineering
Class Imbalance Handling
Model Training
Model Evaluation
Model Deployment
Desktop Application Development
🚀 Features
Machine Learning
Credit Card Fraud Detection
XGBoost Classification Model
SMOTE Oversampling
StandardScaler Normalization
Model Serialization
Data Analysis
Transaction Pattern Analysis
Fraud Distribution Analysis
Feature Correlation Analysis
Data Visualization
Desktop Application
Professional GUI Interface
Real-Time Fraud Prediction
Risk Assessment System
User-Friendly Input Forms
Instant Prediction Results
🎯 Problem Statement

Credit card fraud causes billions of dollars in financial losses annually.

A major challenge is that fraudulent transactions represent only a tiny fraction of total transactions, making the dataset highly imbalanced.

The goal of this project is to develop an intelligent system capable of accurately identifying fraudulent transactions while minimizing false positives.

🧠 Machine Learning Workflow
1. Data Collection

The project utilizes a credit card transaction dataset containing:

Transaction Time
Transaction Amount
PCA-transformed Features (V1–V28)
Fraud Labels
2. Data Preprocessing

The preprocessing pipeline includes:

Missing Value Analysis
Feature Scaling
Data Cleaning
Train-Test Split
Class Balancing using SMOTE
3. Model Training

The fraud detection model is trained using:

XGBoost Classifier

Benefits:

High Performance
Robust to Imbalanced Data
Excellent Generalization
Fast Prediction Speed
4. Model Evaluation

Evaluation metrics include:

Accuracy
Precision
Recall
F1 Score
ROC-AUC Score
Model Performance
Metric	Score
ROC-AUC	0.9835
Performance	Excellent
🖥️ Desktop Application

The project includes a professional desktop application developed using Tkinter.

Application Features
Modern User Interface
Transaction Input Form
Real-Time Fraud Detection
Risk Assessment Display
Prediction Visualization
Prediction Flow
Enter transaction details
Data preprocessing is applied
Trained model evaluates transaction
Prediction is generated
Risk level is displayed
📂 Project Structure
CreditGuard-AI/
│
├── Credit_Card_Fraud.ipynb
├── fraud_detection_app.py
├── fraud_model.pkl
├── amount_scaler.pkl
├── creditcard.csv
├── fraud_detection_dashboard.png
├── install.bat
├── requirements.txt
└── README.md
🛠️ Technology Stack
Programming Language
Python
Machine Learning
Scikit-Learn
XGBoost
Imbalanced-Learn (SMOTE)
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
GUI Development
Tkinter
⚙️ Installation
Clone Repository
git clone https://github.com/your-username/CreditGuard-AI.git
cd CreditGuard-AI
Install Dependencies
pip install -r requirements.txt
Run Application
python fraud_detection_app.py
📊 Business Impact

This solution can help financial institutions:

Detect suspicious transactions instantly
Reduce fraud-related losses
Improve payment security
Increase customer trust
Support intelligent risk management
🎓 Skills Demonstrated
Machine Learning
Fraud Analytics
XGBoost
SMOTE
Data Preprocessing
Feature Engineering
Model Evaluation
Model Deployment
Desktop Application Development
Data Visualization
🔮 Future Enhancements
Deep Learning-Based Fraud Detection
Real-Time API Deployment
Streamlit Web Dashboard
Cloud Deployment
Explainable AI (XAI)
Live Transaction Monitoring


Author
Mubashar Ahmed Rabbani

AI Engineer | AI-Powered Full-Stack Web Developer | Machine Learning • Deep Learning • Computer Vision | Building Production-Ready Intelligent Applications

Connect With Me
LinkedIn: Add Your LinkedIn Profile
GitHub: Add Your GitHub Profile
Portfolio: Add Your Portfolio Website
⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

Your support helps motivate the development of more AI and Machine Learning projects.
