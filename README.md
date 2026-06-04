# CreditGuard AI: Credit Card Fraud Detection System

**AI-powered Credit Card Fraud Detection System using XGBoost, SMOTE, and Python with a professional desktop application for real-time fraud prediction.**

---

# Overview

CreditGuard AI is a Machine Learning-based fraud detection system designed to identify potentially fraudulent credit card transactions with high accuracy.

The project leverages **XGBoost**, **SMOTE**, and advanced preprocessing techniques to address the highly imbalanced nature of fraud datasets. A professional desktop application enables real-time transaction analysis and fraud prediction.

This project demonstrates the complete Machine Learning lifecycle, including:

* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Class Imbalance Handling
* Model Training
* Model Evaluation
* Model Deployment
* Desktop Application Development

---

# Features

## Machine Learning

* Credit Card Fraud Detection
* XGBoost Classification Model
* SMOTE Oversampling
* StandardScaler Normalization
* Model Serialization

## Data Analysis

* Transaction Pattern Analysis
* Fraud Distribution Analysis
* Feature Correlation Analysis
* Data Visualization

## Desktop Application

* Professional GUI Interface
* Real-Time Fraud Prediction
* Risk Assessment System
* User-Friendly Input Forms
* Instant Prediction Results

---

# Problem Statement

Credit card fraud results in significant financial losses for financial institutions and customers worldwide.

One of the major challenges in fraud detection is the highly imbalanced nature of transaction datasets, where fraudulent transactions represent only a small percentage of total transactions.

The objective of this project is to develop an intelligent fraud detection system capable of accurately identifying fraudulent transactions while minimizing false positives.

---

# Machine Learning Workflow

## 1. Data Collection

The project utilizes a credit card transaction dataset containing:

* Transaction Time
* Transaction Amount
* PCA-Transformed Features (V1–V28)
* Fraud Labels

## 2. Data Preprocessing

The preprocessing pipeline includes:

* Missing Value Analysis
* Data Cleaning
* Feature Scaling
* Train-Test Split
* Class Balancing using SMOTE

## 3. Model Training

### XGBoost Classifier

The fraud detection model is trained using the XGBoost algorithm, which provides:

* High Predictive Performance
* Robust Handling of Imbalanced Data
* Excellent Generalization Capability
* Fast Inference Speed

## 4. Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score

### Model Performance

| Metric        | Score     |
| ------------- | --------- |
| ROC-AUC Score | 0.9835    |
| Performance   | Excellent |

---

# Desktop Application

The project includes a professional desktop application developed using Tkinter.

## Application Features

* Modern User Interface
* Transaction Input Form
* Real-Time Fraud Detection
* Risk Assessment Display
* Prediction Visualization

## Prediction Workflow

1. User enters transaction details.
2. Input data is preprocessed.
3. The trained model evaluates the transaction.
4. Fraud prediction is generated.
5. Risk level is displayed to the user.

---

# Project Structure

```text
CreditGuard-AI/
│
├── Credit_Card_Fraud.ipynb          # Model Development Notebook
├── fraud_detection_app.py           # Desktop Application
├── fraud_model.pkl                  # Trained Machine Learning Model
├── amount_scaler.pkl                # Saved StandardScaler
├── creditcard.csv                   # Dataset
├── fraud_detection_dashboard.png    # Application Screenshot
├── install.bat                      # Automated Installation Script
├── requirements.txt                 # Project Dependencies
└── README.md                        # Project Documentation
```

---

# Technology Stack

## Programming Language

* Python

## Machine Learning

* Scikit-Learn
* XGBoost
* Imbalanced-Learn (SMOTE)

## Data Analysis

* Pandas
* NumPy

## Data Visualization

* Matplotlib
* Seaborn

## GUI Development

* Tkinter

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/CreditGuard-AI.git
cd CreditGuard-AI
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python fraud_detection_app.py
```

---

# Business Impact

This solution can help financial institutions:

* Detect suspicious transactions in real time
* Reduce fraud-related financial losses
* Improve payment security
* Increase customer trust
* Support intelligent risk management

---

# Skills Demonstrated

* Machine Learning
* Fraud Analytics
* XGBoost
* SMOTE
* Data Preprocessing
* Feature Engineering
* Model Evaluation
* Model Deployment
* Desktop Application Development
* Data Visualization

---

# Future Enhancements

* Deep Learning-Based Fraud Detection
* Real-Time API Deployment
* Streamlit Web Dashboard
* Cloud Deployment
* Explainable AI (XAI)
* Live Transaction Monitoring

---

# Author

## Mubashar Ahmed Rabbani

**AI Engineer | AI-Powered Full-Stack Web Developer | Machine Learning | Deep Learning | Computer Vision | Building Production-Ready Intelligent Applications**

### Connect With Me

* LinkedIn: Add Your LinkedIn Profile
* GitHub: Add Your GitHub Profile
* Portfolio: Add Your Portfolio Website

---

# Support

If you found this project useful, consider giving it a star on GitHub.

Your support helps motivate the development of more AI and Machine Learning projects.
