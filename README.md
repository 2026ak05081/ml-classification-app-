
ML Classification App
Problem Statement
The objective of this project is to build and compare multiple Machine Learning classification models on a public dataset and deploy them as an interactive Streamlit web application.

The application allows users to upload test data, select a trained model, and view prediction performance metrics, confusion matrix, and classification report without retraining the model.

Dataset Description
Dataset Name: Adult Income Dataset

Source: UCI Machine Learning Repository

Dataset Link: https://archive.ics.uci.edu/ml/datasets/adult

Dataset Information
Total Instances: 48,842
Total Features: 14
Problem Type: Binary Classification
Target Variable: Income
Target Classes
<=50K
50K

Features Used
age
workclass
fnlwgt
education
education-num
marital-status
occupation
relationship
race
sex
capital-gain
capital-loss
hours-per-week
native-country
GitHub Repository Link
Repository URL:

https://github.com/2026ak05081/ml-classification-app-

Live Streamlit Application
Streamlit URL:

PASTE_YOUR_STREAMLIT_LINK_HERE

Project Structure
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── models/
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
│
├── dataset/
│   └── adult.csv
│
└── notebooks/
    └── model_training.ipynb
Models Used
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbor (KNN)
Gaussian Naive Bayes
Support Vector Machine (SVM)
Evaluation Metrics
The following evaluation metrics were calculated for all models:

Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)

Model Observations
ML Model Name	Observation
Logistic Regression	Logistic Regression produced stable and interpretable results with good overall performance.
Decision Tree	Decision Tree was easy to interpret but showed signs of overfitting compared to other models.
KNN	KNN performed reasonably well but was sensitive to preprocessing and scaling.
Naive Bayes	Naive Bayes trained very quickly and provided good baseline performance.
Support Vector Machine	SVM achieved strong classification performance and handled class boundaries effectively.
Overall Winner	UPDATE AFTER FILLING THE METRICS TABLE