
ML Classification App - Adult Income Prediction
Problem Statement
The objective of this project is to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes. This is a binary classification problem where the target variable is income category (<=50K or >50K).
Five machine learning classification algorithms were implemented, evaluated, and compared using multiple performance metrics. A Streamlit web application was developed to allow users to upload test data and evaluate the trained models.
Dataset Description
Dataset Name: Adult Census Income Dataset
Source: Kaggle / UCI Machine Learning Repository
Classification Type: Binary Classification
Target Variable:
Target Variable:
Income
<=50K
>50K
Number of Instances: 48,842
Number of Features: 14
Features
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

Target Column
Income

GitHub Repository Link
https://github.com/2026ak05081/ml-classification-app-(inside master branch)
Streamlit Application Link
https://ml-classification-app-models.streamlit.app/
Models Used
The following machine learning classification models were implemented:
1.	Logistic Regression
2.	Decision Tree Classifier
3.	K-Nearest Neighbors (KNN)
4.	Gaussian Naive Bayes
5.	Support Vector Machine (SVM)

Model Performance Comparison
ML Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8424	0.8980	0.7201	0.5987	0.6538	0.5568
Decision Tree	0.8049	0.7471	0.6028	0.6313	0.6167	0.4862
KNN	0.8256	0.8545	0.6647	0.6027	0.6322	0.5193
Naive Bayes	0.4974	0.7039	0.3241	0.9407	0.4821	0.2811
Support Vector Machine	0.8467	0.8948	0.7382	0.5940	0.6583	0.5665


Observations on Model Performance
ML Model Name	Observation about Model Performance
Logistic Regression	Performed very well with strong Accuracy, AUC, and MCC scores. It proved to be a reliable baseline model for this dataset.
Decision Tree	Achieved moderate performance. It had better Recall than some models but lower Accuracy and AUC, indicating possible overfitting.
KNN	Produced balanced results with acceptable Accuracy and AUC. Performance was better than Decision Tree but lower than Logistic Regression and SVM.
Naive Bayes	Achieved the highest Recall (94.07%) but had very low Precision and Accuracy. It generated many false positives, making it less suitable for this dataset.
Support Vector Machine	Achieved the highest Accuracy (84.67%), Precision (73.82%), F1 Score (65.83%), and MCC (0.5665). It demonstrated the best overall classification performance.
Overall winner for Dataset	Support Vector Machine (SVM)

Support Vector Machine (SVM) emerged as the best-performing model for the Adult Income Dataset.
Reasons
•	Highest Accuracy: 84.67%
•	Highest Precision: 73.82%
•	Highest F1 Score: 65.83%
•	Highest MCC Score: 0.5665
•	Strong AUC Score: 0.8948
These metrics indicate that SVM provides the most balanced and reliable predictions among all implemented classifiers.

Project Structure
ml-classification-app/
│── app.py
│── requirements.txt
│── README.md
│── test_data.csv
│── model/
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── scaler.pkl
│── notebooks/
│   └── training.ipynb
│── src/
│   └── train_models.py

Streamlit Application Features
The Streamlit application includes:
Dataset Upload
Users can upload a CSV file containing test data.
Model Selection
Users can choose among:
•	Logistic Regression
•	Decision Tree
•	KNN
•	Naive Bayes
•	SVM
Evaluation Metrics Display
The application displays:
•	Accuracy
•	AUC Score
•	Precision
•	Recall
•	F1 Score
•	MCC Score
Classification Report
Detailed classification report is displayed.
Confusion Matrix
Visual confusion matrix is displayed using a heatmap.
________________________________________
Requirements
Plain Text
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
________________________________________
Conclusion
This project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, model comparison, deployment, and user interaction through a Streamlit application. Among all implemented classifiers, Support Vector Machine (SVM) achieved the best overall performance on the Adult Income Dataset and was selected as the final model
