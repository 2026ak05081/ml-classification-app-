import streamlit as st


import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

st.title("Adult Income Classification")

uploaded_file = st.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM"
    ]
)

model_files = {
    "Logistic Regression":"models/logistic_regression.pkl",
    "Decision Tree":"models/decision_tree.pkl",
    "KNN":"models/knn.pkl",
    "Naive Bayes":"models/naive_bayes.pkl",
    "SVM":"models/svm.pkl"
}

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    # Define the feature matrix and target before the selected model uses them.
    X, y = data.drop(columns=["income"]), data["income"]

    if model_choice == "Naive Bayes":
        model = joblib.load(model_files[model_choice])
        preprocessor = joblib.load("models/preprocessor.pkl")
        X = preprocessor.transform(X).toarray()
    else:
        model = joblib.load(model_files[model_choice])

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.write("Accuracy:", accuracy_score(y, y_pred))
    st.write("Precision:", precision_score(y, y_pred, pos_label=">50K"))
    st.write("Recall:", recall_score(y, y_pred, pos_label=">50K"))
    st.write("F1:", f1_score(y, y_pred, pos_label=">50K"))
    st.write("AUC:", roc_auc_score(y, y_prob))
    st.write("MCC:", matthews_corrcoef(y, y_pred))
    st.write(confusion_matrix(y, y_pred))
    st.text(classification_report(y, y_pred))