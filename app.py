import streamlit as st
import pandas as pd
import numpy as np
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

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page Title
# --------------------------------------------------

st.title("ML Classification Model Dashboard")

st.write("Adult Income Prediction Assignment")

# --------------------------------------------------
# Select Model
# --------------------------------------------------

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

# --------------------------------------------------
# Model Paths
# --------------------------------------------------

model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "SVM": "models/svm.pkl"
}

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(df.head())

    # ----------------------------------------------
    # Target Column
    # ----------------------------------------------

    target_column = "income"

    X = df.drop(target_column, axis=1)

    y = df[target_column]

    # ----------------------------------------------
    # Load Model
    # ----------------------------------------------

    model = joblib.load(model_paths[model_choice])

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    y_pred = model.predict(X)

    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(
        y,
        y_pred,
        pos_label=">50K"
    )

    recall = recall_score(
        y,
        y_pred,
        pos_label=">50K"
    )

    f1 = f1_score(
        y,
        y_pred,
        pos_label=">50K"
    )

    mcc = matthews_corrcoef(
        y,
        y_pred
    )

    st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ],
        "Value": [
            accuracy,
            precision,
            recall,
            f1,
            mcc
        ]
    })

    st.dataframe(metrics_df)

    # ----------------------------------------------
    # Classification Report
    # ----------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # ----------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.success(
        f"{model_choice} evaluation completed successfully!"
    )