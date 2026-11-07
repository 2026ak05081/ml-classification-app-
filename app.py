import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Classification App")

st.title("Adult Income Classification Dashboard")

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

model_paths = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "SVM": "models/svm.pkl"
}

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

if uploaded_file:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.write(df.head())

        target_column = "income"

        if target_column not in df.columns:
            st.error(
                "income column not found in uploaded file"
            )
            st.stop()

        X = df.drop(target_column, axis=1)

        y = df[target_column]

        model = joblib.load(
            model_paths[model_choice]
        )

        st.write("Model Loaded Successfully")

        y_pred = model.predict(X)

        accuracy = accuracy_score(y, y_pred)

        try:
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

        except:
            precision = precision_score(
                y,
                y_pred,
                average="weighted"
            )

            recall = recall_score(
                y,
                y_pred,
                average="weighted"
            )

            f1 = f1_score(
                y,
                y_pred,
                average="weighted"
            )

        mcc = matthews_corrcoef(
            y,
            y_pred
        )

        metrics = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
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

        st.subheader("Metrics")

        st.dataframe(metrics)

        st.subheader(
            "Classification Report"
        )

        report = classification_report(
            y,
            y_pred,
            output_dict=True
        )

        st.dataframe(
            pd.DataFrame(report).transpose()
        )

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)

    except Exception as e:

        st.error(
            f"ERROR WHILE RUNNING MODEL: {e}"
        )

        st.write(
            "Most likely cause: model was trained using preprocessing/scaling that is missing in app.py"
        )