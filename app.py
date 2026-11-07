from pathlib import Path

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
    classification_report,
)

st.set_page_config(page_title="ML Classification App")

st.title("Adult Income Classification Dashboard")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

model_choice = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "SVM",
    ],
)

model_paths = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
    "Decision Tree": MODELS_DIR / "decision_tree.pkl",
    "KNN": MODELS_DIR / "knn.pkl",
    "Naive Bayes": MODELS_DIR / "naive_bayes.pkl",
    "SVM": MODELS_DIR / "svm.pkl",
}


def get_model_input(X):
    """Apply an optional saved preprocessor to raw features."""
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"
    if not preprocessor_path.exists():
        return X

    preprocessor = joblib.load(preprocessor_path)
    X_processed = preprocessor.transform(X)

    try:
        X_processed = X_processed.toarray()
    except Exception:
        pass

    return X_processed


uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write(df.head())

        target_column = "income"

        if target_column not in df.columns:
            st.error("income column not found in uploaded file")
            st.stop()

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        model = joblib.load(model_paths[model_choice])
        X_model = get_model_input(X)

        try:
            y_pred = model.predict(X_model)
        except Exception:
            y_pred = model.predict(X)

        st.write("Model Loaded Successfully")

        accuracy = accuracy_score(y, y_pred)

        try:
            precision = precision_score(y, y_pred, pos_label=">50K")
            recall = recall_score(y, y_pred, pos_label=">50K")
            f1 = f1_score(y, y_pred, pos_label=">50K")
        except Exception:
            precision = precision_score(y, y_pred, average="weighted")
            recall = recall_score(y, y_pred, average="weighted")
            f1 = f1_score(y, y_pred, average="weighted")

        mcc = matthews_corrcoef(y, y_pred)

        metrics = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1", "MCC"],
                "Value": [accuracy, precision, recall, f1, mcc],
            }
        )

        st.subheader("Metrics")
        st.dataframe(metrics)

        st.subheader("Classification Report")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"ERROR WHILE RUNNING MODEL: {e}")
        st.write(
            "Most likely cause: model was trained using preprocessing/scaling that is missing in app.py"
        )