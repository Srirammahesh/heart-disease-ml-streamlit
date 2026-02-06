import sys
import os

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)
def highlight_prediction(row):
    if row["Predicted_HeartDisease"] == row["HeartDisease"]:
        return "background-color: #d4edda; color: black;"  # green
    else:
        return "background-color: #f8d7da; color: black;"  # red


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------
# Model imports
# -------------------------------
from model.logistic_regression import train_model as train_lr
from model.decision_tree import train_model as train_dt
from model.knn import train_model as train_knn
from model.naive_bayes import train_model as train_nb
from model.random_forest import train_model as train_rf
from model.xgboost_model import train_model as train_xgb

# -------------------------------
# Header
# -------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>❤️ Heart Disease Prediction</h1>
    <h4 style='text-align:center; color: gray;'>
    Train on Provided Dataset • Evaluate on Uploaded Test File
    </h4>
    """,
    unsafe_allow_html=True
)

st.success("Application started successfully")

# ======================================================
# 1️⃣ TRAINING PHASE (STATIC TRAIN DATA ONLY)
# ======================================================
st.markdown("## 🧠 Model Training")

df_train = pd.read_csv("data/heart_train.csv")

X_train = df_train.drop("HeartDisease", axis=1)
y_train = df_train["HeartDisease"]

X_train = pd.get_dummies(X_train, drop_first=True)

# Scaling (only for certain models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------
# Sidebar – Model Selection
# -------------------------------
st.sidebar.markdown("## 🔍 Model Selection")

model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# Push content towards bottom
st.sidebar.markdown("<div style='margin-top: 200px;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    **Training file:** heart_train.csv  
    **Samples:** {len(X_train)}  
    **Target:** HeartDisease
    """
)

st.write(f"### Selected Model: **{model_choice}**")

# -------------------------------
# Train Selected Model
# -------------------------------
if model_choice == "Logistic Regression":
    model = train_lr(X_train_scaled, y_train)

elif model_choice == "Decision Tree":
    model = train_dt(X_train, y_train)

elif model_choice == "KNN":
    model = train_knn(X_train_scaled, y_train)

elif model_choice == "Naive Bayes":
    model = train_nb(X_train_scaled, y_train)

elif model_choice == "Random Forest":
    model = train_rf(X_train, y_train)

elif model_choice == "XGBoost":
    model = train_xgb(X_train, y_train)

st.success("Model trained successfully using heart_train.csv")

# ======================================================
# 2️⃣ TESTING PHASE (UPLOAD TEST FILE)
# ======================================================
st.markdown("## 📤 Upload Test Dataset for Evaluation")

uploaded_file = st.file_uploader(
    "Upload Test CSV File (Must include 'HeartDisease' column)",
    type=["csv"]
)

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)

    if "HeartDisease" not in df_test.columns:
        st.error("Uploaded test file must contain 'HeartDisease' column.")
        st.stop()

    X_test = df_test.drop("HeartDisease", axis=1)
    y_test = df_test["HeartDisease"]

    X_test = pd.get_dummies(X_test, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if model_choice in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------------
    # Per-sample Predictions
    # -------------------------------
    st.markdown("## 📋 Per-Record Predictions")

    results_df = df_test.copy()
    results_df["Predicted_HeartDisease"] = y_pred
    results_df["Prediction_Probability"] = y_prob.round(3)

    # Reorder columns
    cols = (
        ["Predicted_HeartDisease", "Prediction_Probability", "HeartDisease"]
        + [c for c in results_df.columns if c not in ["Predicted_HeartDisease", "Prediction_Probability", "HeartDisease"]]
    )
    results_df = results_df[cols]

    # Apply conditional styling ONLY to Predicted_HeartDisease column
    styled_df = results_df.style.apply(
        lambda row: [
            highlight_prediction(row) if col == "Predicted_HeartDisease" else ""
            for col in results_df.columns
        ],
        axis=1
    )

    st.dataframe(styled_df, use_container_width=True)



    # -------------------------------
    # Metrics
    # -------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.markdown("## 📊 Model Performance on Uploaded Test Data")

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1-score", f"{f1:.3f}")
    c5.metric("AUC", f"{auc:.3f}")
    c6.metric("MCC", f"{mcc:.3f}")

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    st.markdown("## 🔁 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: No Disease", "Actual: Disease"],
        columns=["Predicted: No", "Predicted: Yes"]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.caption(
        """
        - **False Negatives** are critical in healthcare as they indicate missed disease cases.
        """
    )
