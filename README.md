# Heart Disease Prediction using Machine Learning

## Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict the presence of heart disease in patients based on clinical and physiological attributes. Six different classification algorithms are implemented and compared using standard evaluation metrics. The models are further deployed through an interactive Streamlit web application.

## Dataset Description

The Heart Disease Prediction dataset is obtained from Kaggle (UCI repository). It contains 918 patient records with 13 clinical features such as age, sex, chest pain type, resting blood pressure, cholesterol level, maximum heart rate, exercise-induced angina, and ST slope. The target variable indicates whether a patient has heart disease (1) or not (0). The dataset is well balanced and suitable for binary classification tasks.

## Models Used and Evaluation Metrics

The following machine learning models are implemented on the same dataset:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Random Forest (Ensemble)
* XGBoost (Ensemble)

Each model is evaluated using the following metrics:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

## Model Performance Comparison

| ML Model            | Accuracy | AUC | Precision | Recall | F1 | MCC |
| ------------------- | -------- | --- | --------- | ------ | -- | --- |
| Logistic Regression |          |     |           |        |    |     |
| Decision Tree       |          |     |           |        |    |     |
| KNN                 |          |     |           |        |    |     |
| Naive Bayes         |          |     |           |        |    |     |
| Random Forest       |          |     |           |        |    |     |
| XGBoost             |          |     |           |        |    |     |

## Model-wise Observations

| ML Model            | Observation                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs well on this balanced dataset and provides a strong baseline, though it struggles with non-linear patterns.   |
| Decision Tree       | Captures non-linear relationships but is prone to overfitting without depth control.                                   |
| KNN                 | Shows reasonable performance after feature scaling, but is sensitive to the choice of K and computationally expensive. |
| Naive Bayes         | Fast and simple, but its independence assumption limits performance when features are correlated.                      |
| Random Forest       | Improves performance and stability by aggregating multiple trees, reducing overfitting.                                |
| XGBoost             | Achieves the best overall performance by sequentially correcting errors, showing strong generalization and robustness. |

## Streamlit Application

The Streamlit web application allows users to:

* Upload a CSV test dataset
* Select a trained machine learning model
* View evaluation metrics
* Visualize the confusion matrix / classification report

## How to Run the Project

1. Clone the repository
2. Install dependencies using `pip install -r requirements.txt`
3. Run the Streamlit app using `streamlit run app.py`

## Deployment

The application is deployed using Streamlit Community Cloud and can be accessed via the provided live link in the assignment submission.
