# ❤️ Heart Disease Prediction – Machine Learning Assignment

## 📌 Overview

This project implements and compares multiple **machine learning classification models** to predict the presence of heart disease. The solution strictly follows the assignment instructions by separating **training** and **testing** phases and by displaying all required evaluation metrics and visualizations through an interactive **Streamlit application**.

---

## 🎯 Objective

* Train ML models using a **provided training dataset**
* Evaluate trained models on an **externally uploaded test dataset**
* Display performance metrics and visual diagnostics
* Allow per-record prediction inspection

---

## 📂 Project Structure

```
MLAssignment/
│
├── app.py                     # Streamlit application
├── data/
│   └── heart_train.csv        # Training dataset (used ONLY for training)
│
├── model/                     # Model training modules
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── random_forest.py
│   └── xgboost_model.py
│
└── README.md                  # Project documentation
```

---

## 🧠 Machine Learning Models Implemented

The following models are implemented as **separate, modular Python files**:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Random Forest
* XGBoost

Each model exposes a `train_model()` function and is trained using the same training dataset for fair comparison.

---

## 🔁 Training & Testing Strategy

### Training

* **Dataset used:** `data/heart_train.csv`
* **Training approach:**

  * Entire dataset is used for training
  * No internal train–test split

### Testing

* Testing is performed using a **CSV file uploaded via the UI**
* The uploaded test file **must contain the target column `HeartDisease`**
* All evaluation metrics are computed **only on the uploaded test dataset**

This approach simulates a **real-world ML evaluation scenario** using unseen data.

---

## 📊 Evaluation Metrics Displayed

For the uploaded test dataset, the following metrics are computed and displayed:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)

All metrics are calculated using standard `scikit-learn` evaluation functions.

---

## 📋 Per-Record Prediction Display

For each row in the uploaded test dataset, the app displays:

* Actual label (`HeartDisease`)
* Predicted label (`Predicted_HeartDisease`)
* Prediction probability

This allows **row-level error analysis** and improves transparency.

---

## 🖥️ User Interface Features

* Sidebar model selection
* Bottom-aligned dataset information panel
* Metric cards for quick performance assessment
* Interactive data tables
* Clear success and error messages

All UI enhancements are intentionally minimal and aligned with assignment requirements.

---

## ▶️ How to Run the Application

1. Install required dependencies:

```
pip install streamlit scikit-learn pandas numpy matplotlib xgboost
```

2. Run the Streamlit app:

```
streamlit run app.py
```

3. Select a model from the sidebar
4. Upload a test CSV file containing the `HeartDisease` column
5. View metrics, confusion matrix, and per-record predictions

---

## Model Performance Comparison Table

| ML Model                  | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression       | 0.864    | 0.897 | 0.873     | 0.881  | 0.877    | 0.725 |
| Decision Tree             | 0.810    | 0.897 | 0.811     | 0.851  | 0.831    | 0.615 |
| K-Nearest Neighbors (KNN) | 0.853    | 0.886 | 0.843     | 0.901  | 0.871    | 0.704 |
| Naive Bayes               | 0.848    | 0.909 | 0.841     | 0.891  | 0.865    | 0.692 |
| Random Forest (Ensemble)  | 0.848    | 0.898 | 0.835     | 0.901  | 0.867    | 0.693 |
| XGBoost (Ensemble)        | 0.832    | 0.901 | 0.830     | 0.871  | 0.850    | 0.659 |

---

## Observations on Model Performance

| ML Model                  | Observation about model performance                                                                                                                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression       | Achieved strong and balanced performance across all metrics, with the highest overall accuracy and MCC. Its linear decision boundary generalizes well for this dataset and provides good interpretability, making it a reliable baseline model. |
| Decision Tree             | Showed lower accuracy and MCC compared to other models, indicating a tendency to overfit the training patterns. While recall is reasonable, the overall generalization performance is weaker due to sensitivity to data variations.             |
| K-Nearest Neighbors (KNN) | Performed well with high recall and F1-score, indicating strong ability to detect positive heart disease cases. However, slightly lower AUC suggests sensitivity to feature scaling and neighborhood selection.                                 |
| Naive Bayes               | Achieved the highest AUC score, demonstrating excellent probabilistic separation between classes. Despite its strong recall, assumptions of feature independence slightly limited precision and overall accuracy.                               |
| Random Forest (Ensemble)  | Delivered stable and robust performance with high recall and F1-score. Ensemble learning helped reduce variance compared to a single decision tree, resulting in improved generalization.                                                       |
| XGBoost (Ensemble)        | Provided competitive performance with high AUC and balanced metrics. While slightly lower in accuracy than Random Forest, it benefits from gradient boosting’s ability to capture complex feature interactions.                                 |


## ✅ Compliance With Assignment Instructions

✔ Uses only provided training data for model training
✔ Testing performed on external uploaded dataset
✔ Required metrics and confusion matrix included
✔ No unnecessary models or overengineering
✔ Clear explanations and visualizations

---

## 🧠 Key Learning Outcomes

* Proper ML pipeline separation (training vs testing)
* Importance of stratification and evaluation metrics
* Interpretability in healthcare ML problems
* Building reproducible and explainable ML applications

---

## 👤 Author

**Sriram Mahesh**

---

## 📌 Notes

* Uploaded test file must match training feature structure
* Categorical variables are handled using one-hot encoding
* Feature alignment is enforced during prediction to avoid errors

---

**End of README**
