# PCOS-Prediction-Using-Machine-Learning-
# 🧬 PCOS Prediction Using Machine Learning

This project predicts the likelihood of Polycystic Ovary Syndrome (PCOS) in women using machine learning models. It leverages PyCaret for automated model comparison and tuning, and explores additional models like Random Forest and XGBoost with manual hyperparameter optimization.

## 📊 Tech Stack

- Python
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- PyCaret
- XGBoost, LightGBM
- TensorFlow/Keras (optional for deep learning exploration)

---

## 📁 Dataset

The dataset used contains medical features relevant to PCOS diagnosis, such as:

- `FSH`, `LH`, `Age`, `BMI`, `AMH`, etc.
- `PCOS` (Target: 0 = No, 1 = Yes)

You can download the dataset from sources like Kaggle or clinical open data repositories.

---

## ⚙️ Features of This Project

- 🧼 **Data Preprocessing**: Missing values handling, encoding, feature scaling
- 🔍 **Exploratory Data Analysis**: Visualization using seaborn/matplotlib
- 🔁 **AutoML with PyCaret**:
  - Auto splits data
  - Compares 15+ classifiers
  - Selects best model automatically
- 🛠 **Manual Model Tuning**:
  - GridSearchCV for Random Forest & XGBoost
- 📈 **Evaluation Metrics**:
  - Accuracy, AUC, Recall, F1 Score, MCC

---

## 📌 Results Summary

| Model                 | Accuracy | AUC   | F1 Score |
|----------------------|----------|-------|----------|
| Logistic Regression  | 0.7118   | 0.6543| 0.3365   |
| Naive Bayes          | 0.7037   | 0.6952| 0.2956   |
| QDA                  | 0.6985   | 0.7016| 0.3027   |
| XGBoost (tuned)      | 0.6697   | 0.6783| 0.4046   |
| Random
