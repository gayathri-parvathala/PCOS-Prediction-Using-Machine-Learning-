
# PCOS Prediction Using Machine Learning

This project aims to predict the likelihood of Polycystic Ovary Syndrome (PCOS) in women using various machine learning models. It combines traditional model development with PyCaret for automated comparison and evaluation. The entire workflow is implemented and executed in Google Colab.

## Objective

To develop a predictive model that accurately identifies PCOS based on clinical and demographic data. This tool can support healthcare professionals in early diagnosis and screening.

## Dataset

The dataset includes the following types of features:
- Clinical indicators (e.g., LH, FSH, AMH levels)
- Demographic data (e.g., age, weight, BMI)
- Binary target variable: `PCOS` (1 = has PCOS, 0 = no PCOS)

## Features

- Data cleaning: handling missing values, removing duplicates
- Feature scaling and encoding
- Exploratory Data Analysis (EDA)
- PyCaret-based model comparison (auto-selects best models)
- Manual hyperparameter tuning for Random Forest and XGBoost using GridSearchCV
- Comprehensive model evaluation using metrics like Accuracy, AUC, F1 Score, Precision, Recall

## Tools and Libraries

This project is developed entirely in **Google Colab**, utilizing the following libraries:
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- xgboost
- pycaret

No external environment setup or installation is needed beyond `pip install` commands executed directly within the Colab notebook.

## Workflow

1. **Load and Preprocess Data**
   - Remove duplicates
   - Handle missing values
   - Label encode categorical features
   - Standardize numerical columns

2. **Train-Test Split**
   - Stratified 80/20 split to maintain class distribution

3. **Model Comparison**
   - Use PyCaret to automatically evaluate 15+ machine learning classifiers
   - Select the top-performing models based on Accuracy and AUC

4. **Manual Tuning**
   - Apply GridSearchCV to optimize Random Forest and XGBoost parameters

5. **Model Evaluation**
   - Evaluate all models on the test set using classification metrics
   - Generate comparison tables and visualizations

## Sample Results

| Model                         | Accuracy | AUC    | Recall | F1 Score |
|------------------------------|----------|--------|--------|----------|
| Logistic Regression          | 0.7118   | 0.6543 | 0.2359 | 0.3365   |
| LDA                          | 0.7117   | 0.6559 | 0.2192 | 0.3163   |
| Naive Bayes                  | 0.7037   | 0.6952 | 0.1955 | 0.2956   |
| QDA                          | 0.6985   | 0.7016 | 0.2115 | 0.3027   |
| Random Forest (Tuned)        | 0.6881   | 0.6663 | 0.3712 | 0.3984   |
| XGBoost (Tuned)              | 0.6697   | 0.6783 | 0.3724 | 0.4046   |

## Usage

Open the project in Google Colab and execute the notebook step by step. All required libraries can be installed using `pip` cells at the top of the notebook.

## File Structure

```
.
├── pcos_prediction.ipynb   # Main Colab notebook
└── README.md               # Project overview
```

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- Dataset contributors and hosting platforms
- PyCaret community for simplifying ML workflows
- Scikit-learn and open-source ML developers

