# 🫀 Heart Disease Risk Prediction with Machine Learning

## 🎯 Project Objective

This project aims to develop a machine learning model to predict the **severity of heart disease** (multiclass classification: levels 0 to 4) using clinical features from the [UCI Heart Disease Dataset]([https://archive.ics.uci.edu/dataset/45/heart+disease](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)). https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data


> Rather than simplifying the task into a binary classification (0 = no disease, 1 = disease), we preserve the **five risk levels** to explore the model’s ability to predict nuanced clinical stages — making the solution more aligned with real-world medical decision-making.

---

## 📊 1. Exploratory Data Analysis (EDA)

Initial analysis focused on understanding variable distributions, relationships, and class balance:

- Visualized histograms and boxplots for numerical features.
- Identified categorical features (`cp`, `restecg`, `slope`, etc.).
- Discovered **class imbalance**, especially in class 4 (most severe).
- Checked for **missing data** and **outliers**.

---

## 🧹 2. Data Cleaning & Preprocessing (Wrangling)

Key preprocessing steps included:

- ✅ Removed rows with excessive missing values (**7+ NaNs**).
- ❌ Dropped columns `ca` and `thal` due to **>50% missing data** and low clinical interpretability.
- 🧩 Imputed missing values:
  - `chol`, `trestbps`, `thalach`, `oldpeak`: **Median** (due to skewed distributions).
  - `fbs`, `exang`, `restecg`: **Mode** (categorical).
  - `slope`: Introduced a new category → `'unknown'`.

---

## 🔡 3. Categorical Feature Encoding

- ✅ Applied **One-Hot Encoding** to:
  - `cp`, `restecg`, `slope` (including `'unknown'`)
- ✅ Converted boolean features `fbs` and `exang` to **0/1**.
- ❌ Dropped `id` column (non-predictive identifier).

---

## 📐 4. Feature Scaling (Normalization)

To ensure fair treatment of features in distance-sensitive algorithms, we applied:

- **RobustScaler** on:
  - `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
- 📌 Reason: Robust to **outliers** and effective for **non-Gaussian** distributions.
- ✨ Other scalers like `StandardScaler` were considered but discarded due to lack of robustness in medical datasets.

---

## ⚖️ 5. Class Imbalance Handling

The target variable `num` (heart disease level) showed significant imbalance:

| Class (Risk Level) | Count |
|--------------------|-------|
| 0 (No disease)     | 410   |
| 4 (Severe disease) | 30    |

### Applied Solution:

- Used **SMOTE** (Synthetic Minority Oversampling Technique) **only on the training set**.
- Balanced the dataset to **329 samples per class** in training.
- ⚠️ **Decision:** Kept full multiclass structure (0–4) to analyze model performance across different **risk levels**, rather than simplifying to binary.

---

## 🤖 6. Machine Learning Models Used

The following classification models were trained and evaluated:

| Model | Description |
|-------|-------------|
| ✅ Logistic Regression (OvR) | Baseline for multiclass classification |
| ✅ Random Forest | Robust, interpretable ensemble method |
| ✅ Support Vector Machine | Effective for nonlinear boundaries |
| ✅ K-Nearest Neighbors | Simple, distance-based baseline |
| ✅ Gradient Boosting | Boosted decision trees for performance |
| ✅ XGBoost | Highly optimized gradient boosting |

---

## 📈 7. Evaluation Metrics

Model performance was evaluated using:

- ✅ **Accuracy**
- ✅ **Classification Report** (Precision, Recall, F1-score)
- ✅ **Confusion Matrix**

> Models performed best on classes 0 and 1. Predicting classes 3 and 4 remained challenging due to fewer samples and clinical overlap — a realistic reflection of diagnostic complexity.

---

## 📌 8. Key Insights

- **Random Forest** and **XGBoost** were the top performers.
- One-hot encoding and robust scaling contributed significantly to stability.
- Outliers were **retained** due to clinical importance (e.g., extremely high blood pressure).
- Multiclass setting provides richer insights than binary classification.

---

## 🚀 9. Future Improvements

Suggestions for future enhancements:

- 🧠 **Feature Engineering**:
  - Create derived features like age groups, risk scores, etc.
- 🔧 **Hyperparameter Tuning**:
  - Use `GridSearchCV` or `Optuna`.
- 📉 **Feature Selection**:
  - Remove redundant or highly correlated features.
- 🩺 **Model Explainability**:
  - Use `SHAP` or `LIME` to understand model predictions.

---

## 📁 Project Structure

```
📦 uci-heart-disease-ml/
├── Uci-1.ipynb            # Main Jupyter Notebook
├── Uci-1.pdf              # PDF report
├── README.md              # Project summary (this file)
└── data/
    └── heart_disease_uci.csv          # Original UCI dataset
```

---

## ✅ Conclusion

This end-to-end machine learning pipeline demonstrates how to tackle a **real-world multiclass classification problem in healthcare** — from raw data and preprocessing to model evaluation and strategic improvements.

The project showcases the **clinical potential** of AI/ML to assess varying **levels of cardiovascular risk**, rather than simply detecting disease presence.

> 🔍 Interested in contributing? Fork the repo, experiment with other models, or test this framework on new medical datasets.
