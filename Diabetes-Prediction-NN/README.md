![](UTA-DataScience-Logo.png)

# Diabetes Prediction with Neural Networks

* **One Sentence Summary**  
This repository contains a machine learning pipeline that uses a feedforward neural network and SMOTE to predict whether a patient is diabetic, based on a clinical dataset from the Kaggle challenge: [Diabetes Prediction with NN](https://www.kaggle.com/competitions/diabetes-prediction-with-nn).

---

## Overview

The task is to predict diabetes status (1 = diabetic, 0 = non-diabetic) based on structured clinical data (e.g., glucose, HbA1c, BMI).  
We approached this as a binary classification task using a feedforward neural network, and addressed severe class imbalance with SMOTE.  
Our best model achieved:
- **Accuracy:** 97.2%
- **AUC:** 0.9757
- **PR AUC:** 0.8816

We also compared classical ML models (Logistic Regression, Random Forest) against the NN.

---

## Summary of Work Done

### Data

- **Input**: CSV of patient-level medical features
- **Target**: `diabetes` column (binary label)
- **Size**: ~25,000 samples
- **Split**:
  - 80% training (with SMOTE applied)
  - 20% validation (untouched)

---

#### Preprocessing / Cleanup

- Dropped unnecessary columns
- One-hot encoding of categorical variables
- Scaled numeric features using StandardScaler
- Applied SMOTE to oversample the minority class (diabetic patients)

---

#### Data Visualization

- Histograms for key features by diabetes status
- Class imbalance confirmed (~8.5% diabetic)
- Key predictors: HbA1c level, blood glucose level

---

### Problem Formulation

- **Input**: Standardized tabular data (numeric + encoded)
- **Output**: Binary prediction (diabetic or not)
- **Model**: Feedforward neural network
  - 64 → 32 → 1 architecture
  - ReLU activations, dropout = 0.3
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, AUC, Precision, Recall, F1

---

### Training

- **Platform**: Jupyter Notebook (local)
- **Packages**: TensorFlow, scikit-learn, imbalanced-learn, pandas, seaborn
- **Runtime**: ~1 min per model (CPU)
- **Training Duration**: Fixed 20 epochs
- **Stopping Rule**: Manual monitoring of overfitting via validation accuracy
- **Difficulties**: Class imbalance skewed precision and recall — resolved with SMOTE

---

### Performance Comparison

| Model                        | Accuracy | AUC    | Precision | Recall | F1 Score |
|-----------------------------|----------|--------|-----------|--------|----------|
| Logistic Regression         | 0.9605   | 0.9625 | 0.99      | 0.68   | 0.81     |
| Random Forest               | 0.9740   | 0.9613 | 0.92      | 0.76   | 0.83     |
| Neural Network (After SMOTE)| 0.9722   | 0.9757 | 0.47      | 0.90   | 0.62     |

- ROC and PR curves were plotted
- Confusion matrices used to visualize trade-offs

---

### Conclusions

- Neural network after SMOTE had highest recall — best for identifying diabetic patients
- Logistic regression had higher precision — good for avoiding false alarms
- SMOTE increased recall but decreased precision
- Final model choice depends on application:
  - **High recall** → Use SMOTE model
  - **High precision** → Use original model

---

### Future Work

- Train NN on imbalanced data to directly compare both versions
- Try XGBoost or weighted loss functions
- Add SHAP or LIME for feature interpretability
- Optimize thresholds for different clinical contexts

---

## How to Reproduce Results

1. Clone the repo
2. Download the dataset from Kaggle and place in `/data/`
3. Run `Project_Final.ipynb` from top to bottom
4. (Optional) Submit `submission.csv` to Kaggle to evaluate

---

### Overview of Files in Repository

