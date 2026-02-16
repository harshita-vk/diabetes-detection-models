# diabetes-detection-models
Machine learning project to detect diabetes using Logistic Regression and Random Forest
# Diabetes Detection using Machine Learning

## Overview

This project implements a machine learning pipeline to detect diabetes in patients using medical diagnostic features. The system uses Logistic Regression and Random Forest classifiers and evaluates their performance using confusion matrix, accuracy, ROC curve, and AUC score.

The project demonstrates the complete workflow including preprocessing, manual feature standardization, model training, evaluation, and model comparison.

---

## Dataset

The dataset used is the **Pima Indians Diabetes Dataset**, which contains medical records of patients.

Features:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI (Body Mass Index)
* Diabetes Pedigree Function
* Age

Target:

* 0 → No Diabetes
* 1 → Diabetes

Dataset size:

* 768 samples
* 8 features

---

## Project Pipeline

### 1. Data Loading and Exploration

* Loaded dataset using pandas
* Checked dataset shape and structure
* Verified no duplicate records

### 2. Data Preprocessing

* Split dataset into training and testing sets
* Implemented feature standardization manually using training mean and standard deviation
* Prevented data leakage by using training statistics for test data scaling

### 3. Logistic Regression Model

* Trained Logistic Regression classifier on standardized data
* Generated predictions on test data

### 4. Confusion Matrix (Implemented from Scratch)

Manually computed:

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)

Constructed confusion matrix without using sklearn utilities.

### 5. Accuracy Calculation (From Scratch)

Calculated accuracy manually using:

Accuracy = (TP + TN) / Total Samples

### 6. ROC Curve and AUC Score

* Computed probability predictions using predict_proba
* Plotted ROC curve using sklearn
* Calculated AUC score to measure model performance

### 7. Random Forest Implementation

* Trained Random Forest classifier
* Generated probability predictions
* Computed ROC curve and AUC score

### 8. Model Comparison

Compared Logistic Regression and Random Forest using:

* ROC curve
* AUC score
* Accuracy

---

## Results

Both models were evaluated using ROC curve and AUC score.

ROC curve visualization allows comparison of classification performance across multiple thresholds.

The model with higher AUC score demonstrates better classification performance.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Google Colab

---

## Key Concepts Demonstrated

* Machine Learning Classification
* Logistic Regression
* Random Forest
* Feature Standardization (manual implementation)
* Confusion Matrix (manual implementation)
* Accuracy calculation (manual implementation)
* ROC Curve and AUC Score
* Model Performance Evaluation

---

## Project Structure

```
diabetes-detection-ml/
│
├── diabetes_detection.ipynb
├── diabetes.csv
├── README.md
```

---

## How to Run

1. Open the notebook in Google Colab or Jupyter Notebook
2. Run all cells sequentially
3. View evaluation metrics and ROC curve comparison

---

## Conclusion

This project demonstrates a complete machine learning pipeline for diabetes detection, including manual preprocessing, model training, performance evaluation, and comparison using ROC and AUC analysis.

---

## Author

Harshita V K
