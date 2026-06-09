# HR Employee Attrition Prediction

## Project Overview

Employee attrition is a major challenge for organizations because losing experienced employees increases hiring costs, training expenses, and productivity loss.

The objective of this project is to build a machine learning model that predicts whether an employee is likely to leave the organization based on demographic, job-related, and satisfaction-related factors.

---

## Dataset

IBM HR Analytics Employee Attrition Dataset

* Total Records: 1,470
* Target Variable: Attrition

  * 0 = Employee Stays
  * 1 = Employee Leaves

The dataset is naturally imbalanced, with significantly fewer attrition cases than non-attrition cases.

---

## Data Preprocessing

### Removed Unnecessary Columns

The following columns were removed because they contained little or no predictive value:

* EmployeeCount
* EmployeeNumber
* Over18
* StandardHours
* DailyRate
* MonthlyRate
* HourlyRate
* StockOptionLevel
* TrainingTimesLastYear
* YearsWithCurrManager
* NumCompaniesWorked

---

## Feature Engineering

Three additional features were created:

### SatisfactionScore

Average of:

* Environment Satisfaction
* Job Satisfaction
* Relationship Satisfaction
* Work Life Balance

### YearsPerRole

Years in Current Role ÷ Years at Company

### PromotionGap

Years at Company − Years Since Last Promotion

These engineered features were designed to capture employee engagement, role stability, and promotion history.

---

## Handling Categorical Variables

### Binary Encoding

* Attrition
* Gender
* OverTime

### One-Hot Encoding

* Department
* JobRole
* EducationField
* BusinessTravel
* MaritalStatus

One-Hot Encoding was chosen because these variables are nominal categories and do not have any natural ordering.

---

## Class Imbalance Handling

The target variable was highly imbalanced.

Approximate distribution:

* No Attrition: ~84%
* Attrition: ~16%

SMOTE (Synthetic Minority Oversampling Technique) was applied only to the training dataset to generate synthetic minority samples and improve model learning.

---

## Model Selection

LightGBM Classifier was selected because:

* Works exceptionally well on tabular datasets
* Handles feature interactions effectively
* Faster training compared to many ensemble models
* Generally outperforms Decision Trees and Random Forests on structured business data

Hyperparameter tuning was performed using RandomizedSearchCV.

---

## Threshold Optimization

Instead of using the default classification threshold of 0.50, threshold tuning was performed.

The optimal threshold was found to be 0.25, resulting in improved recall and overall F1 score.

---

## Model Performance

### Classification Metrics

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.82  |
| Precision | 0.45  |
| Recall    | 0.53  |
| F1 Score  | 0.49  |
| ROC AUC   | 0.79  |

### Confusion Matrix

|              | Predicted Stay | Predicted Leave |
| ------------ | -------------- | --------------- |
| Actual Stay  | 216            | 31              |
| Actual Leave | 22             | 25              |

---

## Why Is The F1 Score Not Higher?

This is a realistic business dataset with:

* Only 1,470 records
* Limited attrition examples
* Overlapping employee behavior patterns

Many employees who leave and those who stay share similar characteristics, making perfect separation impossible.

The primary goal was not maximizing accuracy but improving the identification of employees at risk of leaving.

For this reason:

* Recall was prioritized
* Class imbalance was addressed using SMOTE
* Threshold tuning was performed

---

## Top Predictive Features

According to LightGBM Feature Importance:

1. MonthlyIncome
2. Age
3. YearsPerRole
4. DistanceFromHome
5. OverTime
6. EnvironmentSatisfaction
7. PromotionGap
8. Marital Status
9. Job Involvement
10. Total Working Years

These features showed the strongest influence on employee attrition predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* Imbalanced-Learn (SMOTE)
* Matplotlib
* Seaborn

---

## Key Learnings

* Handling imbalanced datasets using SMOTE
* Feature Engineering
* One-Hot Encoding
* Hyperparameter Tuning
* Threshold Optimization
* Model Evaluation using Recall, F1 Score, and ROC-AUC
* Feature Importance Analysis
