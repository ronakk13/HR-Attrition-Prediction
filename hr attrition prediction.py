
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

def load_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

def preprocess_data(df):

    drop_cols = [
        'DailyRate', 'EmployeeCount', 'EmployeeNumber',
        'HourlyRate', 'MonthlyRate', 'NumCompaniesWorked',
        'Over18', 'StandardHours', 'StockOptionLevel',
        'TrainingTimesLastYear', 'YearsWithCurrManager'
    ]

    df = df.drop(columns=drop_cols)

    df["SatisfactionScore"] = (
        df["EnvironmentSatisfaction"] +
        df["JobSatisfaction"] +
        df["RelationshipSatisfaction"] +
        df["WorkLifeBalance"]
    ) / 4

    df["YearsPerRole"] = (
        df["YearsInCurrentRole"] /
        (df["YearsAtCompany"] + 1)
    )

    df["PromotionGap"] = (
        df["YearsAtCompany"] -
        df["YearsSinceLastPromotion"]
    )

    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    df = pd.get_dummies(
        df,
        columns=[
            "Department",
            "BusinessTravel",
            "EducationField",
            "JobRole",
            "MaritalStatus"
        ],
        dtype=int
    )

    return df


def train_model(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_train_sm, y_train_sm = smote.fit_resample(
        X_train,
        y_train
    )

    model = LGBMClassifier(random_state=42)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [6, 8, 10, 12],
        "num_leaves": [31, 50, 100]
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="recall",
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_sm, y_train_sm)

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]

    threshold = 0.25

    y_pred = (y_prob >= threshold).astype(int)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("ROC AUC  :", roc_auc_score(y_test, y_prob))


def main():

    df = load_data()

    df = preprocess_data(df)

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = train_model(X_train, y_train)

    evaluate_model(
        model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main() 