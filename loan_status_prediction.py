"""
Loan Status Prediction using Machine Learning
================================================
Predicts whether a loan application will be Approved (Y) or Rejected (N)
based on applicant details such as income, credit history, education,
property area, etc.

Dataset columns expected (standard "Loan Prediction" schema):
Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status

Usage:
    python loan_status_prediction.py --data loan_data.csv

If you have your own dataset (e.g. from Kaggle's "Loan Prediction Problem
Dataset"), just point --data at that CSV — the column names above must match.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, out_prefix: str = "eda"):
    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Target distribution ---")
    print(df["Loan_Status"].value_counts())

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    sns.countplot(x="Loan_Status", data=df, ax=axes[0, 0], palette="Set2")
    axes[0, 0].set_title("Loan Status Distribution")

    sns.countplot(x="Credit_History", hue="Loan_Status", data=df, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("Credit History vs Loan Status")

    sns.countplot(x="Property_Area", hue="Loan_Status", data=df, ax=axes[0, 2], palette="Set2")
    axes[0, 2].set_title("Property Area vs Loan Status")

    sns.countplot(x="Education", hue="Loan_Status", data=df, ax=axes[1, 0], palette="Set2")
    axes[1, 0].set_title("Education vs Loan Status")

    sns.histplot(df["ApplicantIncome"], kde=True, ax=axes[1, 1], color="teal")
    axes[1, 1].set_title("Applicant Income Distribution")

    sns.boxplot(x="Loan_Status", y="LoanAmount", data=df, ax=axes[1, 2], palette="Set2")
    axes[1, 2].set_title("Loan Amount by Status")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_overview.png", dpi=150)
    plt.close()
    print(f"Saved EDA plots -> {out_prefix}_overview.png")


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    df = df.copy()
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    # Fill missing values
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
        df[col] = df[col].fillna(df[col].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])

    # Clean up Dependents ("3+" -> 3)
    df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

    # Feature engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LoanAmount_log"] = np.log1p(df["LoanAmount"])
    df["TotalIncome_log"] = np.log1p(df["TotalIncome"])
    df["Income_to_Loan"] = df["TotalIncome"] / (df["LoanAmount"] + 1)

    # Encode categoricals
    label_cols = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_le = LabelEncoder()
    df["Loan_Status"] = target_le.fit_transform(df["Loan_Status"])  # N=0, Y=1
    encoders["Loan_Status"] = target_le

    feature_cols = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area", "TotalIncome_log", "LoanAmount_log",
        "Income_to_Loan",
    ]
    X = df[feature_cols]
    y = df["Loan_Status"]
    return X, y, encoders, feature_cols


# ---------------------------------------------------------------------------
# 4. Train & compare models
# ---------------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=9),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, scale_cols=None):
    scaler = StandardScaler()
    X_train_s = X_train.copy()
    X_test_s = X_test.copy()
    X_train_s[X_train.columns] = scaler.fit_transform(X_train)
    X_test_s[X_test.columns] = scaler.transform(X_test)

    results = []
    fitted_models = {}

    for name, model in get_models().items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else preds

        cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy")

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, proba),
            "CV Accuracy (mean)": cv_scores.mean(),
            "CV Accuracy (std)": cv_scores.std(),
        })
        fitted_models[name] = model
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    return results_df, fitted_models, scaler


def plot_results(results_df, y_test, fitted_models, X_test_scaled, out_prefix="eda"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", ax=axes[0], colormap="viridis"
    )
    axes[0].set_title("Model Comparison")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="lower right")
    axes[0].tick_params(axis="x", rotation=30)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[1].plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, proba):.3f})")
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve (best model)")
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved model comparison plot -> {out_prefix}_model_comparison.png")

    cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix -> {out_prefix}_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Loan Status Prediction")
    parser.add_argument("--data", type=str, default="loan_data.csv", help="Path to loan dataset CSV")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--tune", action="store_true", help="Run GridSearchCV to tune the best model")
    args = parser.parse_args()

    df = load_data(args.data)
    run_eda(df)

    X, y, encoders, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    results_df, fitted_models, scaler = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\n=== Model comparison (sorted by F1) ===")
    print(results_df.to_string(index=False))

    X_test_scaled = X_test.copy()
    X_test_scaled[X_test.columns] = scaler.transform(X_test)
    plot_results(results_df, y_test, fitted_models, X_test_scaled)

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    if args.tune and best_name == "Random Forest":
        print("\nTuning Random Forest with GridSearchCV...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [4, 6, 8, None],
            "min_samples_split": [2, 5, 10],
        }
        X_train_scaled = X_train.copy()
        X_train_scaled[X_train.columns] = scaler.transform(X_train)
        grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid, cv=5, scoring="f1")
        grid.fit(X_train_scaled, y_train)
        print("Best params:", grid.best_params_)
        best_model = grid.best_estimator_

    # Feature importance (tree-based models)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        plt.title(f"Feature Importance — {best_name}")
        plt.tight_layout()
        plt.savefig("eda_feature_importance.png", dpi=150)
        plt.close()
        print("Saved feature importance plot -> eda_feature_importance.png")
        print("\nTop features:\n", importances.head(8))

    # Save model + scaler + encoders for later inference
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_cols": feature_cols,
    }, "loan_status_model.pkl")
    print("\nSaved trained pipeline -> loan_status_model.pkl")

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("Saved metrics table -> model_comparison_results.csv")


if __name__ == "__main__":
    main()
