# ============================================
# train.py
# ============================================

# Import necessary libraries
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# For saving confusion matrix as image
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ---------------------------------------------------
# Function: load_data
# Purpose:  Load final transformed dataset
# ---------------------------------------------------
def load_data(path):
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    print("Dataset loaded for training.")
    print("Shape:", df.shape)
    return df


# ---------------------------------------------------
# Function: split_data
# Purpose:  Split into train and test sets
# ---------------------------------------------------
def split_data(df):
    """
    Split dataset into train and test sets.
    Using stratify to maintain class distribution.
    """

    # Separate features and target
    X = df.drop("price_category", axis=1)
    y = df["price_category"]

    # Convert target to integer type
    y = y.astype(int)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Data split completed.")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------
# Function: evaluate_model
# Purpose:  Evaluate and log metrics
# ---------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate trained model and log metrics in MLflow.
    """

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Log accuracy in MLflow
    mlflow.log_metric("accuracy", accuracy)

    print(f"{model_name} Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_test, y_pred)
    print(report)

    # Log confusion matrix as image
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {model_name}")

    # Save confusion matrix image
    cm_path = f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path)

    # Log image to MLflow
    mlflow.log_artifact(cm_path)

    # Close plot to free memory
    plt.close()

    return accuracy


# ---------------------------------------------------
# Function: train_and_log_model
# Purpose:  Train model and log into MLflow
# ---------------------------------------------------
def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train given model and log everything in MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        # Train model
        model.fit(X_train, y_train)

        # Log model parameters
        mlflow.log_param("model_name", model_name)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test, model_name)

        # Log trained model
        mlflow.sklearn.log_model(model, "model")

    return accuracy


# ---------------------------------------------------
# Main Execution Block
# ---------------------------------------------------
if __name__ == "__main__":

    # Use a workspace-local tracking directory by default.
    # This avoids stale absolute paths from committed MLflow metadata.
    tracking_dir = Path(os.getenv("MLFLOW_TRACKING_DIR", ".mlruns")).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    # Set experiment name
    mlflow.set_experiment("House_Price_Classification")

    # Load dataset
    df = load_data("data/processed/Housing_final.csv")

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Dictionary to store model performance
    model_scores = {}

    # ---------------------------
    # Model 1: Logistic Regression
    # ---------------------------
    lr_model = LogisticRegression(max_iter=1000)
    lr_accuracy = train_and_log_model(
        lr_model,
        "Logistic_Regression",
        X_train,
        X_test,
        y_train,
        y_test
    )
    model_scores["Logistic_Regression"] = lr_accuracy

    # ---------------------------
    # Model 2: Random Forest
    # ---------------------------
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_accuracy = train_and_log_model(
        rf_model,
        "Random_Forest",
        X_train,
        X_test,
        y_train,
        y_test
    )
    model_scores["Random_Forest"] = rf_accuracy

    # ---------------------------------
    # Select Best Model Based on Accuracy
    # ---------------------------------
    best_model_name = max(model_scores, key=model_scores.get)

    print("Best Model:", best_model_name)

    # Retrain best model on full training data
    if best_model_name == "Logistic_Regression":
        best_model = lr_model
    else:
        best_model = rf_model

    # Save best model locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, f"models/{best_model_name}.pkl")

    print(f"Best model saved in models/{best_model_name}.pkl")
