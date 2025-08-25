#!/usr/bin/env python3
"""
Minimal training script for fraudulent-transactions-detection-ML.

Usage:
  python src/train.py --csv path/to/Fraud.csv

It will:
  - Load the CSV
  - Basic preprocessing (encode "type", drop highly collinear features)
  - Train/test split (80/20)
  - Train a RandomForestClassifier
  - Print accuracy/precision/recall/F1/ROC-AUC
  - Save the model to models/random_forest.joblib
"""

import argparse
import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    return df

def preprocess(df: pd.DataFrame):
    # Encode 'type' if present
    if 'type' in df.columns:
        le = LabelEncoder()
        df['type'] = le.fit_transform(df['type'])

    # Drop highly collinear features if present
    for col in ['newbalanceDest', 'newbalanceOrig']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Target & features
    if 'isFraud' not in df.columns:
        print("[ERROR] Target column 'isFraud' not found.")
        sys.exit(1)

    X = df.drop(columns=['isFraud'], errors='ignore')
    y = df['isFraud'].astype(int)

    # Optional: drop high-cardinality IDs (comment out to keep them)
    for id_col in ['nameOrig', 'nameDest']:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, scaler

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = getattr(clf, "predict_proba", None)
    roc = None
    if y_prob is not None:
        roc = roc_auc_score(y_test, y_prob(X_test)[:,1])

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }
    return clf, metrics

def save_model(model, out_path="models/random_forest.joblib"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Fraud.csv")
    args = parser.parse_args()

    df = load_data(args.csv)
    X, y, scaler = preprocess(df)
    model, metrics = train_and_evaluate(X, y)
    model_path = save_model(model)

    print("\n=== Evaluation ===")
    for k, v in metrics.items():
        if k == "report":
            continue
        print(f"{k}: {v}")
    print("\nClassification report:\n", metrics["report"])
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()
