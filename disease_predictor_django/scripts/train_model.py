import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "symptom2disease.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SYMPTOM_LIST_PATH = os.path.join(BASE_DIR, "data", "symptom_list.txt")

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Drop target column and free-text column
    X = df.drop(["Disease", "description"], axis=1, errors='ignore')
    y = df["Disease"]

    # Encode non-numeric columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Return the features, target, and necessary transformers
    return X_scaled, y, label_encoder, scaler, X.columns.tolist()

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model

def save_artifacts(model, scaler, label_encoder, feature_names):
    # Create necessary directories
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model, scaler, and label_encoder
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # Save symptom list (feature names)
    os.makedirs(os.path.dirname(SYMPTOM_LIST_PATH), exist_ok=True)
    with open(SYMPTOM_LIST_PATH, "w") as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    print("Artifacts saved successfully.")
    print(f"Symptom list saved to {SYMPTOM_LIST_PATH}")

if __name__ == "__main__":
    df = load_data(DATASET_PATH)
    X, y, label_encoder, scaler, feature_names = preprocess(df)
    model = train_model(X, y)
    save_artifacts(model, scaler, label_encoder, feature_names)
