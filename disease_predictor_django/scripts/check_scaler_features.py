import joblib

scaler = joblib.load("models/scaler.pkl")
print("Scaler expects:", scaler.n_features_in_, "features")
