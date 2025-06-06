import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Define the correct base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Create the model directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Load dataset from the correct data directory
dataset_path = os.path.join(DATA_DIR, 'symptom2disease.csv')
if not os.path.exists(dataset_path):
    print(f"Error: The file '{dataset_path}' was not found.")
    exit()

# Load the CSV file
df = pd.read_csv(dataset_path)

# Initialize the TfidfVectorizer for symptom descriptions
vectorizer = TfidfVectorizer(stop_words='english')

# Apply vectorizer on the symptom descriptions to get features
symptom_descriptions = df['Symptoms'].tolist()
X = vectorizer.fit_transform(symptom_descriptions)  # This will create the term-document matrix
X = X.toarray()  # Convert sparse matrix to array for training

y = df['Disease']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature scaling (apply scaler only if required, here it's not strictly necessary for TF-IDF)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dump the vectorizer for future use
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))  # Save the vectorizer
print("Vectorizer has been saved successfully.")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and tools
joblib.dump(model, os.path.join(MODEL_DIR, 'model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, 'label_encoder.pkl'))

print("Model, scaler, and label encoder have been saved successfully.")

# Generate symptoms_list.txt with disease and symptoms
symptoms_list_path = os.path.join(DATA_DIR, 'symptoms_list.txt')
with open(symptoms_list_path, 'w') as f:
    for _, row in df.iterrows():
        f.write(f"{row['Disease']},{row['Symptoms']}\n")

print(f"symptoms_list.txt has been saved to: {symptoms_list_path}")
