from django.shortcuts import render
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessing tools
model = joblib.load(os.path.join(BASE_DIR, "models/model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))

# Load symptom list
with open(os.path.join(BASE_DIR, "data/symptoms_list.txt")) as f:
    symptom_list = [line.strip() for line in f]

def index(request):
    return render(request, "predictor/index.html", {"symptoms": symptom_list})

def predict(request):
    if request.method == 'POST':
        symptoms_str = request.POST.get('symptoms', '')
        symptoms = [s.strip().lower() for s in symptoms_str.split(',') if s.strip()]

        # Prepare input vector for prediction (this should match your model's input)
        input_vector = [1 if symptom in symptoms else 0 for symptom in symptom_list]

        # Apply scaling to the input vector if scaling was used during model training
        input_vector_scaled = scaler.transform([input_vector])  # Apply the scaler if needed

        # Predict the disease (model's numeric label)
        predicted_label = model.predict(input_vector_scaled)[0]

        # Get the disease name using the label_encoder (decode the numeric label)
        disease_name = label_encoder.inverse_transform([predicted_label])[0]

        # Get the confidence percentage (highest probability)
        probabilities = model.predict_proba(input_vector_scaled)[0]
        confidence = round(max(probabilities) * 100, 2)

        # Return the prediction and confidence to the template
        return render(request, "predictor/index.html", {
            "prediction": disease_name,
            "confidence": confidence,
            "symptoms": symptoms_str
        })
