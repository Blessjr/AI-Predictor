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
with open(os.path.join(BASE_DIR, "data/symptom_list.txt")) as f:
    all_symptoms = [line.strip() for line in f]

def index(request):
    return render(request, "predictor/index.html", {"symptoms": all_symptoms})

def predict(request):
    if request.method == 'POST':
        symptoms_str = request.POST.get('symptoms', '')
        symptoms = [s.strip().lower() for s in symptoms_str.split(',') if s.strip()]
        
        print("Selected symptoms:", symptoms)

        input_vector = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
        prediction = model.predict([input_vector])[0]
        probabilities = model.predict_proba([input_vector])[0]
        confidence = round(max(probabilities) * 100, 2)

        return render(request, "predictor/index.html", {
            "prediction": prediction,
            "confidence": confidence,
        })

