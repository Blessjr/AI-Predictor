from django.shortcuts import render
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessing tools once
model = joblib.load(os.path.join(BASE_DIR, "models/model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))

# Load the exact training symptom list (ordered and consistent)
symptom_list = []
with open(os.path.join(BASE_DIR, 'data/symptoms_list.txt')) as f:
    for line in f:
        parts = line.strip().split(',')[1:]  # skip first column if it's a disease name
        symptom_list.extend(parts)

# Final list of symptoms in training order (no deduplication to preserve count)
# If needed, deduplicate but preserve order — ONLY if your training data had deduplication
seen = set()
ordered_symptom_list = [s for s in symptom_list if not (s in seen or seen.add(s))]


def index(request):
    return render(request, "predictor/index.html", {"symptoms": ordered_symptom_list})


def predict(request):
    if request.method == 'POST':
        user_symptoms = request.POST.getlist('symptoms')  # ✅ safe from form input

        # Build input vector based on training symptom list
        input_vector = [1 if symptom in user_symptoms else 0 for symptom in ordered_symptom_list]

        # Ensure feature count matches training
        if len(input_vector) != scaler.n_features_in_:
            return render(request, 'predictor/error.html', {
                'message': f"Feature count mismatch: expected {scaler.n_features_in_}, got {len(input_vector)}"
            })

        # Scale and predict
        input_vector_scaled = scaler.transform([input_vector])
        prediction = model.predict(input_vector_scaled)

        return render(request, 'predictor/result.html', {'prediction': prediction[0]})
    
    else:
        return render(request, 'predictor/predict.html')
