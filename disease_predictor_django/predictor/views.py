from django.shortcuts import render
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessing tools once
model = joblib.load(os.path.join(BASE_DIR, "models/model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models/vectorizer.pkl"))  # TF-IDF or CountVectorizer
symptom_list = joblib.load(os.path.join(BASE_DIR, "models/symptom_list.pkl"))  # Use full path

def index(request):
    return render(request, "predictor/index.html", {"symptoms": symptom_list})

def predict(request):
    if request.method == 'POST':
        # Get the full symptom description from textarea
        symptom_description = request.POST.get('symptoms')
        predicted_disease, accuracy = predict_disease(symptom_description)

        return render(request, 'predictor/result.html', {
            'predicted_disease': predicted_disease,
            'accuracy': accuracy,
        })
    return render(request, 'predictor/index.html', {"symptoms": symptom_list})

def predict_disease(symptom_description):
    """
    Predict disease based on user symptom description (a full sentence).
    """
    # Step 1: Vectorize the input text (symptom description) using the pre-trained vectorizer
    input_vector = vectorizer.transform([symptom_description])  # shape: (1, 2531)

    # Step 2: Predict using the model
    prediction = model.predict(input_vector)[0]

    # Step 3: Map the predicted class to the actual disease name using LabelEncoder
    predicted_disease = label_encoder.inverse_transform([prediction])[0]

    # Step 4: Get the confidence score (probability) of the predicted class
    confidence_scores = model.predict_proba(input_vector)
    accuracy = round(np.max(confidence_scores) * 100, 2)  # The highest probability is used as confidence

    return predicted_disease, accuracy
