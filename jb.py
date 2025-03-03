
import joblib
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Test input
input_data = pd.DataFrame([{
    'budget': 100000000,
    'popularity': 150,
    'runtime': 120,
    'primary_genre': 'Action',
    'director_popularity': 8
}])

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print(f"Prediction: {'Success' if prediction else 'Failure'}")
print(f"Confidence: {probability * 100:.2f}%")