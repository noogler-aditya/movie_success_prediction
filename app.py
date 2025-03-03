
from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('movie_success_model.pkl')

@app.route('/')
def home():
    # Render the form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Validate form data
        if not all(key in data for key in ['budget', 'genre', 'director_popularity', 'cast_popularity']):
            raise ValueError("Missing form data")
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'budget': float(data['budget']),  # Budget in ₹ crore
            'primary_genre': data['genre'],  # Primary genre
            'director_popularity': float(data['director_popularity']),  # Director popularity
            'cast_popularity': float(data['cast_popularity'])  # Cast popularity
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]  # Predict success (1) or failure (0)
        probability = model.predict_proba(input_data)[0][1]  # Probability of success
        
        # Prepare result
        result = "Success" if prediction == 1 else "Failure"
        confidence = round(probability * 100, 2)  # Confidence percentage
        
        # Render result page
        return render_template('result.html', 
                             result=result,
                             confidence=confidence,
                             budget=data['budget'])
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)