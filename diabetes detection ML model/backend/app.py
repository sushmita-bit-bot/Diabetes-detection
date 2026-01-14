from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from model import train_models, predict_diabetes

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to save trained models
MODEL_PATH = 'models'
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# Initialize function to check and train models if needed
def initialize():
    # Check if models exist, if not train them
    if not (os.path.exists(os.path.join(MODEL_PATH, 'logistic_regression.pkl')) and 
            os.path.exists(os.path.join(MODEL_PATH, 'random_forest.pkl')) and 
            os.path.exists(os.path.join(MODEL_PATH, 'svm.pkl'))):
        print("Training models...")
        train_models()
        print("Models trained and saved.")

# Call initialize at startup
with app.app_context():
    initialize()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features from request
        features = {
            'Pregnancies': float(data['pregnancies']),
            'Glucose': float(data['glucose']),
            'BloodPressure': float(data['bloodPressure']),
            'SkinThickness': float(data['skinThickness']),
            'Insulin': float(data['insulin']),
            'BMI': float(data['bmi']),
            'DiabetesPedigreeFunction': float(data['diabetesPedigree']),
            'Age': float(data['age'])
        }
        
        # Convert to DataFrame for prediction
        input_df = pd.DataFrame([features])
        
        # Get predictions from all models
        model_name = data.get('model', 'ensemble')  # Default to ensemble
        result = predict_diabetes(input_df, model_name)
        
        return jsonify({
            'prediction': int(result['prediction']),
            'probability': float(result['probability']),
            'model': result['model']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)