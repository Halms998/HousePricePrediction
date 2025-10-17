from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load model and preprocessor
try:
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('data/preprocessor.pkl')
    print("Model and preprocessor loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    preprocessor = None

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 500
    
    try:
        # Get data from request
        data = request.get_json()
        
        # Create DataFrame from input data with correct column mapping
        input_df = pd.DataFrame([{
            'baths': data.get('bathrooms', 0),  # Map 'bathrooms' form field to 'baths' column
            'bedrooms': data.get('bedrooms', 0),
            'latitude': data.get('latitude', 0),
            'longitude': data.get('longitude', 0),
            'Area Size': data.get('area', 0),  # Map 'area' form field to 'Area Size' column
            'property_type': data.get('property_type', 'House'),
            'city': data.get('city', 'Karachi'),
            'province_name': data.get('province', 'Sindh'),  # Map 'province' form field to 'province_name'
            'Area Category': data.get('area_category', '5-10 Marla')  # Map 'area_category' form field
        }])
        
        # Preprocess the input
        processed_input = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        return jsonify({
            'predicted_price': float(prediction[0]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400
    
    

@app.route('/health')
def health():
    if model is not None and preprocessor is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)