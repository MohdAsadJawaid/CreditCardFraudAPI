from flask import Flask, request, jsonify
import pickle 
from flask_cors import CORS  # Import CORS
import numpy as np


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

with open("creditcard_fraud_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if "features" not in data:
        return jsonify({"error": "Invalid input format"}), 400
    
    features = data["features"]
    
    if len(features) != 29:  # Ensure correct feature count
        return jsonify({"error": "Expected 29 features"}), 400
    
    try:
       
        input_data = np.array(features).reshape(1, -1)
        
       
        prediction = model.predict(input_data)[0]  # Get single prediction
        
       
        return jsonify({"fraud": int(prediction)})  # Convert NumPy int to Python int
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors

if __name__ == '__main__':
    app.run(debug=True)
