# Flask app for model deployment
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the ML Model API!"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "This endpoint is for POST requests only. Use POST to make predictions."

    # POST request logic
    data = request.json
    input_features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
