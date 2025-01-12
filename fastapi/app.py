from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define input data schema
class IrisFeatures(BaseModel):
    features: list

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the ML Model API!"}

@app.post("/predict/")
def predict(input_data: IrisFeatures):
    # Extract features and make prediction
    input_features = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(input_features)
    return {"prediction": int(prediction[0])}
