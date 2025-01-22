import pickle
import numpy as np
import pytest
from model import predict  # Import the function from model.py

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def test_model_prediction(capsys):
    sample_input = [5.1, 3.5, 1.4, 0.2]
    prediction = predict(sample_input)
    
    print(prediction)  # Captured by capsys
    print("Prediction Output:", prediction)
    print("Type of Prediction:", type(prediction))


    captured = capsys.readouterr()  # Get captured output
    assert "Prediction" not in captured.err  # Check no errors
    assert isinstance(prediction[0], int)
    assert prediction[0] in [0, 1, 2]
    
    print("Captured output:", captured.out)  # This will print in pytest

