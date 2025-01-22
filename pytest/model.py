import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Train and save the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

def predict(features):
    """Takes a list of features and returns a prediction"""
    features = np.array(features).reshape(1, -1)
    return model.predict(features)
