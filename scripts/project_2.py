import joblib

import numpy as np
import tensorflow as tf
from flask import Flask

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_7/task_1/diabetes_model.keras")

# Load scaler
scaler = joblib.load("/Users/Bilal/PycharmProjects/LearningAI/models/assignment_7/task_1/scaler.pkl")


@app.route("/")
def predict():
    input_data = np.array([[6, 148, 72, 35, 0, 33.6, 0.625, 50]])

    # Apply scaling
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    print("Scaled Input:", scaled_input)
    print("Prediction:", prediction)

    return "Prediction printed in terminal!"


if __name__ == "__main__":
    app.run(debug=True, port=5001)
