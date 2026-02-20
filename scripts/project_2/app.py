import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_7/task_1/diabetes_model.keras")

# Load scaler
scaler = joblib.load("/Users/Bilal/PycharmProjects/LearningAI/models/assignment_7/task_1/scaler.pkl")

pneumonia_model = tf.keras.models.load_model(
    "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_7/task_2/pneumonia_model.keras")

IMG_SIZE = 224


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)

            pred = pneumonia_model.predict(processed_image)
            prob = float(pred)
            print(pred)
            label = "Pneumonia" if prob >= 0.5 else "Normal"

            prediction = {
                "probability": round(prob, 4),
                "class": label,
                "whole array": pred

            }

    return render_template("pneumonia.html", prediction=prediction)


@app.route("/diabetes", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        input_data = np.array([[
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"])
        ]])

        scaled_input = scaler.transform(input_data)
        pred = model.predict(scaled_input)
        prob = float(pred)

        label = "Diabetes" if prob >= 0.6 else "Not Diabetes"

        prediction = {
            "probability of being pneumonia": round(prob, 4),
            "class": label
        }

    return render_template("diabetes.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
