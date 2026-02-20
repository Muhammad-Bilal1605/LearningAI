import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from PIL import Image
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder

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


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/Pnemonia", methods=["GET", "POST"])
def predictPneumonia():
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
            "probability": round(prob, 4),
            "class": label
        }

    return render_template("diabetes.html", prediction=prediction)


MODEL_PATH = "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_6/task_1/medical_cost_xg.json"
SCALER_PATH = "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_6/task_1/scaler.pkl"
OHE_PATH = "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_6/task_1/ohe_encoder.pkl"
LABEL_ENCODER_PATH = "/Users/Bilal/PycharmProjects/LearningAI/models/assignment_6/task_1/label_encoder.pkl"

# Load model
model_medical_cost = xgb.Booster()
model_medical_cost.load_model(MODEL_PATH)

# Load preprocessors
scaler_medical_cost = joblib.load(SCALER_PATH)
ohe_medical_cost = joblib.load(OHE_PATH)
label_encoder_medical_cost = joblib.load(LABEL_ENCODER_PATH)


@app.route("/MedicalCost", methods=["GET", "POST"])
def predictMedicalCost():
    prediction = None

    if request.method == "POST":
        # Get form data
        age = float(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = float(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        # Convert to DataFrame
        input_df = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        # Encode categorical

        label_encoder = LabelEncoder()
        input_df["sex"] = label_encoder.fit_transform(input_df["sex"])
        input_df["smoker"] = label_encoder.fit_transform(input_df["smoker"])

        region_encoded = ohe_medical_cost.transform(input_df[["region"]])
        input_df = pd.concat([input_df.drop(columns=["region"]), region_encoded], axis=1)

        # Scale bmi and children
        input_df[["bmi", "children"]] = scaler_medical_cost.transform(
            input_df[["bmi", "children"]]
        )

        # Convert to DMatrix
        dmatrix = xgb.DMatrix(input_df)

        # Predict
        prediction_value = model_medical_cost.predict(dmatrix)[0]

        prediction = round(float(prediction_value), 2)

    return render_template("medical_cost.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
