# src/app.py
from flask import Flask, request, render_template
from pickle import load
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "src", "modelo_arbol_diabetes.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    model = load(f)

# Cargar dataset para el scaler
df = pd.read_csv(DATA_PATH)

class_dict = {
    "0": "Sin diabetes",
    "1": "Diabetes",
}

num_variables = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

scaler = StandardScaler()
scaler.fit(df[num_variables])


@app.route("/", methods=["GET", "POST"])
def index():
    pred_class = None

    if request.method == "POST":
        Pregnancies = float(request.form["val1"])
        Glucose = float(request.form["val2"])
        BloodPressure = float(request.form["val3"])
        BMI = float(request.form["val4"])
        DiabetesPedigreeFunction = float(request.form["val5"])
        Age = float(request.form["val6"])

        data = np.array([[Pregnancies, Glucose, BloodPressure,
                          BMI, DiabetesPedigreeFunction, Age]])

        data_normalized = scaler.transform(data)

        prediction = str(model.predict(data_normalized)[0])
        pred_class = class_dict[prediction]

    return render_template("index.html", prediction=pred_class)


@app.route("/health")
def health():
    return "OK"
    
