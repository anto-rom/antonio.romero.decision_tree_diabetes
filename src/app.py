# src/app.py
from flask import Flask, request, render_template
from pickle import load
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

# Directorio donde está este archivo (src/)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta al modelo (está en src/)
MODEL_PATH = os.path.join(CURRENT_DIR, "modelo_arbol_diabetes.pkl")

# URL del dataset original
DATA_URL = "https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv"

app = Flask(
    __name__,
    template_folder=os.path.join(CURRENT_DIR, "templates")  # src/templates
)

# Cargar modelo
with open(MODEL_PATH, "rb") as f:
    model = load(f)

# Cargar dataset desde la URL para ajustar el scaler
df = pd.read_csv(DATA_URL)

class_dict = {
    "0": "Sin diabetes",
    "1": "Diabetes",
}

num_variables = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
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
        SkinThickness = float(request.form["val4"])
        Insulin = float(request.form["val5"])
        BMI = float(request.form["val6"])
        DiabetesPedigreeFunction = float(request.form["val7"])
        Age = float(request.form["val8"])

        # Construimos un DataFrame con los mismos nombres de columnas
        row = {
            "Pregnancies": Pregnancies,
            "Glucose": Glucose,
            "BloodPressure": BloodPressure,
            "SkinThickness": SkinThickness,
            "Insulin": Insulin,
            "BMI": BMI,
            "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
            "Age": Age,
        }

        input_df = pd.DataFrame([row])

        # Mantenemos el orden de columnas usado en el scaler
        X_scaled = scaler.transform(input_df[num_variables])

        y_pred = model.predict(X_scaled)[0]
        print("Predicción cruda:", y_pred, flush=True)   # Para ver en logs

        pred_class = class_dict[str(int(y_pred))]

    return render_template("index.html", prediction=pred_class)
