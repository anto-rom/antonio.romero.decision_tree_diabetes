from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Ruta robusta al modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_arbol_diabetes.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        print(">>> He recibido un POST en /")
        print(">>> request.form:", request.form.to_dict())

        try:
            # Mapear val1..val8 al orden de variables del modelo
            preg = float(request.form["val1"])  # Pregnancies
            glu  = float(request.form["val2"])  # Glucose
            bp   = float(request.form["val3"])  # BloodPressure
            skin = float(request.form["val4"])  # SkinThickness
            ins  = float(request.form["val5"])  # Insulin
            bmi  = float(request.form["val6"])  # BMI
            dpf  = float(request.form["val7"])  # DiabetesPedigreeFunction
            age  = float(request.form["val8"])  # Age

            X = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
            print(">>> Features recibidas:", X)

            pred = model.predict(X)[0]
            resultado = "Diabetes" if pred == 1 else "No diabetes"

            return render_template("index.html", prediction=resultado)
        except Exception as e:
            print(">>> ERROR en predicción:", e)
            return render_template("index.html", error=str(e))

    return render_template("index.html")



