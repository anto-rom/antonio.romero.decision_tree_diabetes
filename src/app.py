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
        print(">>> He recibido un POST en /")  # Para ver en logs de Render
        try:
            # Ejemplo: recoger campos del formulario
            preg = float(request.form["Pregnancies"])
            glu = float(request.form["Glucose"])
            bp = float(request.form["BloodPressure"])
            skin = float(request.form["SkinThickness"])
            ins = float(request.form["Insulin"])
            bmi = float(request.form["BMI"])
            dpf = float(request.form["DiabetesPedigreeFunction"])
            age = float(request.form["Age"])

            X = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
            print(">>> Features recibidas:", X)

            pred = model.predict(X)[0]
            resultado = "Diabetes" if pred == 1 else "No diabetes"

            return render_template("index.html", prediction=resultado)
        except Exception as e:
            print(">>> ERROR en predicción:", e)
            return render_template("index.html", error=str(e))

    return render_template("index.html")

