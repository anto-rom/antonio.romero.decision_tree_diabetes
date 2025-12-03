from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import warnings
from sklearn.exceptions import DataConversionWarning

# Opcional: silenciar el warning de "X does not have valid feature names"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but"
)
warnings.filterwarnings("ignore", category=DataConversionWarning)

app = Flask(__name__)

# Ruta robusta al modelo (.pkl) dentro de src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_arbol_diabetes.pkl")

# Cargar el modelo una sola vez al iniciar la app
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        print(">>> He recibido un POST en /")
        form_data = request.form.to_dict()
        print(">>> request.form:", form_data)

        try:
            # Mapear val1..val8 al orden de variables con el que entrenaste:
            # Pregnancies, Glucose, BloodPressure, SkinThickness,
            # Insulin, BMI, DiabetesPedigreeFunction, Age
            preg = float(form_data["val1"])
            glu  = float(form_data["val2"])
            bp   = float(form_data["val3"])
            skin = float(form_data["val4"])
            ins  = float(form_data["val5"])
            bmi  = float(form_data["val6"])
            dpf  = float(form_data["val7"])
            age  = float(form_data["val8"])

            X = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
            print(">>> Features recibidas:", X)

            pred = model.predict(X)[0]
            resultado = "Diabetes" if int(pred) == 1 else "No diabetes"
            print(">>> Predicción cruda:", pred, "| Resultado:", resultado)

            prediction = resultado

        except Exception as e:
            error = str(e)
            print(">>> ERROR en predicción:", e)

    # Renderiza siempre la plantilla, con o sin resultado
    return render_template("index.html", prediction=prediction, error=error)

if __name__ == "__main__":
    # Para desarrollo local; en Render se usará gunicorn
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)



