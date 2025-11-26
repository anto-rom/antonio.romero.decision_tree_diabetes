# src/app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Cargar el modelo entrenado
MODEL_PATH = os.path.join("models", "modelo_arbol_diabetes.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def health():
    return "Service is up"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Espera un JSON como:
    {
      "features": [valor1, valor2, ..., valorN]
    }
    """
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    pred = int(model.predict(features)[0])
    label = "diabetes" if pred == 1 else "no_diabetes"
    return jsonify({"prediction": pred, "label": label})
