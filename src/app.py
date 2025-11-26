from flask import Flask, request, render_template
from pickle import load
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load(open("decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Sin diabetes",
    "1": "Diabetes",
}
df = pd.read_csv("data.csv")

num_variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'Age']
scaler = StandardScaler()
scaler.fit(df[num_variables])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        Pregnancies = float(request.form["val1"])
        Glucose = float(request.form["val2"])
        BloodPressure = float(request.form["val3"])
        BMI = float(request.form["val4"])
        DiabetesPedigreeFunction = float(request.form["val5"])
        Age = float(request.form["val6"])

        data = np.array([[Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age]])

        data_normalized = scaler.transform(data)

        prediction = str(model.predict(data_normalized)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction=pred_class)
