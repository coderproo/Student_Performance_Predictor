from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import joblib


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["gpa"]),
        float(request.form["study_time"]),
        float(request.form["absences"]),
        int(request.form["volunteering"]),
        int(request.form["extracurricular"]),
        int(request.form["tutoring"]),
    ]
    
    
    input_scaled = scaler.transform([values])
    prediction = model.predict(input_scaled)[0]
    
    result = "Pass" if prediction == 1 else "Fail"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
