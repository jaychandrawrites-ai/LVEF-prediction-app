from flask import Flask, request, render_template_string
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load your pickle model
model_path = os.path.join("model", "model (2).pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LVEF Predictor</title>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
* {
    box-sizing: border-box;
}

body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea, #764ba2);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0;
}

.card {
    background: white;
    width: 420px;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.2);
}

h1 {
    text-align: center;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 0.9em;
    color: #666;
    margin-bottom: 25px;
}

label {
    font-size: 0.85em;
    font-weight: 600;
    margin-top: 12px;
    display: block;
}

input, select {
    width: 100%;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #ddd;
    margin-top: 5px;
    font-size: 0.9em;
}

input:focus, select:focus {
    outline: none;
    border-color: #667eea;
}

button {
    width: 100%;
    padding: 12px;
    margin-top: 25px;
    border-radius: 10px;
    border: none;
    background: #667eea;
    color: white;
    font-size: 1em;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(102,126,234,0.4);
}

.result {
    margin-top: 25px;
    padding: 15px;
    background: #f0f4ff;
    border-radius: 10px;
    text-align: center;
    font-weight: 600;
    color: #333;
}

.error {
    margin-top: 20px;
    padding: 12px;
    background: #ffe6e6;
    border-radius: 10px;
    color: #b00020;
    text-align: center;
    font-weight: 600;
}
</style>
</head>

<body>
<div class="card">
    <h1>LVEF Predictor</h1>
    <div class="subtitle">Machine learning–based cardiac function estimate</div>

    <form method="POST">
        <label>Age</label>
        <input type="number" name="age" required>

        <label>Gender</label>
        <select name="gender" required>
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>

        <label>Race</label>
        <select name="race" required>
            <option value="white">White</option>
            <option value="black">Black</option>
            <option value="asian">Asian</option>
            <option value="other">Other</option>
        </select>

        <label>QRS Duration (ms)</label>
        <input type="number" name="qrs_duration" step="0.1" required>

        <label>QT Corrected (ms)</label>
        <input type="number" name="qt_corrected" step="0.1" required>

        <label>Ventricular Rate (bpm)</label>
        <input type="number" name="ventricular_rate" step="0.1" required>

        <button type="submit">Predict LVEF</button>
    </form>

    {% if lvef_value is not none %}
    <div class="result">
        Predicted LVEF: {{ lvef_value }}
    </div>
    {% endif %}

    {% if error %}
    <div class="error">
        {{ error }}
    </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    lvef_value = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            gender = request.form["gender"]
            race = request.form["race"]
            qrs_duration = float(request.form["qrs_duration"])
            qt_corrected = float(request.form["qt_corrected"])
            ventricular_rate = float(request.form["ventricular_rate"])

            gender_val = 0 if gender.lower() == "male" else 1
            race_val = {"white": 0, "black": 1, "asian": 2, "other": 3}.get(race.lower(), 3)

            features = np.array([[age, gender_val, race_val,
                                  qrs_duration, qt_corrected, ventricular_rate]])

            lvef_value = round(float(model.predict(features)[0]), 2)

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        lvef_value=lvef_value,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
