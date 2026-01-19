from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import os

# Optional but recommended
from docx import Document

app = Flask(__name__)

# -------------------------
# Load ML model
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model (2).pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

# -------------------------
# Load article (DOCX)
# -------------------------
def load_article():
    docx_path = os.path.join(BASE_DIR, "Jay 2025 science fair Narrative school.docx")
    if not os.path.exists(docx_path):
        return "<i>Research article file not found.</i>"

    doc = Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "<br><br>".join(paragraphs)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    article_html = load_article()
    return render_template_string(HTML, article=article_html)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        age = float(data["age"])
        gender = 0 if data["gender"] == "male" else 1
        race = {"white": 0, "black": 1, "asian": 2, "other": 3}[data["race"]]
        qrs = float(data["qrs"])
        qt = float(data["qt"])
        vr = float(data["vr"])

        X = np.array([[age, gender, race, qrs, qt, vr]])
        lvef = round(float(model.predict(X)[0]), 2)

        status = (
            "Normal" if lvef >= 55 else
            "Borderline Reduced" if lvef >= 40 else
            "Reduced"
        )

        return jsonify({"lvef": lvef, "status": status})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------
# HTML TEMPLATE
# -------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LVEF.ai | AI-Powered Cardiac Assessment</title>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">

<style>
body {
    margin: 0;
    font-family: Inter, sans-serif;
    background: #f8fafc;
    color: #111;
}

nav {
    padding: 20px 60px;
    display: flex;
    justify-content: space-between;
    background: white;
    box-shadow: 0 1px 10px rgba(0,0,0,0.05);
}

nav b { font-size: 1.2em; }
nav a {
    margin-left: 30px;
    text-decoration: none;
    color: #333;
    font-weight: 600;
}

.hero {
    padding: 100px 60px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

.section {
    padding: 80px 60px;
    max-width: 1100px;
    margin: auto;
}

.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 20px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

.demo {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    max-width: 520px;
}

label {
    font-weight: 600;
    font-size: 0.85em;
    margin-top: 14px;
    display: block;
}

input, select {
    width: 100%;
    margin-top: 6px;
}

button {
    width: 100%;
    margin-top: 24px;
    padding: 14px;
    border-radius: 12px;
    border: none;
    background: #667eea;
    color: white;
    font-weight: 700;
    cursor: pointer;
}

#gauge {
    text-align: center;
    margin-top: 30px;
}

.circle {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    background: conic-gradient(#667eea 0deg, #eee 0deg);
    display: flex;
    align-items: center;
    justify-content: center;
    margin: auto;
}

.value {
    font-size: 1.8em;
    font-weight: 800;
}

.article-box {
    background: white;
    padding: 50px;
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.08);
    line-height: 1.8;
    font-size: 1.05em;
    max-height: 700px;
    overflow-y: auto;
}

footer {
    padding: 40px;
    text-align: center;
    background: #111;
    color: #bbb;
}
</style>
</head>

<body>

<nav>
    <b>LVEF.ai</b>
    <div>
        <a href="#product">Product</a>
        <a href="#demo">Live Demo</a>
        <a href="#research">Research</a>
    </div>
</nav>

<section class="hero">
    <h1>AI-Powered LVEF Estimation</h1>
    <p>
        Estimate left ventricular ejection fraction from ECG-derived
        parameters using machine learning.
    </p>
</section>

<section id="product" class="section">
    <h2>Why LVEF.ai?</h2>
    <div class="cards">
        <div class="card">⚡ Real-time ECG-based inference</div>
        <div class="card">🧠 ML trained on large clinical datasets</div>
        <div class="card">🏥 Potential to reduce echo costs</div>
        <div class="card">🔬 Built as a research-grade prototype</div>
    </div>
</section>

<section id="demo" class="section">
    <h2>Interactive Demo</h2>
    <div class="demo">

        <label>Age</label>
        <input id="age" type="number" value="55">

        <label>Gender</label>
        <select id="gender">
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>

        <label>Race</label>
        <select id="race">
            <option value="white">White</option>
            <option value="black">Black</option>
            <option value="asian">Asian</option>
            <option value="other">Other</option>
        </select>

        <label>QRS Duration (ms)</label>
        <input id="qrs" type="number" value="100">

        <label>QTc (ms)</label>
        <input id="qt" type="number" value="420">

        <label>Ventricular Rate (bpm)</label>
        <input id="vr" type="number" value="70">

        <button onclick="predict()">Run Prediction</button>

        <div id="gauge">
            <div class="circle" id="circle">
                <div class="value" id="lvef">--%</div>
            </div>
            <div id="status"></div>
            <div id="error" style="color:red;"></div>
        </div>

    </div>
</section>

<section id="research" class="section">
    <h2>Research Article</h2>
    <div class="article-box">
        {{ article | safe }}
    </div>

    <p style="margin-top:20px;">
        📄 <a href="/static/LVEF_Narrative.pdf" target="_blank">
        Download full PDF version</a>
    </p>
</section>

<footer>
    © 2026 LVEF.ai · Research prototype · Not for clinical use
</footer>

<script>
async function predict() {
    const data = {
        age: document.getElementById("age").value,
        gender: document.getElementById("gender").value,
        race: document.getElementById("race").value,
        qrs: document.getElementById("qrs").value,
        qt: document.getElementById("qt").value,
        vr: document.getElementById("vr").value
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const out = await res.json();

    if (out.error) {
        document.getElementById("error").innerText = out.error;
        return;
    }

    document.getElementById("lvef").innerText = out.lvef + "%";
    document.getElementById("status").innerText = out.status;
    document.getElementById("circle").style.background =
        `conic-gradient(#667eea ${out.lvef * 3.6}deg, #eee 0deg)`;
}
</script>

</body>
</html>
"""

# -------------------------
# Run (Render-compatible)
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
