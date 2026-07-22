"""
app.py
Flask web app for the Hepatitis outcome-risk prediction model.

Run:
    python train_and_save_model.py   # once, to create model_artifacts.joblib
    python app.py                    # starts the server on http://localhost:5000
"""

import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

MODEL_PATH = "model_artifacts.joblib"

app = Flask(__name__)

# --- Load model artifacts at startup (train first if missing) ---
if not os.path.exists(MODEL_PATH):
    print("No saved model found — training one now (first run only)...")
    import train_and_save_model
    train_and_save_model.main()

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
imputer = artifacts["imputer"]
FEATURE_COLUMNS = artifacts["feature_columns"]

# Fields the form collects, in the order the model expects them.
# type: "number" for continuous labs/age, "binary" for Yes/No (1=Yes, 2=No),
# "sex" for the Sex field (1=Male, 2=Female).
FIELD_SPECS = [
    ("Age", "number", "Age (years)"),
    ("Sex", "sex", "Sex"),
    ("Steroid", "binary", "On steroid treatment?"),
    ("Antivirals", "binary", "On antiviral treatment?"),
    ("Fatigue", "binary", "Fatigue"),
    ("Malaise", "binary", "Malaise"),
    ("Anorexia", "binary", "Anorexia (loss of appetite)"),
    ("LiverBig", "binary", "Liver enlarged?"),
    ("LiverFirm", "binary", "Liver firm?"),
    ("SpleenPalpable", "binary", "Spleen palpable?"),
    ("Spiders", "binary", "Spider angiomata present?"),
    ("Ascites", "binary", "Ascites present?"),
    ("Varices", "binary", "Varices present?"),
    ("Bilirubin", "number", "Bilirubin (mg/dL)"),
    ("AlkPhosphate", "number", "Alkaline Phosphatase (U/L)"),
    ("Sgot", "number", "SGOT / AST (U/L)"),
    ("Albumin", "number", "Albumin (g/dL)"),
    ("Protime", "number", "Prothrombin time (seconds)"),
    ("Histology", "binary", "Abnormal histology?"),
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FIELD_SPECS, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {}
        for name, ftype, _ in FIELD_SPECS:
            raw = request.form.get(name, "").strip()
            if raw == "":
                row[name] = None
            elif ftype == "number":
                row[name] = float(raw)
            else:
                # binary / sex fields come from <select> as "1" or "2"
                row[name] = float(raw)

        X = pd.DataFrame([row])[FEATURE_COLUMNS]
        X_imputed = pd.DataFrame(imputer.transform(X), columns=FEATURE_COLUMNS)

        proba = model.predict_proba(X_imputed)[0][1]  # probability of "Die"/severe class
        pred = int(proba >= 0.5)

        result = {
            "prediction": "Higher risk" if pred == 1 else "Lower risk",
            "probability": round(proba * 100, 1),
            "risk_class": "high" if pred == 1 else "low",
        }
        return render_template("index.html", fields=FIELD_SPECS, result=result, form_values=request.form)

    except Exception as e:
        error = f"Could not process input: {e}"
        return render_template("index.html", fields=FIELD_SPECS, result=None, error=error, form_values=request.form)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)