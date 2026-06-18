import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from config.config import MODEL_PATH, FEATURE_COLUMNS


def predict_defect(payload):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src\\train_model.py")

    model = joblib.load(MODEL_PATH)
    row = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        row[col] = pd.to_numeric(row[col], errors="coerce")

    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0][1])

    if prob >= 0.70:
        risk = "High"
        recommendation = "Prioritize this module for regression testing, code review, and automated test generation."
    elif prob >= 0.40:
        risk = "Medium"
        recommendation = "Add focused unit tests and monitor this module in the next QA cycle."
    else:
        risk = "Low"
        recommendation = "Standard test coverage is sufficient. Continue routine QA validation."

    return {
        "defect_prediction": "Defective" if pred == 1 else "Non-Defective",
        "defect_probability": prob,
        "risk_level": risk,
        "recommendation": recommendation
    }
