import json
import pandas as pd

from config.config import METADATA_PATH, LOG_PATH


def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_logs():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()


def monitoring_summary():
    logs = load_logs()
    if logs.empty:
        return {
            "total_predictions": 0,
            "high_risk_modules": 0,
            "average_defect_probability": 0,
            "last_prediction_time": "No predictions yet"
        }

    return {
        "total_predictions": int(len(logs)),
        "high_risk_modules": int((logs["risk_level"] == "High").sum()),
        "average_defect_probability": round(float(logs["defect_probability"].mean() * 100), 2),
        "last_prediction_time": str(logs["timestamp"].iloc[-1])
    }
