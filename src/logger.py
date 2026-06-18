from datetime import datetime
import pandas as pd
from config.config import LOG_PATH


def log_prediction(payload, result):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": result["defect_prediction"],
        "defect_probability": round(result["defect_probability"], 4),
        "risk_level": result["risk_level"]
    }
    row.update(payload)
    df = pd.DataFrame([row])

    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(LOG_PATH, index=False)
