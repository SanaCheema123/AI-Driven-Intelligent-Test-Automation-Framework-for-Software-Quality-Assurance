import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from config.config import RAW_DATA_PATH, MODEL_PATH, METADATA_PATH
from src.preprocessing import load_dataset, split_features_target, build_preprocessor


def train():
    df = load_dataset(RAW_DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=220, random_state=42, class_weight="balanced", max_depth=12
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = {}
    best_name = None
    best_pipeline = None
    best_score = -1

    for name, model in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", model)
        ])
        pipeline.fit(X_train, y_train)

        pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            prob = pred

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        }
        results[name] = metrics

        ranking_score = metrics["f1_score"] + metrics["roc_auc"]
        if ranking_score > best_score:
            best_score = ranking_score
            best_name = name
            best_pipeline = pipeline

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "project": "AI-Driven Intelligent Test Automation Framework for Software Quality Assurance",
        "dataset": "NASA JM1 Software Defect Dataset",
        "records": int(len(df)),
        "features": int(X.shape[1]),
        "target": "defects",
        "best_model": best_name,
        "metrics": results[best_name],
        "all_model_results": results,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_columns": list(X.columns)
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Training completed successfully.")
    print("Best model:", best_name)
    print("Metrics:", results[best_name])


if __name__ == "__main__":
    train()
