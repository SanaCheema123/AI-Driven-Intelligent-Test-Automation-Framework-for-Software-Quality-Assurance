from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "jm1.csv"
MODEL_PATH = BASE_DIR / "models" / "qa_defect_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "qa_preprocessor.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
LOG_PATH = BASE_DIR / "data" / "logs" / "qa_prediction_logs.csv"

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"
]
TARGET_COLUMN = "defects"
