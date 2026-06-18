import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config.config import FEATURE_COLUMNS, TARGET_COLUMN


def load_dataset(path):
    df = pd.read_csv(path)
    return clean_dataset(df)


def clean_dataset(df):
    df = df.copy()
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.lower().map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })

    df = df.dropna(subset=[TARGET_COLUMN])
    return df


def build_preprocessor():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


def split_features_target(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return X, y
