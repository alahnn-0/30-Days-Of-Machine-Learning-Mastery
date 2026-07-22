"""
data_loader.py
Loads the UCI Hepatitis dataset. Falls back to a synthetic dataset with the
same schema if the download fails (e.g. no internet access), so the rest of
the pipeline can still be developed/tested offline.
"""

import numpy as np
import pandas as pd

COLUMNS = [
    "Class", "Age", "Sex", "Steroid", "Antivirals", "Fatigue", "Malaise",
    "Anorexia", "LiverBig", "LiverFirm", "SpleenPalpable", "Spiders",
    "Ascites", "Varices", "Bilirubin", "AlkPhosphate", "Sgot", "Albumin",
    "Protime", "Histology",
]

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"

RANDOM_STATE = 42


def _make_synthetic(n: int = 155, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Class": rng.choice([1, 2], size=n, p=[0.2, 0.8]),
        "Age": rng.integers(7, 80, n),
        "Sex": rng.choice([1, 2], n),
        "Steroid": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Antivirals": rng.choice([1, 2], n),
        "Fatigue": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Malaise": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Anorexia": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "LiverBig": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "LiverFirm": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "SpleenPalpable": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Spiders": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Ascites": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Varices": rng.choice([1, 2, np.nan], n, p=[0.45, 0.45, 0.10]),
        "Bilirubin": np.round(rng.gamma(2, 0.7, n), 1),
        "AlkPhosphate": rng.integers(30, 250, n).astype(float),
        "Sgot": rng.integers(10, 400, n).astype(float),
        "Albumin": np.round(rng.normal(3.8, 0.6, n), 1),
        "Protime": rng.integers(30, 100, n).astype(float),
        "Histology": rng.choice([1, 2], n),
    })


def load_data(url: str = UCI_URL) -> pd.DataFrame:
    """Load the UCI Hepatitis dataset, or a synthetic fallback on failure."""
    try:
        df = pd.read_csv(url, names=COLUMNS, na_values="?")
        print(f"Loaded real UCI dataset: {df.shape}")
        return df
    except Exception as e:
        print(f"Could not download dataset ({e}). Using synthetic fallback data.")
        return _make_synthetic()


if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum().sort_values(ascending=False))