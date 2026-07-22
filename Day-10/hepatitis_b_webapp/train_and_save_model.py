"""
train_and_save_model.py
Trains the Random Forest model (the one that doesn't need scaling, which
keeps the Flask inference code simpler) and saves it + the fitted imputer +
feature column order to model_artifacts.joblib.

Run this once before starting the Flask app:
    python train_and_save_model.py
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import pandas as pd

from data_loader import load_data, RANDOM_STATE

MODEL_PATH = "model_artifacts.joblib"


def main():
    df = load_data()

    X = df.drop(columns=["Class"])
    y = (df["Class"] == 1).astype(int)  # 1 = Die/severe (positive class), 0 = Live

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_imputed, y)

    joblib.dump(
        {
            "model": model,
            "imputer": imputer,
            "feature_columns": list(X.columns),
        },
        MODEL_PATH,
    )
    print(f"Saved model + imputer + feature order to {MODEL_PATH}")


if __name__ == "__main__":
    main()