"""
preprocess.py
Imputation, target encoding, stratified train/test split, and scaling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_loader import load_data, RANDOM_STATE


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """
    Returns X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
    X_train/X_test are imputed but unscaled (for tree models).
    X_train_scaled/X_test_scaled are additionally scaled (for linear/SVM models).
    """
    X = df.drop(columns=["Class"])
    y = (df["Class"] == 1).astype(int)  # 1 = Die/severe (positive class), 0 = Live

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(df)
    print("Train shape:", X_train.shape, " Test shape:", X_test.shape)
    print("Train class balance:\n", y_train.value_counts())