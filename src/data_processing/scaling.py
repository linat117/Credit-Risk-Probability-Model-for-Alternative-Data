import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_numeric_features(df: pd.DataFrame, num_cols: list) -> pd.DataFrame:
    """Standardize numeric features (mean=0, std=1)."""
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df
