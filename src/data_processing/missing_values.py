import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Impute missing values for numeric columns."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    imputer = SimpleImputer(strategy=strategy)
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df
