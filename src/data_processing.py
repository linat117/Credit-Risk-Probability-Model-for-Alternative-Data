import pandas as pd
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df["Amount"] = df["Amount"].fillna(0)
    df["Value"] = df["Value"].fillna(0)
    return df
