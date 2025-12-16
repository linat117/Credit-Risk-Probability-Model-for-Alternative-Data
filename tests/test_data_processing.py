import pandas as pd
from src.prepare_data import prepare_model_data

def test_prepare_model_data_returns_dataframe():
    df_sample = pd.DataFrame({
        "CustomerId": [1,2],
        "TransactionStartTime": ["2025-12-01","2025-12-02"],
        "Amount": [100, 200]
    })
    df_ready = prepare_model_data(df_sample)
    assert isinstance(df_ready, pd.DataFrame)
    assert "recency" in df_ready.columns
