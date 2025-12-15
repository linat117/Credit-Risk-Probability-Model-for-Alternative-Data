import pandas as pd

def create_rfm_features(df: pd.DataFrame, snapshot_date=None):
    """Create RFM features for each customer."""

    df = df.copy()

    # ✅ Convert to datetime (CRITICAL FIX)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    else:
        snapshot_date = pd.to_datetime(snapshot_date)

    rfm_df = (
        df.groupby('CustomerId')
        .agg(
            recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
            frequency=('TransactionId', 'count'),
            monetary=('Amount', 'sum')
        )
        .reset_index()
    )

    return rfm_df
