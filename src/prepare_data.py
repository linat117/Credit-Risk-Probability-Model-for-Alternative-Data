from src.data_processing.aggregate_features import create_rfm_features
from src.data_processing.time_features import extract_time_features
from src.data_processing.categorical import encode_categorical_features
from src.data_processing.missing_values import handle_missing_values
from src.data_processing.scaling import scale_numeric_features
from src.data_processing.woe_encoding import woe_encode_features



def prepare_model_data(raw_df, target_col=None):
    """Combine all preprocessing steps to prepare model-ready data."""
    # Step 1: RFM features
    rfm_df = create_rfm_features(raw_df)
    df = raw_df.merge(rfm_df, on='CustomerId', how='left')

    # Step 2: Time features
    df = extract_time_features(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)
   

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = encode_categorical_features(df, cat_cols)

    # Step 4: Scale numeric features
    num_cols = ['recency', 'frequency', 'monetary']
    df = scale_numeric_features(df, num_cols)

    # Step 5: Optional WoE encoding
    if target_col:
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        df = woe_encode_features(df, target_col, cat_cols)

    return df
