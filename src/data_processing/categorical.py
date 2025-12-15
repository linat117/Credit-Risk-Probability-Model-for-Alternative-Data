import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode_categorical_features(df: pd.DataFrame, cat_cols: list):
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    encoded = encoder.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=df.index
    )

    df = df.drop(columns=cat_cols)
    df = pd.concat([df, encoded_df], axis=1)

    return df
