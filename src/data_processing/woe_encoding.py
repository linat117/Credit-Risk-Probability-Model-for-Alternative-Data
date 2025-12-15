import pandas as pd
import numpy as np

def woe_encode_features(df: pd.DataFrame, target_col: str, cat_cols: list):
    """
    Manually compute WoE encoding for categorical variables.
    """
    df_encoded = df.copy()

    for col in cat_cols:
        woe_map = {}

        grouped = df.groupby(col)[target_col].agg(['sum', 'count'])
        grouped['non_event'] = grouped['count'] - grouped['sum']

        total_event = grouped['sum'].sum()
        total_non_event = grouped['non_event'].sum()

        for category in grouped.index:
            event_rate = grouped.loc[category, 'sum'] / total_event if total_event > 0 else 0
            non_event_rate = grouped.loc[category, 'non_event'] / total_non_event if total_non_event > 0 else 0

            woe = np.log((event_rate + 1e-6) / (non_event_rate + 1e-6))
            woe_map[category] = woe

        df_encoded[col] = df[col].map(woe_map)

    return df_encoded
