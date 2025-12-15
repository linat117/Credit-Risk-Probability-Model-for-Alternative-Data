import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def cluster_customers_rfm(rfm_df: pd.DataFrame, n_clusters=3, random_state=42):
    """
    Cluster customers based on RFM features using KMeans.
    """

    rfm_features = rfm_df[['recency', 'frequency', 'monetary']]

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    rfm_df['rfm_cluster'] = kmeans.fit_predict(rfm_scaled)

    return rfm_df

def assign_high_risk_label(rfm_df: pd.DataFrame):
    """
    Identify least engaged cluster and assign binary risk label.
    """

    cluster_summary = (
        rfm_df
        .groupby('rfm_cluster')
        .agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': 'mean'
        })
    )

    # Least engaged cluster
    high_risk_cluster = cluster_summary.sort_values(
        by=['frequency', 'monetary'],
        ascending=True
    ).index[0]

    rfm_df['is_high_risk'] = (
        rfm_df['rfm_cluster'] == high_risk_cluster
    ).astype(int)

    return rfm_df
