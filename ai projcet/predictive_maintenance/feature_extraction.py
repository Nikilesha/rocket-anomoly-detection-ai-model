import pandas as pd
import numpy as np


def extract_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rolling statistics and first differences for predictive maintenance.

    Parameters
    ----------
    df : pd.DataFrame
        Telemetry data (e.g., engine_temp, vibration, pressure, failure_flag)

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame preserving 'failure_flag' label.
    """
    df = df.reset_index(drop=True).copy()
    features = df.copy()

    # Identify label column
    label_col = "failure_flag" if "failure_flag" in features.columns else None

    # Exclude label from feature generation
    features_no_label = features.drop(columns=[label_col]) if label_col else features

    # Add rolling mean and first differences for numeric columns
    for col in features_no_label.columns:
        if not np.issubdtype(features_no_label[col].dtype, np.number):
            continue
        features[f"{col}_rolling_mean"] = features_no_label[col].rolling(window=3, min_periods=1).mean()
        features[f"{col}_diff"] = features_no_label[col].diff().fillna(0)

    # Replace infinite/NaN values safely
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Keep label at the end
    if label_col:
        features[label_col] = df[label_col]

    return features


def normalize_features(df: pd.DataFrame, min_vals=None, max_vals=None):
    """
    Normalize all numeric columns between 0–1 except 'failure_flag'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    min_vals : pd.Series or None
        Minimum values for each column (for consistent normalization).
    max_vals : pd.Series or None
        Maximum values for each column (for consistent normalization).

    Returns
    -------
    df_norm : pd.DataFrame
        Normalized DataFrame.
    min_vals : pd.Series
        Column-wise min values used.
    max_vals : pd.Series
        Column-wise max values used.
    """
    df_norm = df.copy()
    label = None

    # Separate label
    if "failure_flag" in df_norm.columns:
        label = df_norm["failure_flag"]
        df_norm = df_norm.drop(columns=["failure_flag"])

    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns

    # Compute min/max if not provided
    if min_vals is None:
        min_vals = df_norm[numeric_cols].min()
    if max_vals is None:
        max_vals = df_norm[numeric_cols].max()

    # Normalize numeric columns
    df_norm[numeric_cols] = (df_norm[numeric_cols] - min_vals) / (max_vals - min_vals + 1e-8)

    # Add label back
    if label is not None:
        df_norm["failure_flag"] = label

    return df_norm, min_vals, max_vals
