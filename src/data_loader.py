"""
Data Loader Module for German Credit Risk Dataset.
Handles robust raw data ingestion, schema validation, type integrity, and dataset introspection.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.datasets import fetch_openml
from src.utils import RAW_DATA_PATH, DATA_DIR, get_logger

logger = get_logger("DataLoader")

def load_raw_data(filepath: str = RAW_DATA_PATH, force_download: bool = False) -> pd.DataFrame:
    """
    Loads the raw German Credit dataset from local storage or fetches it from OpenML (data_id=31).
    
    Parameters:
        filepath (str): Target CSV path.
        force_download (bool): If True, forces redownload from OpenML.
        
    Returns:
        pd.DataFrame: Raw dataset.
    """
    if os.path.exists(filepath) and not force_download:
        logger.info(f"Loading raw dataset from local disk: {filepath}")
        df = pd.read_csv(filepath)
    else:
        logger.info("Local raw dataset not found or force_download=True. Fetching from OpenML (data_id=31)...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        ds = fetch_openml(data_id=31, as_frame=True, parser="auto")
        df = ds.frame
        df.to_csv(filepath, index=False)
        logger.info(f"Saved freshly downloaded raw dataset to {filepath}")
        
    logger.info(f"Successfully loaded dataset with shape: {df.shape} ({df.shape[0]} rows, {df.shape[1]} columns)")
    return df

def inspect_dataset_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes dataset schema, identifying numerical and categorical feature partitions.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Remove target if present
    if "class" in categorical_cols:
        categorical_cols.remove("class")
    if "class" in numerical_cols:
        numerical_cols.remove("class")
        
    schema_info = {
        "num_rows": int(df.shape[0]),
        "num_cols": int(df.shape[1]),
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "target_column": "class" if "class" in df.columns else None,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    logger.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features.")
    return schema_info

def check_missing_and_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive missing value and duplicate analysis.
    """
    missing_series = df.isnull().sum()
    missing_dict = missing_series[missing_series > 0].to_dict()
    duplicates_count = int(df.duplicated().sum())
    
    report = {
        "total_missing_values": int(missing_series.sum()),
        "missing_per_column": missing_dict,
        "duplicate_rows_count": duplicates_count
    }
    logger.info(f"Missing values check: {report['total_missing_values']} missing, {duplicates_count} duplicate rows.")
    return report

def analyze_target_distribution(df: pd.DataFrame, target_col: str = "class") -> Dict[str, Any]:
    """
    Calculates class counts and proportions for the target variable.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        
    counts = df[target_col].value_counts().to_dict()
    proportions = df[target_col].value_counts(normalize=True).to_dict()
    
    distribution_info = {
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_proportions": {str(k): float(round(v, 4)) for k, v in proportions.items()},
        "imbalance_ratio": float(round(max(counts.values()) / min(counts.values()), 2))
    }
    logger.info(f"Target Distribution: {distribution_info['class_counts']} (Imbalance Ratio: {distribution_info['imbalance_ratio']}:1)")
    return distribution_info

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes extended descriptive statistics including mean, std, median, skewness, IQR.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().T
    stats["median"] = numeric_df.median()
    stats["skewness"] = numeric_df.skew()
    stats["kurtosis"] = numeric_df.kurtosis()
    stats["iqr"] = stats["75%"] - stats["25%"]
    return stats

if __name__ == "__main__":
    df = load_raw_data()
    schema = inspect_dataset_schema(df)
    missing = check_missing_and_duplicates(df)
    target_dist = analyze_target_distribution(df)
    stats = compute_summary_statistics(df)
    print("\n--- Dataset Summary Statistics ---")
    print(stats.round(2))
