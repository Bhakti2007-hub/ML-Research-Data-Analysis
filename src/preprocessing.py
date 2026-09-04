"""
Data Preprocessing Module for German Credit Risk Dataset.
Constructs scikit-learn ColumnTransformer and Pipeline architecture.
Guarantees zero data leakage via strict stratified train/test partitioning.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils import (
    PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_SEED, get_logger
)
from src.feature_engineering import add_domain_features

logger = get_logger("Preprocessing")

TARGET_COLUMN = "class"
TARGET_MAPPING = {"good": 0, "bad": 1}
REVERSE_TARGET_MAPPING = {0: "good", 1: "bad"}

def split_raw_data(
    df: pd.DataFrame, 
    test_size: float = 0.20, 
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Performs stratified train/test partitioning prior to any feature transformation,
    ensuring zero data leakage between training and evaluation splits.
    
    Parameters:
        df (pd.DataFrame): Raw DataFrame containing predictor features and target column.
        test_size (float): Proportion of dataset allocated to holdout evaluation (default: 0.20).
        random_state (int): Seed for deterministic reproducibility.
        
    Returns:
        Tuple[X_train, X_test, y_train, y_test]: Raw partition subsets and binary target vectors.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataframe.")
        
    X_raw = df.drop(columns=[TARGET_COLUMN]).copy()
    y_raw = df[TARGET_COLUMN].map(TARGET_MAPPING)
    
    if y_raw.isnull().any():
        # Fallback if already numeric or differently formatted
        y_raw = df[TARGET_COLUMN].apply(lambda v: 1 if str(v).lower() in ["bad", "1", "true"] else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw
    )
    
    logger.info(
        f"Partitioned data: Train={X_train.shape[0]} samples ({y_train.value_counts().to_dict()}), "
        f"Test={X_test.shape[0]} samples ({y_test.value_counts().to_dict()})"
    )
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Constructs a robust scikit-learn ColumnTransformer handling imputation,
    scaling, and one-hot encoding across respective column modalities.
    
    Parameters:
        numerical_features (List[str]): List of continuous/discrete numeric column names.
        categorical_features (List[str]): List of categorical/nominal column names.
        
    Returns:
        ColumnTransformer: Preprocessing transformer ready for fitting.
    """
    # Numerical Pipeline: Median Imputation + Standard Scaling
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical Pipeline: Most Frequent Imputation + One-Hot Encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    save_artifacts: bool = True
) -> Dict[str, Any]:
    """
    Executes the complete preprocessing workflow:
    1. Stratified train/test split on raw data.
    2. Feature engineering on train and test subsets independently.
    3. Pipeline construction and fitting strictly on train subset.
    4. Transformation of train and test sets.
    5. Artifact saving (CSVs and pipeline PKL).
    
    Returns:
        Dict containing transformed matrices, feature names, raw splits, and pipeline.
    """
    # 1. Stratified split
    X_train_raw, X_test_raw, y_train, y_test = split_raw_data(df)
    
    # 2. Domain Feature Engineering
    X_train_fe = add_domain_features(X_train_raw)
    X_test_fe = add_domain_features(X_test_raw)
    
    # Identify numerical and categorical columns after feature engineering
    numerical_cols = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()
    
    logger.info(f"Features for pipeline: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical.")
    
    # 3. Build & fit ColumnTransformer on training data only
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    preprocessor.fit(X_train_fe)
    
    # 4. Transform train and test matrices
    X_train_trans = preprocessor.transform(X_train_fe)
    X_test_trans = preprocessor.transform(X_test_fe)
    
    # Extract encoded feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numerical_cols + cat_feature_names
    
    # Convert transformed arrays back to structured DataFrames for reporting and persistence
    df_train_processed = pd.DataFrame(X_train_trans, columns=all_feature_names, index=X_train_fe.index)
    df_test_processed = pd.DataFrame(X_test_trans, columns=all_feature_names, index=X_test_fe.index)
    
    if save_artifacts:
        # Save processed CSVs
        df_train_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "train_features.csv"), index=False)
        df_test_processed.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_features.csv"), index=False)
        pd.DataFrame({"target": y_train}).to_csv(os.path.join(PROCESSED_DATA_DIR, "train_labels.csv"), index=False)
        pd.DataFrame({"target": y_test}).to_csv(os.path.join(PROCESSED_DATA_DIR, "test_labels.csv"), index=False)
        
        # Save fitted preprocessing pipeline and metadata
        pipeline_artifact = {
            "preprocessor": preprocessor,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "all_feature_names": all_feature_names,
            "target_mapping": TARGET_MAPPING
        }
        pipeline_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
        joblib.dump(pipeline_artifact, pipeline_path)
        logger.info(f"Saved preprocessing pipeline artifact to {pipeline_path}")
        
    return {
        "X_train_trans": X_train_trans,
        "X_test_trans": X_test_trans,
        "X_train_df": df_train_processed,
        "X_test_df": df_test_processed,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_fe": X_train_fe,
        "X_test_fe": X_test_fe,
        "all_feature_names": all_feature_names,
        "preprocessor": preprocessor
    }

def transform_new_sample(sample_dict: dict, pipeline_artifact_path: str = None) -> np.ndarray:
    """
    Transforms a single new inference dictionary into the exact model-ready numerical vector.
    Used by the interactive Streamlit prediction interface.
    """
    if pipeline_artifact_path is None:
        pipeline_artifact_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
        
    pipeline_data = joblib.load(pipeline_artifact_path)
    preprocessor = pipeline_data["preprocessor"]
    
    df_single = pd.DataFrame([sample_dict])
    df_fe = add_domain_features(df_single)
    
    X_trans = preprocessor.transform(df_fe)
    return X_trans

if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data()
    artifacts = run_preprocessing_pipeline(df, save_artifacts=True)
    print(f"Preprocessing completed. Feature matrix shape: {artifacts['X_train_trans'].shape}")
