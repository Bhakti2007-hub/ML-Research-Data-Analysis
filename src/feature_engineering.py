"""
Feature Engineering Module for German Credit Risk Dataset.
Creates domain-grounded financial risk ratios, liquidity composite indices,
purpose categorizations, and logarithmic variance stabilization transforms.
"""

import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger("FeatureEngineering")

def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-grounded financial risk features for credit default modeling.
    
    Engineered Attributes:
    1. `credit_to_duration_ratio`: Monthly capital repayment burden (Credit Amount / Duration).
    2. `credit_to_age_ratio`: Financial leverage relative to life stage (Credit Amount / Age).
    3. `liquidity_index`: Composite ordinal liquidity rating across checking & savings accounts.
    4. `high_risk_purpose`: Binary indicator for speculative/unsecured expenditure (education, vacation, business).
    5. `log_credit_amount`: Log-transformed loan amount to normalize positive skewness.
    6. `log_duration`: Log-transformed loan tenure.
    
    Parameters:
        df (pd.DataFrame): Raw or pre-split feature DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame augmented with engineered features.
    """
    df_out = df.copy()
    
    # 1. Monthly Debt Burden / Burn Rate (Credit to Duration Ratio)
    # Higher value indicates greater monthly cash-flow strain on the borrower.
    df_out["credit_to_duration_ratio"] = df_out["credit_amount"] / (df_out["duration"] + 1e-5)
    
    # 2. Credit to Age Leverage Ratio
    # High leverage relative to age captures younger borrowers with disproportionately large credit exposure.
    df_out["credit_to_age_ratio"] = df_out["credit_amount"] / (df_out["age"] + 1e-5)
    
    # 3. Composite Liquidity Index
    # Checking status mapping
    checking_map = {
        "<0": 0,
        "0<=X<200": 1,
        ">=200": 2,
        "no checking": 1
    }
    # Savings status mapping
    savings_map = {
        "<100": 0,
        "100<=X<500": 1,
        "500<=X<1000": 2,
        ">=1000": 3,
        "no known savings": 1
    }
    
    c_rank = df_out["checking_status"].astype(str).map(checking_map).fillna(1)
    s_rank = df_out["savings_status"].astype(str).map(savings_map).fillna(1)
    df_out["liquidity_index"] = c_rank + s_rank
    
    # 4. High-Risk Purpose Indicator
    high_risk_purposes = ["education", "vacation", "retraining", "business", "repairs"]
    df_out["high_risk_purpose"] = df_out["purpose"].astype(str).isin(high_risk_purposes).astype(int)
    
    # 5. Logarithmic Transformations to mitigate heavy right-skewness
    df_out["log_credit_amount"] = np.log1p(df_out["credit_amount"])
    df_out["log_duration"] = np.log1p(df_out["duration"])
    
    logger.info(f"Engineered 6 domain features. New shape: {df_out.shape}")
    return df_out

def get_feature_descriptions() -> dict:
    """Returns documentation and mathematical rationale for all engineered features."""
    return {
        "credit_to_duration_ratio": {
            "formula": "credit_amount / duration",
            "rationale": "Represents the monthly repayment obligation. High values compress disposable income, amplifying default risk."
        },
        "credit_to_age_ratio": {
            "formula": "credit_amount / age",
            "rationale": "Captures credit exposure scaled against borrower earning maturity and asset accumulation stage."
        },
        "liquidity_index": {
            "formula": "ordinal_rank(checking_status) + ordinal_rank(savings_status)",
            "rationale": "Synthesizes multi-account liquidity reserves into a single monotonic buffer rating."
        },
        "high_risk_purpose": {
            "formula": "1 if purpose in [education, vacation, business, retraining, repairs] else 0",
            "rationale": "Separates uncollateralized / high-variance loan applications from standard consumer asset purchases."
        },
        "log_credit_amount": {
            "formula": "log(1 + credit_amount)",
            "rationale": "Compensates for right-skewed loan size distribution, linearizing relationships for gradient and kernel models."
        },
        "log_duration": {
            "formula": "log(1 + duration)",
            "rationale": "Compresses long-tail tenures to normalize variance across credit durations."
        }
    }

if __name__ == "__main__":
    from src.data_loader import load_raw_data
    df = load_raw_data()
    df_fe = add_domain_features(df)
    print("Engineered feature columns preview:")
    print(df_fe[["credit_amount", "duration", "credit_to_duration_ratio", "liquidity_index", "high_risk_purpose"]].head())
