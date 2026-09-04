"""
Model Evaluation and Statistical Analysis Module.
Calculates comprehensive classification metrics, Stratified 5-Fold CV metrics,
asymmetric financial risk error breakdown, and rigorous statistical hypothesis tests.
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, brier_score_loss,
    classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

from src.utils import METRICS_DIR, TABLES_DIR, RANDOM_SEED, get_logger

logger = get_logger("Evaluation")

def evaluate_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Evaluates a single model comprehensively on test data and Stratified K-Fold CV.
    """
    # 1. Prediction & Latency
    start_infer = time.time()
    y_pred = model.predict(X_test)
    infer_latency_ms = (time.time() - start_infer) * 1000 / max(len(y_test), 1)
    
    # 2. Probability Scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_scores = model.decision_function(X_test)
        # Min-max scale decision function to [0, 1]
        y_prob = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-8)
    else:
        y_prob = y_pred.astype(float)
        
    # 3. Core Classification Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc_auc = 0.5
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC Curve points
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    
    # 4. Stratified K-Fold Cross-Validation on Training Set
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores_acc = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    try:
        cv_scores_roc = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")
    except Exception:
        cv_scores_roc = cv_scores_acc
        
    metrics = {
        "model_name": model_name,
        "accuracy": float(round(acc, 4)),
        "precision": float(round(prec, 4)),
        "recall": float(round(rec, 4)),
        "f1_score": float(round(f1, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "brier_score": float(round(brier, 4)),
        "cv_accuracy_mean": float(round(cv_scores_acc.mean(), 4)),
        "cv_accuracy_std": float(round(cv_scores_acc.std(), 4)),
        "cv_roc_auc_mean": float(round(cv_scores_roc.mean(), 4)),
        "cv_roc_auc_std": float(round(cv_scores_roc.std(), 4)),
        "inference_latency_ms": float(round(infer_latency_ms, 3)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    }
    
    logger.info(
        f"[{model_name}] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | "
        f"CV ROC: {metrics['cv_roc_auc_mean']:.4f} (+/- {metrics['cv_roc_auc_std']:.4f})"
    )
    return metrics

def build_comparison_dataframe(metrics_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Assembles a standardized benchmark comparison table across all evaluated models.
    """
    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1_score"],
            "ROC-AUC": m["roc_auc"],
            "Brier Score": m["brier_score"],
            "CV Mean (ROC-AUC)": m["cv_roc_auc_mean"],
            "CV Std (ROC-AUC)": m["cv_roc_auc_std"],
            "CV Mean (Acc)": m["cv_accuracy_mean"],
            "CV Std (Acc)": m["cv_accuracy_std"],
            "Latency (ms/sample)": m["inference_latency_ms"]
        })
    df_comp = pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return df_comp

def perform_statistical_tests(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Conducts rigorous statistical hypothesis tests on the German Credit dataset:
    1. Chi-Square Tests of Independence for Categorical Variables vs. Credit Risk.
    2. Two-Sample Student's t-test and Mann-Whitney U tests for Continuous Attributes across classes.
    3. One-Way ANOVA tests across loan purposes.
    """
    records = []
    
    # 1. Categorical Features Chi-Square Tests
    cat_cols = ["checking_status", "credit_history", "savings_status", "purpose", "housing", "property_magnitude", "employment"]
    for col in cat_cols:
        if col in df_raw.columns:
            contingency_table = pd.crosstab(df_raw[col], df_raw["class"])
            chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
            records.append({
                "Test Type": "Chi-Square Independence",
                "Feature": col,
                "Comparison": f"{col} vs. Credit Risk Class",
                "Statistic (Chi2 / t / F)": round(chi2, 4),
                "p-value": float(f"{p_val:.4e}"),
                "Degrees of Freedom": dof,
                "Statistically Significant (alpha=0.05)": "Yes" if p_val < 0.05 else "No",
                "Interpretation": f"Significant association (p={p_val:.2e})" if p_val < 0.05 else "No significant association"
            })
            
    # 2. Continuous Features: Two-Sample t-test & Mann-Whitney U test
    num_cols = ["duration", "credit_amount", "age", "installment_commitment"]
    good_mask = (df_raw["class"] == "good") | (df_raw["class"] == 0)
    bad_mask = (df_raw["class"] == "bad") | (df_raw["class"] == 1)
    
    for col in num_cols:
        if col in df_raw.columns:
            good_vals = df_raw.loc[good_mask, col].dropna()
            bad_vals = df_raw.loc[bad_mask, col].dropna()
            
            # Student's t-test
            t_stat, t_pval = stats.ttest_ind(good_vals, bad_vals, equal_var=False)
            records.append({
                "Test Type": "Welch's Two-Sample t-test",
                "Feature": col,
                "Comparison": "Good vs. Bad Borrowers Mean",
                "Statistic (Chi2 / t / F)": round(t_stat, 4),
                "p-value": float(f"{t_pval:.4e}"),
                "Degrees of Freedom": len(good_vals) + len(bad_vals) - 2,
                "Statistically Significant (alpha=0.05)": "Yes" if t_pval < 0.05 else "No",
                "Interpretation": f"Mean difference is significant (t={t_stat:.2f}, p={t_pval:.2e})" if t_pval < 0.05 else "No significant difference in means"
            })
            
            # Non-parametric Mann-Whitney U test
            u_stat, u_pval = stats.mannwhitneyu(good_vals, bad_vals, alternative="two-sided")
            records.append({
                "Test Type": "Mann-Whitney U Test",
                "Feature": col,
                "Comparison": "Good vs. Bad Distribution Rank",
                "Statistic (Chi2 / t / F)": round(u_stat, 2),
                "p-value": float(f"{u_pval:.4e}"),
                "Degrees of Freedom": np.nan,
                "Statistically Significant (alpha=0.05)": "Yes" if u_pval < 0.05 else "No",
                "Interpretation": f"Distribution ranks differ significantly (p={u_pval:.2e})" if u_pval < 0.05 else "Distribution ranks similar"
            })
            
    # 3. One-Way ANOVA across Loan Purposes for Credit Amount
    if "purpose" in df_raw.columns and "credit_amount" in df_raw.columns:
        purposes = df_raw["purpose"].unique()
        groups = [df_raw.loc[df_raw["purpose"] == p, "credit_amount"].values for p in purposes if len(df_raw.loc[df_raw["purpose"] == p]) > 5]
        f_stat, f_pval = stats.f_oneway(*groups)
        records.append({
            "Test Type": "One-Way ANOVA",
            "Feature": "credit_amount",
            "Comparison": "Credit Amount across Loan Purposes",
            "Statistic (Chi2 / t / F)": round(f_stat, 4),
            "p-value": float(f"{f_pval:.4e}"),
            "Degrees of Freedom": len(groups) - 1,
            "Statistically Significant (alpha=0.05)": "Yes" if f_pval < 0.05 else "No",
            "Interpretation": "Mean credit amounts differ significantly across purpose categories" if f_pval < 0.05 else "No significant difference"
        })
        
    df_stats = pd.DataFrame(records)
    table_path = os.path.join(TABLES_DIR, "statistical_tests.csv")
    df_stats.to_csv(table_path, index=False)
    logger.info(f"Saved statistical hypothesis testing results to {table_path}")
    return df_stats

def analyze_model_errors(
    model: Any,
    X_test_trans: np.ndarray,
    y_test: np.ndarray,
    X_test_raw: pd.DataFrame
) -> Dict[str, Any]:
    """
    Conducts in-depth diagnostic error analysis on test predictions:
    Identifies False Positives (Safe borrower rejected / flagged as bad) and
    False Negatives (Defaulting borrower approved / predicted as good).
    """
    y_pred = model.predict(X_test_trans)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_trans)[:, 1]
    else:
        y_prob = y_pred.astype(float)
        
    cm = confusion_matrix(y_test, y_pred)
    # cm layout: [[TN, FP], [FN, TP]] where 0=good, 1=bad
    tn, fp, fn, tp = cm.ravel()
    
    df_eval = X_test_raw.copy().reset_index(drop=True)
    df_eval["actual_class"] = y_test
    df_eval["predicted_class"] = y_pred
    df_eval["default_probability"] = y_prob.round(4)
    
    # Error subsets
    df_fp = df_eval[(df_eval["actual_class"] == 0) & (df_eval["predicted_class"] == 1)]
    df_fn = df_eval[(df_eval["actual_class"] == 1) & (df_eval["predicted_class"] == 0)]
    df_tp = df_eval[(df_eval["actual_class"] == 1) & (df_eval["predicted_class"] == 1)]
    df_tn = df_eval[(df_eval["actual_class"] == 0) & (df_eval["predicted_class"] == 0)]
    
    summary = {
        "total_test_samples": int(len(y_test)),
        "correct": int(tn + tp),
        "true_negatives": int(tn),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "type_i_error_rate": float(round(fp / (tn + fp), 4)) if (tn + fp) > 0 else 0.0,
        "type_ii_error_rate": float(round(fn / (fn + tp), 4)) if (fn + tp) > 0 else 0.0,
        "fp_profile": {
            "mean_duration": float(round(df_fp["duration"].mean(), 1)) if len(df_fp) > 0 else 0,
            "mean_credit_amount": float(round(df_fp["credit_amount"].mean(), 1)) if len(df_fp) > 0 else 0,
            "mean_age": float(round(df_fp["age"].mean(), 1)) if len(df_fp) > 0 else 0
        },
        "fn_profile": {
            "mean_duration": float(round(df_fn["duration"].mean(), 1)) if len(df_fn) > 0 else 0,
            "mean_credit_amount": float(round(df_fn["credit_amount"].mean(), 1)) if len(df_fn) > 0 else 0,
            "mean_age": float(round(df_fn["age"].mean(), 1)) if len(df_fn) > 0 else 0
        }
    }
    
    # Save error analysis table
    error_table_path = os.path.join(TABLES_DIR, "error_analysis_samples.csv")
    df_eval.to_csv(error_table_path, index=False)
    
    logger.info(
        f"Error Analysis: {summary['correct']} Correct, {fp} False Positives (Type I), "
        f"{fn} False Negatives (Type II / Costly Defaults)"
    )
    return summary
