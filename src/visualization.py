"""
Visualization Module for German Credit Risk ML Research Study.
Renders publication-quality academic figures in a strict Silver/Grey/Graphite/White aesthetic.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from src.utils import (
    FIGURES_DIR, THEME_COLORS, MODEL_CARD_COLORS,
    setup_matplotlib_style, get_logger
)

logger = get_logger("Visualization")
setup_matplotlib_style()

# Academic Research Colormap definitions
RESEARCH_CMAP = sns.dark_palette("#162033", as_cmap=True)
RESEARCH_PALETTE = ["#3B82F6", "#162033", "#243247", "#16805C", "#5B6678", "#C47A00", "#6366F1"]
CLASS_PALETTE = {"Good Credit (0)": "#3B82F6", "Bad Credit / Default (1)": "#C44545"}
BINARY_RESEARCH_PALETTE = ["#3B82F6", "#C44545"]

def plot_target_distribution(df: pd.DataFrame, target_col: str = "class", save_filename: str = "target_distribution.png") -> str:
    """Plots and saves the credit risk target variable frequency and percentage distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    counts = df[target_col].value_counts()
    
    # Bar plot
    bars = ax1.bar(
        [str(k).capitalize() for k in counts.index],
        counts.values,
        color=["#3B82F6", "#C44545"],
        edgecolor="#D7DEE8",
        width=0.55
    )
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 12, f"{int(h)} ({h/len(df):.1%})",
                 ha="center", va="bottom", fontsize=10, fontweight="bold", color="#172033")
    ax1.set_title("Credit Risk Class Frequency", fontweight="bold", pad=12, color="#172033")
    ax1.set_xlabel("Credit Risk Category", fontweight="bold", color="#172033")
    ax1.set_ylabel("Number of Applicants", fontweight="bold", color="#172033")
    ax1.set_ylim(0, max(counts.values) * 1.18)
    
    # Donut Chart
    wedges, texts, autotexts = ax2.pie(
        counts.values,
        labels=[f"{k.capitalize()} Credit" for k in counts.index],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#3B82F6", "#C44545"],
        wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=2.0),
        textprops=dict(color="#172033", fontsize=10, fontweight="bold")
    )
    for autotext in autotexts:
        autotext.set_color("#FFFFFF")
        autotext.set_fontsize(10)
        autotext.set_weight("bold")
    ax2.set_title("Class Proportion (70:30 Imbalance)", fontweight="bold", pad=12, color="#172033")
    
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved target distribution figure to {save_path}")
    return save_path

def plot_numerical_distributions(df: pd.DataFrame, num_cols: List[str] = None, save_filename: str = "numerical_feature_distributions.png") -> str:
    """Plots histograms with KDE overlays for key numerical features."""
    if num_cols is None:
        num_cols = ["duration", "credit_amount", "age", "installment_commitment", "existing_credits", "residence_since"]
        
    cols = [c for c in num_cols if c in df.columns]
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.8 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, col in enumerate(cols):
        ax = axes[idx]
        sns.histplot(df[col], kde=True, ax=ax, color="#3B82F6", edgecolor="#D7DEE8", line_kws={"color": "#162033", "linewidth": 2})
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#162033", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="#3B82F6", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.1f}")
        ax.set_title(f"Distribution: {col.replace('_', ' ').title()}", fontweight="bold", color="#172033")
        ax.set_xlabel(col.replace('_', ' ').title(), color="#172033")
        ax.set_ylabel("Density / Count", color="#172033")
        ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D7DEE8")
        
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
        
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved numerical distributions figure to {save_path}")
    return save_path

def plot_boxplots_and_outliers(df: pd.DataFrame, num_cols: List[str] = None, save_filename: str = "boxplots_outliers.png") -> str:
    """Generates boxplots across numerical features stratified by credit risk status."""
    if num_cols is None:
        num_cols = ["duration", "credit_amount", "age", "installment_commitment"]
    cols = [c for c in num_cols if c in df.columns]
    
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4.5))
    axes = axes if len(cols) > 1 else [axes]
    
    for idx, col in enumerate(cols):
        ax = axes[idx]
        sns.boxplot(
            x="class", y=col, data=df, ax=ax,
            palette=["#E8F1FB", "#FEE2E2"],
            boxprops=dict(edgecolor="#162033", linewidth=1.2),
            whiskerprops=dict(color="#162033", linewidth=1.2),
            capprops=dict(color="#162033", linewidth=1.2),
            medianprops=dict(color="#162033", linewidth=2.0)
        )
        ax.set_title(f"{col.replace('_', ' ').title()} by Risk", fontweight="bold", color="#172033")
        ax.set_xlabel("Credit Class", fontweight="bold", color="#172033")
        ax.set_ylabel(col.replace('_', ' ').title(), fontweight="bold", color="#172033")
        
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved boxplots figure to {save_path}")
    return save_path

def plot_correlation_heatmap(df: pd.DataFrame, save_filename: str = "correlation_heatmap.png") -> str:
    """Computes and renders correlation matrix heatmap across numerical and engineered features."""
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, cmap="Blues", vmin=-1.0, vmax=1.0, center=0,
        annot=True, fmt=".2f", square=True, linewidths=0.8, linecolor="#FFFFFF",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax, annot_kws={"size": 9, "fontweight": "medium"}
    )
    ax.set_title("Inter-Feature Pearson Correlation Heatmap", fontweight="bold", pad=14, color="#172033")
    
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved correlation heatmap to {save_path}")
    return save_path

def plot_categorical_analysis(df: pd.DataFrame, cat_cols: List[str] = None, save_filename: str = "categorical_feature_analysis.png") -> str:
    """Analyzes default rates across critical categorical features."""
    if cat_cols is None:
        cat_cols = ["checking_status", "credit_history", "savings_status", "purpose", "employment", "housing"]
    cols = [c for c in cat_cols if c in df.columns]
    
    n_cols = 2
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.2 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for idx, col in enumerate(cols):
        ax = axes[idx]
        cross_tab = pd.crosstab(df[col], df["class"], normalize="index") * 100
        cross_tab.plot(
            kind="barh", stacked=True, ax=ax,
            color=["#3B82F6", "#C44545"], edgecolor="#FFFFFF", linewidth=0.8
        )
        ax.set_title(f"Risk Proportion by {col.replace('_', ' ').title()}", fontweight="bold", color="#172033")
        ax.set_xlabel("Percentage (%)", color="#172033")
        ax.set_ylabel("", color="#172033")
        ax.legend(["Good (0)", "Bad / Default (1)"], frameon=True, facecolor="#FFFFFF", edgecolor="#D7DEE8", loc="lower right")
        
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
        
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved categorical analysis figure to {save_path}")
    return save_path

def plot_roc_curves(roc_dict: Dict[str, Any], save_filename: str = "roc_curves_all_models.png") -> str:
    """Plots comparative Receiver Operating Characteristic (ROC) curves across all algorithms."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    
    styles = ["-", "-", "-", "-", "--", "--", "-."]
    palette = ["#162033", "#3B82F6", "#16805C", "#C47A00", "#6366F1", "#0D9488", "#C44545"]
    
    for idx, (model_name, data) in enumerate(roc_dict.items()):
        fpr = data["fpr"]
        tpr = data["tpr"]
        auc_score = data["auc"]
        style = styles[idx % len(styles)]
        color_val = palette[idx % len(palette)]
        ax.plot(fpr, tpr, linestyle=style, color=color_val, linewidth=2.0, label=f"{model_name} (AUC = {auc_score:.3f})")
        
    ax.plot([0, 1], [0, 1], linestyle=":", color="#8FA0B8", linewidth=1.5, label="Random Baseline (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", color="#172033")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold", color="#172033")
    ax.set_title("Comparative ROC Curves Across Classification Models", fontweight="bold", pad=12, color="#172033")
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#D7DEE8")
    
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved ROC curves figure to {save_path}")
    return save_path

def plot_confusion_matrices(cm_dict: Dict[str, np.ndarray], save_filename: str = "confusion_matrices_comparison.png") -> str:
    """Renders confusion matrices across all 7 evaluated machine learning models."""
    n_models = len(cm_dict)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.8 * n_rows))
    axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Good (0)", "Bad (1)"], yticklabels=["Good (0)", "Bad (1)"],
            linewidths=1.0, linecolor="#D7DEE8",
            annot_kws={"size": 11, "fontweight": "bold"}
        )
        ax.set_title(f"{model_name}", fontweight="bold", fontsize=11, color="#172033")
        ax.set_xlabel("Predicted Class", color="#172033")
        ax.set_ylabel("Actual Class", color="#172033")
        
    for j in range(n_models, len(axes)):
        fig.delaxes(axes[j])
        
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved confusion matrices figure to {save_path}")
    return save_path

def plot_model_comparison_bars(comparison_df: pd.DataFrame, save_filename: str = "model_comparison_metrics.png") -> str:
    """Plots multi-metric comparative bar chart in academic navy/blue theme."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ["Accuracy", "F1", "ROC-AUC", "Recall"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    x = np.arange(len(comparison_df))
    width = 0.20
    shades = ["#93C5FD", "#3B82F6", "#162033", "#16805C"]
    
    for idx, metric in enumerate(available_metrics):
        ax.bar(
            x + (idx - len(available_metrics)/2 + 0.5) * width,
            comparison_df[metric],
            width,
            label=metric,
            color=shades[idx % len(shades)],
            edgecolor="#D7DEE8",
            linewidth=0.8
        )
        
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["Model"], rotation=25, ha="right", fontweight="semibold", color="#172033")
    ax.set_ylabel("Score (0.0 to 1.0)", fontweight="bold", color="#172033")
    ax.set_ylim(0, 1.08)
    ax.set_title("Benchmarked Model Performance Across Primary Metrics", fontweight="bold", pad=12, color="#172033")
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D7DEE8", loc="lower right")
    
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved model comparison bar chart to {save_path}")
    return save_path

def plot_feature_importances(importances: pd.Series, top_n: int = 15, title: str = "Top Feature Importances", save_filename: str = "feature_importance_best_model.png") -> str:
    """Renders top feature importances horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    top_features = importances.sort_values(ascending=True).tail(top_n)
    y_pos = np.arange(len(top_features))
    
    bars = ax.barh(y_pos, top_features.values, color="#3B82F6", edgecolor="#162033", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feat.replace("num__", "").replace("cat__", "") for feat in top_features.index], fontweight="medium", color="#172033")
    ax.set_xlabel("Relative Importance Score", fontweight="bold", color="#172033")
    ax.set_title(title, fontweight="bold", pad=12, color="#172033")
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2., f"{w:.3f}",
                ha="left", va="center", fontsize=9, fontweight="bold", color="#172033")
                
    ax.set_xlim(0, max(top_features.values) * 1.15)
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved feature importance figure to {save_path}")
    return save_path

def plot_hyperparameter_tuning_gain(tuning_df: pd.DataFrame, save_filename: str = "hyperparameter_tuning_comparison.png") -> str:
    """Renders before vs. after hyperparameter tuning performance bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x = np.arange(len(tuning_df))
    width = 0.35
    
    ax.bar(x - width/2, tuning_df["Baseline Score (ROC-AUC)"], width, label="Baseline (Default)", color="#93C5FD", edgecolor="#D7DEE8")
    ax.bar(x + width/2, tuning_df["Tuned Score (ROC-AUC)"], width, label="Optimized (GridSearchCV)", color="#162033", edgecolor="#D7DEE8")
    
    ax.set_xticks(x)
    ax.set_xticklabels(tuning_df["Model"], rotation=15, ha="right", fontweight="semibold", color="#172033")
    ax.set_ylabel("ROC-AUC Score", fontweight="bold", color="#172033")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Hyperparameter Optimization: Baseline vs. Tuned Models", fontweight="bold", pad=12, color="#172033")
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D7DEE8")
    
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved hyperparameter tuning comparison to {save_path}")
    return save_path

def plot_error_analysis(error_summary: Dict[str, Any], save_filename: str = "error_analysis_breakdown.png") -> str:
    """Plots misclassification error breakdown (Type I False Positives vs Type II False Negatives)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # Error type counts
    error_types = ["Correct Predictions", "False Positives (Type I)", "False Negatives (Type II / High Risk)"]
    error_counts = [error_summary["correct"], error_summary["false_positives"], error_summary["false_negatives"]]
    
    ax1.bar(
        ["Correct", "False Pos (FP)", "False Neg (FN)"],
        error_counts,
        color=["#16805C", "#C47A00", "#C44545"],
        edgecolor="#D7DEE8",
        width=0.55
    )
    ax1.set_title("Prediction Outcomes Breakdown", fontweight="bold", color="#172033")
    ax1.set_ylabel("Number of Holdout Test Instances", fontweight="bold", color="#172033")
    for idx, count in enumerate(error_counts):
        ax1.text(idx, count + 2, f"{count} ({count/sum(error_counts):.1%})", ha="center", fontweight="bold", color="#172033")
        
    # Cost impact chart: Asymmetric loss
    # Typical credit loss cost: FP = 1 unit (opportunity cost), FN = 5 units (default loss)
    fp_cost = error_summary["false_positives"] * 1
    fn_cost = error_summary["false_negatives"] * 5
    
    ax2.bar(
        ["Opportunity Cost (FP)", "Default Loss (FN, 5x Cost)"],
        [fp_cost, fn_cost],
        color=["#C47A00", "#C44545"],
        edgecolor="#D7DEE8",
        width=0.5
    )
    ax2.set_title("Financial Risk Cost Impact (Asymmetric 1:5)", fontweight="bold", color="#172033")
    ax2.set_ylabel("Estimated Financial Risk Loss Units", fontweight="bold", color="#172033")
    ax2.text(0, fp_cost + 2, f"{fp_cost} units", ha="center", fontweight="bold", color="#172033")
    ax2.text(1, fn_cost + 4, f"{fn_cost} units", ha="center", fontweight="bold", color="#172033")
    
    save_path = os.path.join(FIGURES_DIR, save_filename)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved error analysis figure to {save_path}")
    return save_path
