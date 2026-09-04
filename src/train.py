"""
Model Training and Hyperparameter Optimization Pipeline.
Trains 7 benchmark classification algorithms, performs Stratified 5-Fold Cross-Validation,
executes GridSearchCV hyperparameter optimization, and saves the final best model artifact.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

# Scikit-Learn Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.inspection import permutation_importance

# Project imports
from src.utils import (
    MODELS_DIR, TABLES_DIR, METRICS_DIR, RANDOM_SEED, get_logger
)
from src.data_loader import load_raw_data
from src.preprocessing import run_preprocessing_pipeline
from src.evaluate import (
    evaluate_single_model, build_comparison_dataframe,
    perform_statistical_tests, analyze_model_errors
)
from src.visualization import (
    plot_target_distribution, plot_numerical_distributions,
    plot_boxplots_and_outliers, plot_correlation_heatmap,
    plot_categorical_analysis, plot_roc_curves,
    plot_confusion_matrices, plot_model_comparison_bars,
    plot_feature_importances, plot_hyperparameter_tuning_gain,
    plot_error_analysis
)

logger = get_logger("TrainPipeline")

def initialize_baseline_models() -> Dict[str, Any]:
    """Instantiates the 7 benchmark machine learning algorithms with reproducible configurations."""
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_split=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED
        ),
        "Support Vector Machine": SVC(
            C=1.0, kernel="rbf", gamma="scale", probability=True,
            class_weight="balanced", random_state=RANDOM_SEED
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-8
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=3,
            subsample=0.85, random_state=RANDOM_SEED
        )
    }

def train_and_benchmark_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], pd.DataFrame]:
    """
    Trains all 7 baseline algorithms, records training latencies, and computes evaluation metrics.
    """
    models = initialize_baseline_models()
    trained_models = {}
    metrics_dict = {}
    
    logger.info("--- Beginning Training & Benchmarking of 7 Machine Learning Models ---")
    
    for name, model in models.items():
        logger.info(f"Training [{name}]...")
        start_t = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_t
        
        # Evaluate
        eval_metrics = evaluate_single_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=name
        )
        eval_metrics["training_time_sec"] = float(round(train_time, 4))
        
        trained_models[name] = model
        metrics_dict[name] = eval_metrics
        
        # Save individual model metrics JSON
        metric_file = os.path.join(METRICS_DIR, f"{name.lower().replace(' ', '_')}_metrics.json")
        with open(metric_file, "w") as f:
            json.dump(eval_metrics, f, indent=2)
            
    df_comparison = build_comparison_dataframe(metrics_dict)
    comp_table_path = os.path.join(TABLES_DIR, "model_comparison.csv")
    df_comparison.to_csv(comp_table_path, index=False)
    logger.info(f"Saved model comparison table to {comp_table_path}")
    
    return trained_models, metrics_dict, df_comparison

def run_hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_metrics: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs systematic GridSearchCV hyperparameter optimization on top candidate model families.
    """
    logger.info("--- Initiating Systematic GridSearchCV Hyperparameter Tuning ---")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    tuning_grids = {
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [6, 10, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            }
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=RANDOM_SEED),
            "params": {
                "n_estimators": [100, 150],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 4],
                "subsample": [0.85, 1.0]
            }
        },
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced"),
            "params": {
                "C": [0.05, 0.1, 0.5, 1.0, 5.0],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "Support Vector Machine": {
            "estimator": SVC(probability=True, random_state=RANDOM_SEED, class_weight="balanced"),
            "params": {
                "C": [0.5, 1.0, 2.0],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"]
            }
        }
    }
    
    tuned_models = {}
    tuning_records = []
    
    for model_name, cfg in tuning_grids.items():
        logger.info(f"GridSearchCV tuning for [{model_name}]...")
        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["params"],
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        best_est = grid.best_estimator_
        
        # Evaluate tuned model on test set
        tuned_eval = evaluate_single_model(
            model=best_est,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{model_name} (Tuned)"
        )
        
        baseline_roc = baseline_metrics[model_name]["roc_auc"]
        tuned_roc = tuned_eval["roc_auc"]
        improvement = tuned_roc - baseline_roc
        
        tuned_models[model_name] = {
            "best_estimator": best_est,
            "best_params": grid.best_params_,
            "best_cv_score": float(round(grid.best_score_, 4)),
            "tuned_metrics": tuned_eval,
            "improvement_roc_auc": float(round(improvement, 4))
        }
        
        tuning_records.append({
            "Model": model_name,
            "Best Parameters": str(grid.best_params_),
            "Baseline Score (ROC-AUC)": baseline_roc,
            "Tuned Score (ROC-AUC)": tuned_roc,
            "Delta Improvement": float(round(improvement, 4)),
            "Tuned F1-Score": tuned_eval["f1_score"],
            "Tuned Accuracy": tuned_eval["accuracy"],
            "Best CV (ROC-AUC)": float(round(grid.best_score_, 4))
        })
        
        logger.info(
            f"[{model_name}] Tuned ROC-AUC: {tuned_roc:.4f} (Baseline: {baseline_roc:.4f}, "
            f"Delta: {improvement:+.4f}) | Best Params: {grid.best_params_}"
        )
        
    df_tuning = pd.DataFrame(tuning_records).sort_values(by="Tuned Score (ROC-AUC)", ascending=False).reset_index(drop=True)
    tuning_table_path = os.path.join(TABLES_DIR, "hyperparameter_tuning_results.csv")
    df_tuning.to_csv(tuning_table_path, index=False)
    logger.info(f"Saved hyperparameter tuning results to {tuning_table_path}")
    
    return tuned_models, df_tuning

def select_and_save_best_model(
    baseline_models: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
    tuned_models: Dict[str, Any],
    feature_names: list
) -> Dict[str, Any]:
    """
    Selects the champion model using multi-criteria research reasoning:
    Prioritizes ROC-AUC, F1-Score on default class, and cross-validation stability.
    Persists champion model to models/best_model.pkl.
    """
    logger.info("--- Selecting Final Champion Research Model ---")
    
    # Compare all candidates (baseline + tuned)
    candidates = {}
    for name, model in baseline_models.items():
        candidates[name] = {
            "model": model,
            "metrics": baseline_metrics[name],
            "params": model.get_params(),
            "source": "Baseline Default"
        }
    for name, t_data in tuned_models.items():
        candidates[f"{name} (Tuned)"] = {
            "model": t_data["best_estimator"],
            "metrics": t_data["tuned_metrics"],
            "params": t_data["best_params"],
            "source": "GridSearchCV Tuned"
        }
        
    # Selection Score formula: 0.50 * ROC-AUC + 0.35 * F1 + 0.15 * (1 - CV_Std)
    def composite_score(cand):
        m = cand["metrics"]
        return (0.50 * m["roc_auc"]) + (0.35 * m["f1_score"]) + (0.15 * (1.0 - m["cv_roc_auc_std"]))
        
    best_candidate_name = max(candidates.keys(), key=lambda k: composite_score(candidates[k]))
    best_cand = candidates[best_candidate_name]
    best_model = best_cand["model"]
    
    logger.info(
        f"CHAMPION MODEL SELECTED: [{best_candidate_name}] | "
        f"ROC-AUC: {best_cand['metrics']['roc_auc']:.4f}, "
        f"F1: {best_cand['metrics']['f1_score']:.4f}, "
        f"Accuracy: {best_cand['metrics']['accuracy']:.4f}"
    )
    
    # Save Best Model Artifact
    best_model_artifact = {
        "model_name": best_candidate_name,
        "model_object": best_model,
        "metrics": best_cand["metrics"],
        "parameters": best_cand["params"],
        "feature_names": feature_names,
        "source": best_cand["source"]
    }
    model_save_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model_artifact, model_save_path)
    logger.info(f"Saved Champion Model artifact to {model_save_path}")
    
    return best_model_artifact

def run_full_training_pipeline() -> Dict[str, Any]:
    """
    Orchestrates the entire research workflow:
    1. Load Raw Dataset & Run Preprocessing.
    2. Conduct EDA and save figures.
    3. Run Statistical Hypothesis Testing.
    4. Train 7 Benchmark Algorithms and evaluate.
    5. Execute GridSearchCV Hyperparameter Tuning.
    6. Select Champion Model and persist artifacts.
    7. Generate comprehensive Model Interpretation & Error Analysis.
    """
    logger.info("========== Starting Complete Machine Learning Research Workflow ==========")
    
    # 1. Ingestion & Preprocessing
    df_raw = load_raw_data()
    prep_data = run_preprocessing_pipeline(df_raw, save_artifacts=True)
    
    X_train = prep_data["X_train_trans"]
    X_test = prep_data["X_test_trans"]
    y_train = prep_data["y_train"]
    y_test = prep_data["y_test"]
    all_feature_names = prep_data["all_feature_names"]
    X_test_raw = prep_data["X_test_raw"]
    
    # 2. Generate EDA Visualizations
    logger.info("Generating publication-quality EDA figures...")
    plot_target_distribution(df_raw)
    plot_numerical_distributions(df_raw)
    plot_boxplots_and_outliers(df_raw)
    plot_correlation_heatmap(prep_data["X_train_df"])
    plot_categorical_analysis(df_raw)
    
    # 3. Statistical Hypothesis Testing
    logger.info("Executing Statistical Hypothesis Tests...")
    df_stats = perform_statistical_tests(df_raw)
    
    # 4. Train 7 Benchmark Algorithms
    baseline_models, baseline_metrics, df_comparison = train_and_benchmark_all(
        X_train, y_train, X_test, y_test
    )
    
    # 5. Hyperparameter Optimization
    tuned_models, df_tuning = run_hyperparameter_tuning(
        X_train, y_train, X_test, y_test, baseline_metrics
    )
    
    # 6. Select Champion Model
    best_artifact = select_and_save_best_model(
        baseline_models, baseline_metrics, tuned_models, all_feature_names
    )
    best_model = best_artifact["model_object"]
    
    # 7. Model Interpretation & Plots
    logger.info("Generating Model Interpretation figures...")
    # ROC curves dict
    roc_dict = {name: {"fpr": m["roc_curve"]["fpr"], "tpr": m["roc_curve"]["tpr"], "auc": m["roc_auc"]}
                for name, m in baseline_metrics.items()}
    plot_roc_curves(roc_dict)
    
    # Confusion matrices dict
    cm_dict = {name: np.array(m["confusion_matrix"]) for name, m in baseline_metrics.items()}
    plot_confusion_matrices(cm_dict)
    
    # Bar comparison plot
    plot_model_comparison_bars(df_comparison)
    plot_hyperparameter_tuning_gain(df_tuning)
    
    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=all_feature_names)
        plot_feature_importances(importances, title=f"Gini Feature Importance ({best_artifact['model_name']})")
    elif hasattr(best_model, "coef_"):
        importances = pd.Series(np.abs(best_model.coef_[0]), index=all_feature_names)
        plot_feature_importances(importances, title=f"Logistic Regression Absolute Coefficients ({best_artifact['model_name']})")
    else:
        # Permutation importance fallback
        perm_res = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED)
        importances = pd.Series(perm_res.importances_mean, index=all_feature_names)
        plot_feature_importances(importances, title=f"Permutation Feature Importance ({best_artifact['model_name']})")
        
    # Also compute Permutation Importance across all features for comparison
    perm_res_all = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_SEED)
    perm_series = pd.Series(perm_res_all.importances_mean, index=all_feature_names)
    plot_feature_importances(perm_series, title="Permutation Importance on Out-of-Sample Test Split", save_filename="permutation_importance_comparison.png")
    
    # 8. Diagnostic Error Analysis
    error_summary = analyze_model_errors(best_model, X_test, y_test, X_test_raw)
    plot_error_analysis(error_summary)
    
    logger.info("========== Machine Learning Research Workflow Completed Successfully ==========")
    return {
        "df_comparison": df_comparison,
        "df_tuning": df_tuning,
        "df_stats": df_stats,
        "best_artifact": best_artifact,
        "error_summary": error_summary
    }

if __name__ == "__main__":
    run_full_training_pipeline()
