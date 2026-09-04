"""
Utility functions and configuration constants for the ML Research Project.
Enforces reproducibility, path integrity, logging, and academic silver/grey styling.
"""

import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility Seed
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds across standard libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Project Directory Root & Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "german_credit_data.csv")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Ensure core directories exist
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Academic Research Dashboard Palette (Navy, Blue, White, High-Contrast)
THEME_COLORS = {
    "primary": "#162033",        # Primary Navy
    "secondary": "#243247",      # Secondary Navy
    "accent": "#3B82F6",         # Accent Blue
    "light_blue": "#E8F1FB",     # Light Blue Tint
    "background": "#F4F6F8",     # Main Light Background
    "cards": "#FFFFFF",          # Pure White
    "main_text": "#172033",      # Deep Primary Text
    "secondary_text": "#5B6678", # Slate Secondary Text
    "border": "#D7DEE8",         # Clean Border
    "success": "#16805C",        # Success / Creditworthy
    "warning": "#C47A00",        # Warning / Moderate Risk
    "danger": "#C44545",         # Danger / Default Risk
    "muted_icon": "#8FA0B8",     # Muted Blue-Grey
    "graphite": "#162033",
    "light_silver": "#E8F1FB",
    "dark_grey": "#243247",
    "charcoal": "#172033",
    "silver": "#D7DEE8",
    "steel": "#5B6678",
    "white": "#FFFFFF",
}

# Model Card Theme Specifications
MODEL_CARD_COLORS = {
    "Logistic Regression": "#FFFFFF",
    "Decision Tree": "#FFFFFF",
    "Random Forest": "#FFFFFF",
    "Support Vector Machine": "#FFFFFF",
    "K-Nearest Neighbors": "#FFFFFF",
    "Naive Bayes": "#FFFFFF",
    "Gradient Boosting": "#FFFFFF",
}

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    """Returns a standardized logger instance."""
    return logging.getLogger(name)

def setup_matplotlib_style() -> None:
    """Configures Matplotlib with clean, high-DPI academic research aesthetics."""
    plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D7DEE8",
        "axes.labelcolor": "#172033",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "axes.grid": True,
        "grid.color": "#E8EDF2",
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
        "xtick.color": "#172033",
        "ytick.color": "#172033",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D7DEE8",
        "legend.fontsize": 10,
        "figure.autolayout": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })
