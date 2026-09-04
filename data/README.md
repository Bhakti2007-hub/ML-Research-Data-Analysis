# Data Directory Architecture

This directory houses raw and processed partitions of the **UCI German Credit Risk Benchmark Dataset** ($N=1,000$).

```
data/
├── raw/
│   └── german_credit_data.csv        # Immutable raw dataset downloaded from OpenML/UCI repository
├── processed/
│   ├── train_features.csv            # Engineered training feature matrix (N=800)
│   ├── test_features.csv             # Engineered test feature matrix (N=200)
│   ├── train_labels.csv              # Training target labels (0=good, 1=bad)
│   └── test_labels.csv               # Testing target labels (0=good, 1=bad)
└── README.md
```

## Data Lineage and Integrity
1. **Raw Tier (`data/raw/`)**:
   - Primary snapshot stored directly as downloaded from the UCI repository / OpenML benchmark suite (`data_id=31`).
   - Contains 21 original columns and 1,000 instances.
   - Preserves original categorical values and data distributions.

2. **Processed Tier (`data/processed/`)**:
   - Generated strictly via `src/preprocessing.py` and `src/feature_engineering.py`.
   - Stratified 80:20 train/test split (`random_state=42`).
   - Zero data leakage: feature encoders, imputers, and scalers are fitted exclusively on the 800 training observations and applied out-of-sample to the 200 testing records.
