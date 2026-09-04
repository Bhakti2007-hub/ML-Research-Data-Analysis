# Machine Learning Public Dataset Study
*An Experimental and Research-Oriented Machine Learning Investigation on Credit Default Risk*

---

## Abstract
This project delivers an end-to-end, academic/research-grade machine learning system evaluating the **UCI German Credit Risk Benchmark Dataset** ($N=1,000$, 20 predictor attributes). Through a strict zero-data-leakage pipeline, domain-grounded financial ratio engineering, Stratified 5-Fold Cross-Validation, and systematic hyperparameter optimization via GridSearchCV, we benchmark seven classification algorithms (Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, K-Nearest Neighbors, Gaussian Naive Bayes, and Gradient Boosting). Our findings demonstrate that tuned tree ensembles (Random Forest ROC-AUC = $0.8135$, Gradient Boosting ROC-AUC = $0.8082$) achieve superior discriminatory capacity, while Support Vector Machines maximize minority-class default recall ($76.67\%$). Furthermore, regularized Logistic Regression provides a parsimonious, regulatory-compliant linear baseline (ROC-AUC = $0.8045$, latency = $0.025$ ms/sample). All results are presented via an interactive, 9-page research analytics platform built in Streamlit using an academic Silver/Grey/Graphite/White aesthetic.

---

## Problem Statement
In financial credit underwriting, predicting borrower default is characterized by severe class imbalance ($70:30$) and asymmetric misclassification economics: granting credit to an applicant who defaults (Type II Error / False Negative) causes catastrophic capital loss, whereas rejecting a creditworthy applicant (Type I Error / False Positive) forfeits only the interest spread. This study formulates and evaluates models that optimize the balance between discriminatory power (ROC-AUC), default recall, and economic risk exposure.

---

## Research Objectives
1. Implement a complete academic research workflow from dataset acquisition to post-hoc interpretability.
2. Conduct statistical hypothesis tests ($\chi^2$, Welch's $t$-test, Mann-Whitney U, ANOVA) to evaluate predictor significance.
3. Prevent data leakage through unified scikit-learn `Pipeline` and `ColumnTransformer` constructs.
4. Benchmark seven algorithm families under identical Stratified 5-Fold Cross-Validation splits.
5. Optimize hyperparameters via GridSearchCV and quantify empirical performance gains.
6. Diagnose misclassification patterns through Type I / Type II error profiling and feature attribution.
7. Deliver an interactive Streamlit research analytics dashboard.

---

## Research Questions
- **RQ1**: Which financial and demographic attributes exert the strongest predictive influence on credit default risk?
- **RQ2**: Which machine learning algorithm family achieves the optimal trade-off between discriminatory power (ROC-AUC) and default recall (F1-score)?
- **RQ3**: Does systematic hyperparameter optimization via GridSearchCV yield statistically significant generalization gains over baseline models?
- **RQ4**: How stable and resilient are different model architectures under Stratified 5-Fold Cross-Validation?
- **RQ5**: What specific applicant profiles characterize misclassified instances (false positives vs. false negatives)?
- **RQ6**: Can an interpretable regularized linear model (Logistic Regression) perform competitively with complex non-linear ensembles?

---

## Dataset
- **Name**: UCI German Credit Risk Dataset (Statlog German Credit Data)
- **Observations**: 1,000 instances (800 training, 200 holdout test)
- **Attributes**: 20 baseline features (7 numerical, 13 categorical) expanded to 26 features via domain synthesis.
- **Target**: `class` (`good` = 700 [70%], `bad` / default = 300 [30%])

---

## Dataset Source
- **Repository**: UCI Machine Learning Repository / OpenML Benchmark Suite (`data_id = 31`)
- **Citation**: Prof. Hans Hofmann (1994), Institute for Statistics and Econometrics, University of Hamburg, Germany.
- **Reference**: [UCI Machine Learning Repository Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

---

## Technologies Used
- **Language**: Python 3.12 / 3.x
- **Core Analytics**: Pandas, NumPy, SciPy
- **Machine Learning**: Scikit-learn, Joblib
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Interactive Platform**: Streamlit
- **Notebook Development**: Jupyter, IPyKernel

---

## Project Architecture
```
Machine_Learning_Public_Dataset_Study/
├── data/
│   ├── raw/
│   │   └── german_credit_data.csv        # Immutable raw dataset
│   ├── processed/
│   │   ├── train_features.csv            # Processed training matrix (N=800)
│   │   ├── test_features.csv             # Processed testing matrix (N=200)
│   │   ├── train_labels.csv              # Training targets
│   │   └── test_labels.csv               # Testing targets
│   └── README.md
├── notebooks/
│   ├── 01_data_understanding.ipynb       # Data ingestion & schema profiling
│   ├── 02_data_preprocessing.ipynb       # Zero-leakage pipeline & transformation
│   ├── 03_exploratory_data_analysis.ipynb# Publication figures & EDA
│   ├── 04_feature_engineering.ipynb      # Domain ratio synthesis
│   ├── 05_model_training.ipynb           # Model fitting & cross-validation
│   ├── 06_model_comparison.ipynb         # GridSearchCV tuning & comparison
│   └── 07_final_analysis.ipynb           # Interpretation & error diagnosis
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Ingestion & schema validation
│   ├── preprocessing.py                  # Pipeline & ColumnTransformer
│   ├── feature_engineering.py            # Financial ratio engineering
│   ├── train.py                          # Training & tuning orchestrator
│   ├── evaluate.py                       # Metrics, CV & statistical tests
│   ├── visualization.py                  # Academic greyscale plotting
│   └── utils.py                          # Configuration, paths & theme
├── models/
│   ├── best_model.pkl                    # Champion model artifact
│   └── preprocessing_pipeline.pkl        # Fitted preprocessing pipeline
├── results/
│   ├── figures/                          # 16 publication-quality figures
│   ├── metrics/                          # Model-specific JSON metrics
│   └── tables/                           # CSV summary & benchmark tables
├── reports/
│   ├── methodology.md                    # Mathematical & empirical design
│   ├── research_findings.md              # RQ1-RQ6 answers & empirical findings
│   ├── model_analysis.md                 # Comparative mechanics & error breakdown
│   └── final_report.md                   # 22-section academic research paper
├── app/
│   └── streamlit_app.py                  # 9-page research analytics dashboard
├── requirements.txt                      # Dependency manifest
├── README.md                             # Primary project documentation
├── dataset_information.md                # 11-section dataset metadata
├── .gitignore                            # VCS exclusions
└── LICENSE                               # MIT License
```

---

## Methodology
The experimental protocol guarantees zero data leakage by splitting the raw dataset into Stratified 80:20 partitions prior to computing descriptive parameters or constructing pipelines. Features are scaled using `StandardScaler` and encoded via `OneHotEncoder`, followed by 5-fold cross-validation and hyperparameter optimization.

---

## Data Preprocessing
- **Partitioning**: 800 training records, 200 holdout testing records (`random_state=42`).
- **Imputation**: Median imputation for numerical features; modal imputation for categorical features.
- **Scaling & Encoding**: `StandardScaler` for continuous features; `OneHotEncoder(handle_unknown='ignore')` for nominal attributes.

---

## Exploratory Data Analysis
- **Target Balance**: $70\%$ Good vs $30\%$ Bad.
- **Liquidity Association**: Checking account status $<0$ DM experiences a $49.3\%$ default frequency vs $11.6\%$ for accounts $\ge 200$ DM.
- **Tenure Sensitivity**: Default rates scale monotonically with loan duration ($>45\%$ default rate for terms $>36$ months).

---

## Machine Learning Models
1. **Logistic Regression ($L_2$)**
2. **Decision Tree (CART)**
3. **Random Forest (Ensemble Bagging)**
4. **Support Vector Machine (RBF Kernel)**
5. **K-Nearest Neighbors (Distance-Weighted)**
6. **Gaussian Naive Bayes**
7. **Gradient Boosting Classifier (Tree Boosting)**

---

## Evaluation Metrics
- **Accuracy, Precision, Recall, F1-Score, ROC-AUC, Brier Score Loss, and Mean Inference Latency**.

---

## Cross Validation
- **Protocol**: Stratified 5-Fold Cross-Validation preserving the 70:30 class distribution.
- **Result**: SVM and Random Forest exhibited superior cross-validation stability ($\sigma < 0.035$).

---

## Hyperparameter Tuning
- **Random Forest**: ROC-AUC improved from $0.7985 \to 0.8135$ ($\Delta = +0.0150$).
- **Gradient Boosting**: ROC-AUC improved from $0.7956 \to 0.8082$ ($\Delta = +0.0126$).
- **Logistic Regression**: Optimized $C=0.05$ (`liblinear`), attaining ROC-AUC = $0.8074$.

---

## Model Interpretation
- **Dominant Predictors**: `checking_status`, `duration`, `credit_to_duration_ratio`, `credit_history`, `savings_status`, `credit_amount`.
- **Permutation Impact**: Permuting `checking_status` degrades out-of-sample test ROC-AUC by $>0.08$.

---

## Results
```
========================================================================================
Model                   Accuracy   Precision   Recall   F1-Score   ROC-AUC   CV ROC-AUC
----------------------------------------------------------------------------------------
Random Forest (Tuned)   0.7900     0.6852      0.6167   0.6491     0.8135    0.8050 (+/- 0.0267)
Gradient Boosting       0.8000     0.6939      0.5667   0.6296     0.7956    0.7798 (+/- 0.0348)
Logistic Regression     0.7500     0.5641      0.7333   0.6528     0.8045    0.7760 (+/- 0.0488)
SVM (RBF)               0.7600     0.5897      0.7667   0.6471     0.8001    0.8028 (+/- 0.0207)
Naive Bayes (Gaussian)  0.6950     0.4938      0.6667   0.5674     0.7533    0.7144 (+/- 0.0516)
K-Nearest Neighbors     0.7150     0.5294      0.3000   0.4242     0.7092    0.7473 (+/- 0.0284)
Decision Tree (CART)    0.6750     0.4706      0.6667   0.5324     0.7104    0.7082 (+/- 0.0747)
========================================================================================
```

---

## Research Findings
- **RQ1**: Liquidity (`checking_status`) and debt burden (`credit_to_duration_ratio`) are the preeminent determinants of default solvency.
- **RQ2**: Random Forest and Gradient Boosting deliver top discriminatory accuracy; SVM delivers highest default recall ($76.67\%$).
- **RQ3**: GridSearchCV yields statistically significant generalization gains for tree ensembles ($\Delta > +0.012$).
- **RQ4**: Ensemble models and SVM show tight cross-validation stability ($\sigma < 0.035$).
- **RQ5**: False positives stem from high credit amounts relative to age; false negatives stem from short employment tenures hidden behind clean past credit.
- **RQ6**: Regularized Logistic Regression provides a highly competitive, parsimonious linear alternative ($\text{ROC-AUC}=0.8045$).

---

## Streamlit Dashboard
An interactive 9-page research platform implementing a strict **Silver / Grey / White / Graphite** aesthetic:
1. **Overview**: Abstract, objectives, metrics, and methodology workflow.
2. **Dataset Explorer**: Schema inspection, distributions, and summary statistics.
3. **Data Analysis**: Interactive distributions, boxplots, and correlation matrices.
4. **Model Training**: Execution dashboard with greyscale model cards.
5. **Model Comparison**: Metric bar charts, ROC curves, and CV stability boxplots.
6. **Model Interpretation**: Tree Gini importance, permutation attribution, and confusion matrices.
7. **Prediction**: Live applicant scoring interface with real-time probability gauge and risk card.
8. **Research Findings**: Empirical answers to RQ1–RQ6 and statistical hypothesis test summaries.
9. **About Project**: Citations, architecture, and academic metadata.

---

## Installation
```bash
# 1. Clone repository
git clone https://github.com/your-username/Machine_Learning_Public_Dataset_Study.git
cd Machine_Learning_Public_Dataset_Study

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run
```bash
# Execute the complete research pipeline (Training, Tuning, Evaluation, Figures, Reports)
python -m src.train

# Launch the interactive Streamlit research analytics dashboard
streamlit run app/streamlit_app.py
```

---

## Project Structure
Detailed in Section **Project Architecture** above.

---

## Limitations
1. Sample size $N=1,000$ precludes deep neural tabular architectures.
2. Monetary values reflect historical Deutsche Mark baselines.
3. Static dataset without dynamic macroeconomic indicators.

---

## Future Scope
1. Implement TabNet and attention-based tabular models.
2. Integrate algorithmic fairness constraints (Equalized Odds, Demographic Parity).
3. Evaluate on contemporary multi-institutional credit datasets.

---

## Research Publication Potential
Suitable for:
- IEEE / Springer conference proceedings on Applied AI in Financial Econometrics
- Academic B.Tech Capstone / Master's thesis portfolio
- Research internship demonstration of rigorous empirical ML methodology

---

## License
Distributed under the **MIT License**. See `LICENSE` for details.
