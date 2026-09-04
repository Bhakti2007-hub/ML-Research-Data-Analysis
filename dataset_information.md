# Dataset Information: UCI German Credit Risk Benchmark

## 1. Dataset Name
**German Credit Risk Dataset (Statlog German Credit)**  
*Primary Citation Reference: Prof. Hans Hofmann, Institute for Statistics and Econometrics, University of Hamburg, Germany.*

---

## 2. Dataset Source
- **Repository**: UCI Machine Learning Repository / OpenML Benchmark Archive
- **OpenML Identifier**: `data_id = 31` (`credit-g`)
- **UCI URL**: [https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **OpenML URL**: [https://www.openml.org/d/31](https://www.openml.org/d/31)

---

## 3. Dataset URL / Source Reference
- Direct raw data archive: `data/raw/german_credit_data.csv`
- Original creator: Prof. Hans Hofmann (1994), University of Hamburg
- Donated to UCI Machine Learning Repository and OpenML benchmarking suite.

---

## 4. Number of Records (Instances)
- **Total Records ($N$)**: 1,000 observations (individual loan applications).
- **Train Split ($N_{train}$)**: 800 observations (80% stratified).
- **Test Split ($N_{test}$)**: 200 observations (20% stratified).

---

## 5. Number of Features
- **Total Attributes**: 21 (20 predictor features + 1 target variable `class`).
- **Feature Breakdown**:
  - **7 Numerical / Continuous / Discrete Features**
  - **13 Categorical / Ordinal / Nominal Features**

---

## 6. Feature Descriptions

| Feature Name | Data Type | Domain / Values | Description |
| :--- | :--- | :--- | :--- |
| `checking_status` | Categorical | `<0`, `0<=X<200`, `>=200`, `no checking` | Status of applicant's existing checking account (liquidity gauge) |
| `duration` | Numerical (Integer) | 4 to 72 months | Credit / loan duration in months |
| `credit_history` | Categorical | `no credits/all paid`, `all paid`, `existing paid`, `delayed previously`, `critical/other existing credit` | Applicant's past credit repayment behavior |
| `purpose` | Categorical | `new car`, `used car`, `furniture/equipment`, `radio/tv`, `domestic appliances`, `repairs`, `education`, `vacation`, `retraining`, `business`, `other` | Purpose of the credit application |
| `credit_amount` | Numerical (Integer) | 250 to 18,424 DM (Deutsche Mark) | Total loan / credit amount requested |
| `savings_status` | Categorical | `<100`, `100<=X<500`, `500<=X<1000`, `>=1000`, `no known savings` | Balance in applicant's savings account or bonds |
| `employment` | Categorical | `unemployed`, `<1`, `1<=X<4`, `4<=X<7`, `>=7` | Duration of present employment (in years) |
| `installment_commitment` | Numerical (Integer) | 1, 2, 3, 4 | Installment rate in percentage of disposable income |
| `personal_status` | Categorical | `male div/sep`, `female div/dep/mar`, `male single`, `male mar/wid`, `female single` | Combined gender and marital status |
| `other_parties` | Categorical | `none`, `co-applicant`, `guarantor` | Presence of co-debtors or guarantors |
| `residence_since` | Numerical (Integer) | 1, 2, 3, 4 | Duration of present residence (in years) |
| `property_magnitude` | Categorical | `real estate`, `building society savings/life insurance`, `car`, `no known property` | Most valuable asset / property owned by borrower |
| `age` | Numerical (Integer) | 19 to 75 years | Age of the borrower in years |
| `other_payment_plans` | Categorical | `bank`, `stores`, `none` | Other installment / financing plans with other institutions |
| `housing` | Categorical | `rent`, `own`, `for free` | Housing accommodation type |
| `existing_credits` | Numerical (Integer) | 1, 2, 3, 4 | Number of existing credits at this bank |
| `job` | Categorical | `unemp/unskilled non res`, `unskilled resident`, `skilled`, `high qualif/emp/mgmt` | Employment qualification and employment status |
| `num_dependents` | Numerical (Integer) | 1, 2 | Number of people being financially maintained / liable |
| `own_telephone` | Categorical | `none`, `yes` | Registered telephone number in applicant's name |
| `foreign_worker` | Categorical | `yes`, `no` | Whether the applicant is a foreign worker |
| **`class` (Target)** | Categorical (Binary) | `good`, `bad` | Credit rating / risk classification (`good` = creditworthy, `bad` = default/high-risk) |

---

## 7. Target Variable
- **Target Column**: `class`
- **Classes**:
  - `good` (Creditworthy applicant, low risk of default) — **700 records (70.0%)**
  - `bad` (High risk / default applicant, non-creditworthy) — **300 records (30.0%)**
- **Binary Encoding for Modeling**:
  - `0`: `good` (Majority class)
  - `1`: `bad` (Minority / Positive default risk class)

---

## 8. Classification / Regression Problem
- **Problem Type**: Supervised Binary Classification (Credit Default Risk Scoring).
- **Asymmetric Cost Dynamics**: In financial credit underwriting, misclassifying a bad credit applicant as good (False Negative / Type II Error) incurs a severe economic loss (loss of principal balance), typically costed at 5× the opportunity loss of rejecting a good credit applicant (False Positive / Type I Error).

---

## 9. Reason for Selecting the Dataset
1. **Academic & Research Prestige**: The German Credit dataset is a gold-standard benchmark in machine learning and financial risk management literature.
2. **Rich Multi-Modal Feature Space**: Contains a balanced mixture of continuous numeric metrics (duration, credit amount, age), discrete ordinal attributes (installment rate, employment years, existing credits), and high-cardinality categorical features (credit history, loan purpose, property type).
3. **Realistic Asymmetry and Class Imbalance**: Reflects real-world credit dynamics with a 70:30 class distribution, requiring cost-sensitive evaluation (Precision-Recall trade-offs, ROC-AUC, Balanced Accuracy, F1-Score) beyond raw accuracy.
4. **Interpretability & Fair Lending Significance**: Provides a platform for feature importance attribution, permutation importance, odds-ratio diagnostics, and fairness/bias analysis across demographic subgroups.

---

## 10. Potential Research Questions
- **RQ1 (Feature Attribution)**: Which financial (e.g., checking liquidity, duration, credit amount) and demographic (age, housing, personal status) attributes exert the strongest statistical and predictive influence on credit default?
- **RQ2 (Comparative Model Performance)**: Which classification family (Linear/Logistic, Tree-based, Ensemble Random Forest/Gradient Boosting, Kernel SVM, Instance-based KNN, or Probabilistic Naive Bayes) achieves the optimal discriminatory power (ROC-AUC) and minority-class default identification (Recall/F1)?
- **RQ3 (Hyperparameter Optimization Gains)**: Does systematic hyperparameter tuning via GridSearchCV yield statistically significant performance gains over default baseline configurations?
- **RQ4 (Model Stability under Cross-Validation)**: How consistent are the models across Stratified 5-Fold Cross-Validation, and which model family exhibits minimal cross-fold variance?
- **RQ5 (Error Diagnosis & Misclassification Profiling)**: What specific applicant characteristics, loan sizes, or liquidity profiles characterize false-positive and false-negative errors in the top-performing model?
- **RQ6 (Linear vs. Non-linear Parsimony)**: Can a regularized, interpretable linear model (Logistic Regression) perform comparably to complex non-linear ensemble methods (Gradient Boosting, Random Forest)?

---

## 11. Dataset Limitations
1. **Historical Context**: The dataset was originally compiled in Germany (Deutsche Mark era), so nominal currency values (`credit_amount`) reflect historical monetary baselines.
2. **Sample Size**: While $N=1,000$ is sufficient for statistical hypothesis testing and 5-fold cross-validation, deep neural networks would be prone to overfitting; tree ensembles and kernel methods are appropriately suited.
3. **Compound Attributes**: Attributes like `personal_status` combine sex and marital status (`male single`, `female div/dep/mar`), requiring careful preprocessing and fairness considerations.
4. **Coarse Discretization**: Several features like `checking_status` and `savings_status` are pre-binned into ordinal ranges rather than continuous account balances.

---

## 12. Dataset Placement & Acquisition Instructions
- **Local Storage Path**: `data/raw/german_credit_data.csv`
- **Automated Ingestion**: The dataset can be re-fetched or loaded offline using `src/data_loader.py` which connects to OpenML (`data_id=31`) or falls back gracefully to `data/raw/german_credit_data.csv`.
