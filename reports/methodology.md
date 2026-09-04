# Research Methodology: Machine Learning Public Dataset Study

## 1. Research Framework Overview
This study presents an academic, empirical machine learning research investigation conducted on the **UCI German Credit Risk Benchmark Dataset** ($N=1,000$, 20 predictor features). The overarching objective is to rigorously evaluate predictive capacity, feature attribution, cross-validation stability, hyperparameter sensitivity, and asymmetric misclassification cost dynamics across 7 distinct algorithm families.

```mermaid
flowchart LR
    A["Raw Dataset\n(N=1000, 20 Features)"] --> B["Stratified Split\n(80% Train, 20% Test)"]
    B --> C["Domain Feature\nEngineering"]
    C --> D["Scikit-Learn Pipeline\n(Impute, Scale, OHE)"]
    D --> E["Cross-Validation\n(Stratified 5-Fold)"]
    E --> F["Benchmark 7 Models"]
    F --> G["GridSearchCV\nHyperparameter Tuning"]
    G --> H["Multi-Criteria\nChampion Selection"]
    H --> I["Model Interpretation\n& Error Profiling"]
    I --> J["Research Dashboard\n& Findings"]
```

---

## 2. Data Preprocessing & Leakage Prevention Protocol
To ensure uncompromising scientific validity and eliminate data leakage:
1. **Partitioning Strategy**: A **Stratified 80:20 Train/Test Split** ($N_{train}=800, N_{test}=200$) was applied directly to the raw dataset prior to calculating any feature-level statistics, encodings, or transformations.
2. **ColumnTransformer Architecture**:
   - **Numerical Pipeline**: Missing value imputation via median statistics ($SimpleImputer(strategy='median')$) followed by Z-score standardization ($StandardScaler()$ with $\mu=0, \sigma=1$).
   - **Categorical Pipeline**: Missing value imputation via modal statistics ($SimpleImputer(strategy='most\_frequent')$) followed by sparse-free One-Hot Encoding ($OneHotEncoder(handle\_unknown='ignore')$).
3. **Strict Parameter Encapsulation**: All scaling parameters ($\mu, \sigma$), imputation medians, and categorical level mappings were learned strictly from $N_{train}=800$ and transformed out-of-sample onto $N_{test}=200$.

---

## 3. Domain Feature Engineering Rationale

| Feature Name | Mathematical Definition | Empirical & Economic Rationale |
| :--- | :--- | :--- |
| **`credit_to_duration_ratio`** | $\frac{\text{Credit Amount}}{\text{Duration}}$ | Estimates the monthly capital amortization burden. A high ratio compresses the borrower's disposable monthly cash flow, significantly escalating immediate default risk. |
| **`credit_to_age_ratio`** | $\frac{\text{Credit Amount}}{\text{Age}}$ | Quantifies debt leverage relative to career/earnings maturity and lifecycle wealth accumulation. Younger borrowers with excessive credit exposure exhibit elevated vulnerability. |
| **`liquidity_index`** | $\text{Rank}(\text{checking}) + \text{Rank}(\text{savings})$ | Aggregates multi-tier depository reserves into a composite ordinal liquidity rating ($0$ to $5$), capturing total liquid solvency. |
| **`high_risk_purpose`** | $\mathbb{I}(\text{purpose} \in \{\text{edu, vac, bus, repair}\})$ | Isolates discretionary, speculative, or uncollateralized loan purposes from asset-backed borrowing (e.g., automobile or home electronics). |
| **`log_credit_amount`** | $\ln(1 + \text{Credit Amount})$ | Normalizes severe right-skewness and long-tail variance in loan sizes, improving linear and distance-based model convergence. |
| **`log_duration`** | $\ln(1 + \text{Duration})$ | Compresses loan duration distribution to stabilize gradient descent and kernel margin optimization. |

---

## 4. Machine Learning Algorithm Formulations

### 4.1. Logistic Regression (Regularized Linear Baseline)
Models the posterior probability of default using the logistic sigmoid function:
$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
Optimized via binary cross-entropy loss with $L_2$ Tikhonov regularization and class weighting:
$$\min_{\mathbf{w}, b} \left[ -\sum_{i=1}^{N} w_{y_i} \left( y_i \ln \hat{y}_i + (1-y_i) \ln(1-\hat{y}_i) \right) + \frac{1}{2C} \|\mathbf{w}\|_2^2 \right]$$

### 4.2. Decision Tree Classifier
Recursively partitions feature space by maximizing Gini impurity reduction:
$$\Delta I_G(S, A) = I_G(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} I_G(S_v), \quad \text{where } I_G(S) = 1 - \sum_{k=0}^{1} p_k^2$$

### 4.3. Random Forest (Ensemble Bagging)
Aggregates $B$ decorrelated bootstrap decision trees:
$$\hat{f}_{RF}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(\mathbf{x})$$
Random feature sub-spacing ($\sqrt{p}$) ensures inter-tree decorrelation and variance reduction:
$$\text{Var}(\hat{f}_{RF}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2 \xrightarrow{B \to \infty} \rho \sigma^2$$

### 4.4. Support Vector Machine (Kernel SVM)
Constructs maximum-margin hyperplanes in a reproducing kernel Hilbert space using the Radial Basis Function (RBF) kernel:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
Dual formulation solved via Sequential Minimal Optimization (SMO):
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t. } 0 \le \alpha_i \le C, \, \sum_{i=1}^N \alpha_i y_i = 0$$

### 4.5. K-Nearest Neighbors (KNN)
Predicts class membership based on distance-weighted voting across the $k$-nearest neighbors:
$$P(y=1|\mathbf{x}) = \frac{\sum_{i \in N_k(\mathbf{x})} w_i \cdot \mathbb{I}(y_i = 1)}{\sum_{i \in N_k(\mathbf{x})} w_i}, \quad w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$$

### 4.6. Gaussian Naive Bayes
Applies Bayes' theorem under the conditional feature independence assumption:
$$P(y=k|\mathbf{x}) \propto P(y=k) \prod_{j=1}^D \mathcal{N}(x_j; \mu_{kj}, \sigma_{kj}^2 + \epsilon)$$

### 4.7. Gradient Boosting (Tree Ensemble Boosting)
Sequentially fits shallow regression trees to the pseudo-residuals of the cross-entropy loss function:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{I}(\mathbf{x} \in R_{jm})$$
where $\eta \in (0, 1]$ represents the learning rate (shrinkage parameter).

---

## 5. Model Evaluation and Multi-Criteria Champion Selection

### 5.1. Evaluation Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (Positive Predictive Value)**: $\frac{TP}{TP + FP}$
- **Recall (Sensitivity / Default Detection)**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC**: Integral of True Positive Rate vs. False Positive Rate across all decision thresholds.
- **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (P_i - y_i)^2$

### 5.2. Stratified 5-Fold Cross-Validation Protocol
Stratified 5-Fold Cross-Validation preserves class balance ($70:30$) across each partition:
$$\overline{\text{Metric}}_{CV} = \frac{1}{5} \sum_{k=1}^5 \text{Metric}_k, \quad \sigma_{CV} = \sqrt{\frac{1}{4}\sum_{k=1}^5 (\text{Metric}_k - \overline{\text{Metric}}_{CV})^2}$$

### 5.3. Composite Multi-Criteria Selection Function
The champion model is selected through a multi-dimensional objective balancing discriminatory power, minority recall, and stability:
$$\mathcal{S}_{\text{model}} = 0.50 \cdot \text{ROC-AUC} + 0.35 \cdot \text{F1-Score} + 0.15 \cdot (1.0 - \sigma_{\text{CV-ROC}})$$
