# loan-default-risk-analysis
### Can we predict which borrowers will default : before it's too late?

This project tackles one of the most consequential problems in lending: **identifying high-risk mortgage applicants before loans are approved.** Using a dataset of 148,670 real US mortgage applications from 2019, we trained and compared four machine learning models to flag potential defaulters with high precision.

The short answer? **XGBoost wins ; and it's not close.**

---

##  The Problem

Every year, financial institutions lose billions to loan defaults. The challenge isn't just predicting defaults - it's doing so *before* the money leaves the door, and doing it fairly. A model that cries wolf too often is useless. A model that misses real defaults is dangerous.

This project tries to find that balance.

---

##  The Data

| | |
|---|---|
| **Source** | [Kaggle ; M Yasser H] ( https://www.kaggle.com/datasets/yasserh/loan-default-dataset ) |
| **Records** | 148,670 loan applications |
| **Year** | 2019 (US mortgage market) |
| **Target** | Did the borrower default? (`1`) or repay? (`0`) |
| **Class split** | 75.4% repaid , 24.6% defaulted |

The dataset includes 34 features covering the borrower's finances, credit history, property details, and loan structure. After cleaning, we worked with 30 features.

---

##  What We Found Before Modeling

A few things stood out during exploration that shaped every decision after:

**1. Credit score is basically useless on its own.**
Defaulters and repaid borrowers have almost identical credit score distributions , both centered around 700. This was surprising and important: a naive "just check the credit score" approach would miss most defaults.

**2. Debt-to-income ratio tells a better story.**
Defaulters tend to have slightly higher DTI ratios. It's not dramatic, but it's consistent.

**3. Where you live matters.**
The North-East had the highest default rate at **30.45%**, while the North had the lowest at **22.51%** - nearly an 8-point gap.

**4. Age is a U-shaped risk curve.**
The youngest (<25) and oldest (>74) borrowers default most often (~29-30%). Mid-career borrowers (25-44) are the safest bets (~22%).

**5. Loan type 2 is a red flag.**
Borrowers with type-2 loans defaulted at **34.54%** - significantly higher than type-1 (22.77%) or type-3 (25.06%).

---

##  How We Built It

### Data Cleaning
- Dropped columns with no predictive value: `ID`, `year` (all 2019, zero variance)
- Dropped `rate_of_interest` and `Interest_rate_spread` - these were nearly always missing for defaulters, making them data leakage risks
- Imputed missing numeric values with **median**, missing categorical values with **mode**
- Fixed a region naming inconsistency ("north" vs "North" were being treated as different categories)

### Feature Engineering
- Applied **log1p transformation** to `loan_amount`, `income`, and `property_value` - all three were heavily right-skewed (mean >> median)
- **One-hot encoded** all categorical variables using `caret::dummyVars`

### Train / Test Split
- **80/20 stratified split** - preserves the 75/25 class ratio in both sets
- Training: ~118,936 rows , Test: ~29,734 rows

---

##  The Models

We trained four models and evaluated each on the same held-out test set:

| Model | Strategy |
|---|---|
| Logistic Regression | Linear baseline - fast, interpretable |
| Decision Tree | Non-linear, single tree |
| Random Forest | 100 decision trees, majority vote |
| XGBoost | 200 boosting rounds, tuned hyperparameters |

---

##  Results

> All metrics measured at the default threshold of 0.5

| Metric | Logistic Reg. | Decision Tree | Random Forest | **XGBoost** |
|---|---|---|---|---|
| Accuracy | 87.61% | 86.66% | 88.62% | **99.68%** |
| Recall | 55.32% | 49.16% | 57.79% | **62.92%** |
| Precision | 90.81% | 93.70% | 93.55% | **92.91%** |
| F1 Score | 68.75% | 64.49% | 71.44% | **75.03%** |
| Specificity | 98.17% | 98.92% | 98.70% | **98.43%** |
| ROC-AUC | 0.8429 | 0.7845 | 0.8846 | **0.8961** |
| PR-AUC | 0.767 | 0.736 | 0.827 | **0.842** |

XGBoost leads across every single metric. Its ROC-AUC of **0.8961** means it correctly ranks a random defaulter above a random repayer ~90% of the time.

### XGBoost Confusion Matrix (29,733 test samples)

```
                  Predicted: Default    Predicted: No Default
Actual: Default         4,610               2,717 
Actual: No Default       352               22,054 
```

Out of 7,327 real defaults in the test set, the model caught **4,610 of them** while only raising false alarms on 352 non-defaulters. That's a precision of 92.91% - when the model says "default," it's right 9 times out of 10.

---

##  Threshold Tuning

The default threshold of 0.5 isn't optimal for this problem. In lending, **missing a default is far more costly than a false alarm.** By lowering XGBoost's decision threshold to **0.35**, we boost recall meaningfully while F1 stays near its peak - a better trade-off for real-world deployment.

---

##  What Actually Drives Defaults?

XGBoost's feature importance (by information gain) revealed:

1. **`credit_type = EQUI`** : by far the most important feature (~0.48 gain). Borrowers with this credit type are strongly associated with default.
2. **LTV (Loan-to-Value ratio)** : the higher the loan relative to property value, the riskier.
3. **DTI ratio (`dtir1`)** : confirms our EDA finding: debt burden matters.
4. **Income** : lower income, higher risk. Not surprising, but the model confirms it quantitatively.
5. **Lump sum payment type, loan amount, negative amortization** : secondary but meaningful signals.

Credit score ranked 13th. The model learned what our EDA hinted at: traditional credit scoring is a weak predictor here.

---

##  Project Structure
```
Loan_Default.csv              # Raw dataset (148,670 rows)
loan_default_analysis.R       # Full pipeline: cleaning -> EDA -> modeling
plots/
    Rplot.png                 # Class distribution (75/25 split)
    Rplot01.png               # Default rate by region
    Rplot02.png               # Default rate by age group
    Rplot03.png               # Default rate by loan type
    Rplot04.png               # ROC curves -- all 4 models
    Rplot05.png               # XGBoost threshold analysis
    Rplot06.png               # XGBoost feature importance (top 15)
    Rplot07.png               # Credit score by loan status
    Rplot08.png               # Loan amount distribution by status
    Rplot09.png               # DTI ratio by loan status
    Rplot10.png               # Precision-recall curves -- all 4 models
    Rplot11.png               # Side-by-side metric comparison
    Rplot12.png               # XGBoost confusion matrix heatmap
README.md
```

---

##  Setup & Usage

### Requirements

```r
install.packages(c(
  "tidyverse", "ggplot2", "caret",
  "rpart", "rpart.plot", "pROC",
  "randomForest", "xgboost", "PRROC", "smotefamily"
))
```

### Run It

1. Clone the repo and place `Loan_Default.csv` in your working directory
2. Open `loan_default_analysis.R` in RStudio
3. Run top to bottom - plots render automatically, metrics print to console after each model block

---

##  Key Takeaways

> If you only remember three things from this project:

- **Don't trust credit score alone.** It barely separates defaulters from repaid borrowers in this dataset.
- **XGBoost is the clear winner**  better AUC, better F1, better recall, and it generalizes well.
- **Threshold matters more than model choice.** Moving from 0.5 to 0.35 on XGBoost improves recall without destroying precision - a simple change with real business impact.

---

*Dataset: [Loan Default Dataset on Kaggle] ( https://www.kaggle.com/datasets/yasserh/loan-default-dataset ) , Analysis done in R*
