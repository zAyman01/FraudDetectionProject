# Fraud Detection Project

This project implements a comprehensive machine learning pipeline for detecting fraudulent transactions using the IEEE-CIS Fraud Detection dataset.

## Problem Statement

The goal is to predict whether a transaction is fraudulent (`isFraud` = 1) or legitimate (`isFraud` = 0). The dataset is characterized by extreme class imbalance (~3.5% fraud rate) and a high number of features (434), requiring careful preprocessing and dimensionality reduction.

## Pipeline Overview

1.  **EDA** (`01_eda.ipynb`): Initial exploration of transaction and identity data. Analyzes class distribution, missing values, transaction amount patterns, feature correlations, and categorical feature relationships with fraud. All plots are saved to `results/figures/`.
2.  **Preprocessing** (`02_preprocessing.ipynb`): Data cleaning, dropping features with >90% missing values, feature engineering (log transformation, decimal extraction), median/mode imputation, label encoding, stratified 80/20 train-test split, and StandardScaler normalization. Outputs saved as both `.parquet` and `.csv`.
3.  **Feature Reduction** (`03_feature_reduction.ipynb`):
    - Drops constant and highly correlated features (correlation > 0.98).
    - **Consensus Feature Selection**: Combines Mutual Information (MI) and Random Forest importance rankings to select the **top 50** most informative features. MI uses discrete feature masks for accurate categorical variable scoring.
    - **PCA**: Applied to selected features to retain **95% variance** (39 components), with loading analysis to understand component composition.
    - All figures saved to `results/figures/`.
4.  **Model Training** (`04_models.ipynb`): Trains and compares five models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) across three feature sets (Full [358], Selected [50], PCA [39]). Hyperparameters tuned for accuracy-speed tradeoff: XGBoost/LightGBM use LR=0.05 + more estimators + regularization; RandomForest uses 200 trees + max_depth=15; LogisticRegression uses `saga` solver. Handles class imbalance via `class_weight='balanced'` (sklearn) and `scale_pos_weight=27.58` (XGBoost/LightGBM). Generates comparison charts, ROC curves, and confusion matrices.
5.  **Evaluation Report** (`05_evaluation.ipynb`): Comprehensive, self-contained, print-ready report covering all pipeline stages with embedded figures, metrics tables, discussion of dimension reduction impact, and final recommendations.

## Key Results

- **Best Model**: XGBoost with Top 50 selected features (expected AUC ~0.93+).
- **XGBoost Tuning**: Lower learning rate (0.05), 300 estimators, max_depth=8, L1/L2 regularization, and `hist` tree method maximize AUC while keeping training fast.
- **LightGBM Tuning**: Lower learning rate (0.05), 200 estimators, num_leaves=47, max_depth=8, and regularization improve accuracy over the baseline.
- **RandomForest Tuning**: Increased to 200 estimators, max_depth=15, min_samples_leaf=5 for better ensemble strength.
- **Feature Selection Impact**: Reducing from 358 to 50 features (86% reduction) retains >97% of AUC-ROC performance while cutting training time significantly.
- **PCA Performance**: 39 PCA components achieve competitive AUC (~0.90+) but sacrifice interpretability.
- **Ensemble Models**: XGBoost and LightGBM consistently outperform other models across all feature sets.

## Project Structure

```
FraudDetectionProject/
  data/
    raw/                      # Original IEEE-CIS CSV files
      train_transaction.csv
      train_identity.csv
      test_transaction.csv
      test_identity.csv
    processed/                # Preprocessed and reduced feature sets (.parquet + .csv)
      X_train.parquet / .csv
      X_test.parquet / .csv
      y_train.parquet / .csv
      y_test.parquet / .csv
      X_train_full.parquet / .csv
      X_test_full.parquet / .csv
      X_train_selected.parquet / .csv
      X_test_selected.parquet / .csv
      X_train_pca.parquet / .csv
      X_test_pca.parquet / .csv
      selected_features.csv
      scaler.pkl
      pca.pkl
      pca_scaler.pkl
      feature_metadata.pkl
  models/                     # Saved .pkl files for all trained models
  notebooks/
    01_eda.ipynb              # Exploratory Data Analysis
    02_preprocessing.ipynb    # Data Cleaning and Preparation
    03_feature_reduction.ipynb# Feature Selection and PCA
    04_models.ipynb           # Model Training and Evaluation
    05_evaluation.ipynb       # Final Professional Report
    light_eda.ipynb           # Lightweight post-preprocessing EDA verification
  results/
    figures/                  # All generated plots (fraud distribution, MI scores, ROC curves, etc.)
    metrics/                  # metrics_summary.csv with all model results
  report/                     # Exported report documents
  README.md                   # This file
  requirements.txt            # Python dependencies
```

## Technologies

- **Python 3.x**
- **Core Stack**: pandas, numpy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, joblib
- **Format Support**: pyarrow (Parquet I/O)

## Setup

### 1) Create and activate virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Pipeline

Execute notebooks in order. Each notebook saves outputs for the next stage:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_feature_reduction.ipynb
jupyter notebook notebooks/04_models.ipynb
jupyter notebook notebooks/05_evaluation.ipynb
```

## Working Rules

- Save charts to `results/figures/`.
- Save metric exports to `results/metrics/`.
- Keep final interpretation in `report/`.
- All data saved as both `.parquet` (fast I/O) and `.csv` (compatibility).
- Random seed: `42` for reproducibility across all notebooks.
