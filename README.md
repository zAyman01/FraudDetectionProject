# Fraud Detection Project

This project implements a comprehensive machine learning pipeline for detecting fraudulent transactions using the IEEE-CIS Fraud Detection dataset.

## Problem Statement

The goal is to predict whether a transaction is fraudulent (`isFraud` = 1) or legitimate (`isFraud` = 0). The dataset is characterized by extreme class imbalance and a high number of features, requiring careful preprocessing and dimensionality reduction.

## Pipeline Overview

1.  **EDA**: Initial exploration of transaction and identity data.
2.  **Preprocessing**: Data cleaning, handling missing values (median/mode imputation), log transformations, and label encoding.
3.  **Feature Reduction**: 
    - Dropping highly correlated features (>0.95).
    - **Advanced Consensus Selection**: Combining Mutual Information (MI) and Random Forest (RF) importance to retain the **top 50** most informative features.
    - **Optimized MI**: Using discrete feature masks for more accurate information gain on categorical variables.
    - **PCA Analysis**: Capture **95% variance** (~49 components) with loading analysis to understand component composition.
4.  **Model Training**: Training and comparing Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM across different feature sets (Full, Selected, PCA).
5.  **Evaluation**: Detailed performance analysis using Accuracy, AUC-ROC, Precision, Recall, and F1-score.

## Key Results

- **Feature Selection**: Retaining the top 50 features via MI provided a significant speedup with minimal impact on model performance.
- **Model Performance**: XGBoost and Random Forest consistently achieved the best results, with AUC-ROC scores exceeding 0.85.
- **Dimensionality Impact**: The project demonstrates that 50 carefully selected features can perform as well as the full set of 200+ features, offering better efficiency and maintainability.

## Project Structure

```text
FraudDetectionProject/
  data/
    raw/           # Original IEEE-CIS CSV files (Transaction + Identity)
    processed/     # Preprocessed (Full, Selected, PCA) feature sets
  models/          # Saved .pkl files for all trained models
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_feature_reduction.ipynb
    04_models.ipynb
    05_evaluation.ipynb
  results/
    figures/       # Comparison charts, ROC curves, and Confusion Matrices
    metrics/       # metrics_summary.csv containing all results
  report/          # Final documentation and reports
  README.md
  requirements.txt
```

## Team Workflow

- **Member 1**: Exploratory Data Analysis.
- **Member 2**: Data Preprocessing and Imputation.
- **Member 3**: Feature Selection and PCA.
- **Member 4**: Model Development and Training.
- **Member 5**: Final Evaluation and Report Generation.

## Technologies

- **Python 3.x**
- **Core Stack**: pandas, numpy, scikit-learn, XGBoost, LightGBM, matplotlib, seaborn, joblib.

## Setup

### 1) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run Each Stage

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
- Keep shared code in `src/`; avoid notebook-only logic for core functions.
