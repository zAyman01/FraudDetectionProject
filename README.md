# Fraud Detection Project

A starter machine learning repository for fraud detection using binary classification (`isFraud`).

## Problem Statement

This project predicts whether a transaction is fraudulent:

- `0` = legitimate transaction
- `1` = fraudulent transaction

The dataset is expected to be large and imbalanced, so the workflow focuses on reproducible preprocessing, feature reduction, robust modeling, and clear evaluation.

## Dataset Notes (Large Files)

The team is working with more than **1.2 GB** of data. To keep the repository fast and manageable:

- All files inside `data/raw/` and `data/processed/` are gitignored.
- Keep data files local or in shared storage (shared drive, cloud bucket, or internal data platform).
- Do not commit CSV/Parquet/model dump files to git.
- Only commit code, notebooks, documentation, and lightweight metadata.

## Project Structure

```text
FraudDetectionProject/
  data/
    raw/
      .gitkeep
    processed/
      .gitkeep
  notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_feature_reduction.ipynb
    04_models.ipynb
    05_evaluation.ipynb
  src/
    preprocessing.py
    features.py
    models.py
    evaluation.py
  results/
    figures/
    metrics/
  report/
  .gitignore
  README.md
  requirements.txt
```

## Team Workflow (5 Members)

Recommended stage order:

1. EDA
2. Preprocessing
3. Feature Reduction (feature selection + PCA)
4. Model Training
5. Evaluation

Suggested parallel ownership:

- Member 1: `notebooks/01_eda.ipynb`
- Member 2: `src/preprocessing.py` and `notebooks/02_preprocessing.ipynb`
- Member 3: `src/features.py` and `notebooks/03_feature_reduction.ipynb`
- Member 4: `src/models.py` and `notebooks/04_models.ipynb`
- Member 5: `src/evaluation.py`, `notebooks/05_evaluation.ipynb`, and `report/`

## Technologies

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- jupyter

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
