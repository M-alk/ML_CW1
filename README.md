# CW1 Regression Coursework

This repository contains my pipeline for predicting `outcome` on the CW1 dataset.

## Project structure

- `data/` - training and test CSV files
- `notebooks/` - analysis and model development
  - `01_eda.ipynb`
  - `02_baseline.ipynb`
  - `03_tree_models.ipynb`
- `src/train_model.py` - final training + prediction script
- `outputs/` - submission files
- `evaluate/CW1_eval_script.py` - provided baseline template 

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce final predictions

Run from project root:

```bash
python src/train_model.py
```

This writes:

- `outputs/CW1_submission_k22056537.csv`

Submission format is a single column named `yhat` with one prediction per test row.

## Notebook order

1. `notebooks/01_eda.ipynb` - EDA and feature checks
2. `notebooks/02_baseline.ipynb` - baseline models
3. `notebooks/03_tree_models.ipynb` - tree models, tuning, and final model selection
