# Diabetes ML

Replication of an ML paper on diabetes prediction using classical and ensemble machine learning methods.

## Goal

This project replicates and extends research on predicting diabetes onset from clinical and demographic features. It covers:

- Data preprocessing and class imbalance handling (SMOTE via imbalanced-learn)
- Model training: Logistic Regression, Random Forest, XGBoost, LightGBM
- Evaluation: accuracy, AUC-ROC, precision, recall, F1
- Explainability: SHAP, LIME, and InterpretML for feature importance and model transparency

## Structure

```
diabetes-ml/
├── data/          # Raw and processed datasets (CSV files are gitignored)
├── notebooks/     # Exploratory analysis and experiment notebooks
├── src/           # Reusable Python modules (preprocessing, training, evaluation)
├── outputs/       # Saved models, figures, and metrics (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Place your dataset CSV in `data/`, then run notebooks in `notebooks/` for end-to-end experiments.
