"""
Phase 3a: Train LightGBM Model
Paper: arXiv 2501.18071v2 — Diabetes Prediction with ML + XAI

Expected results (from paper):
  Accuracy: 92.03%
  ROC-AUC:  0.97

Usage:
  python3 train_lgbm.py
"""

import os
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, precision_recall_curve
)
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = "outputs"

def load_arrays():
    print("[1/4] Loading preprocessed arrays...")
    data = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        data[name] = np.load(os.path.join(OUTPUTS_DIR, f"{name}.npy"))
        print(f"      {name}: {data[name].shape}")

    feature_names = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "feature_names.csv"), header=None
    )[0].tolist()
    print(f"      Features: {feature_names}")
    return data, feature_names

def train(data):
    """
    LightGBM hyperparameters — tuned for recall on imbalanced diabetes data.

    Key parameters explained:
      n_estimators    : number of trees to build (more = better but slower)
      learning_rate   : how much each tree corrects the previous (smaller = more careful)
      num_leaves      : max leaves per tree — controls model complexity
      max_depth       : limits how deep each tree grows (prevents overfitting)
      min_child_samples: minimum rows needed to form a leaf (smooths out noise)
      subsample       : fraction of rows sampled per tree (adds randomness, prevents overfitting)
      colsample_bytree: fraction of features sampled per tree (same idea)
      reg_alpha/lambda: L1/L2 regularization — reduced to allow more complex fits
      n_jobs          : use all CPU cores
      random_state    : for reproducibility

    Note: class_weight removed — SMOTE already balanced training to 50/50,
    so adding 'balanced' would double-penalise the majority class.
    Early stopping now monitors AUC (more meaningful than logloss here).
    """
    import lightgbm as lgb
    print("\n[2/4] Training LightGBM...")
    model = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=127,        # more leaves → richer trees
        max_depth=-1,
        min_child_samples=10,  # allow splits on smaller groups
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.05,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    start = time.time()
    model.fit(
        data["X_train"], data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        eval_metric="auc",     # early stopping on AUC instead of logloss
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    elapsed = time.time() - start
    print(f"      Training time: {elapsed:.1f}s")
    print(f"      Best iteration: {model.best_iteration_}")
    return model

def best_threshold(y_true, y_prob):
    """Find threshold on val set that maximises F1 for the diabetic class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    idx = f1s.argmax()
    return thresholds[idx], f1s[idx]

def evaluate(model, data, feature_names):
    print("\n[3/4] Evaluating on val and test sets...")

    # Find optimal threshold on validation set
    val_prob  = model.predict_proba(data["X_val"])[:, 1]
    thresh, _ = best_threshold(data["y_val"], val_prob)
    print(f"\n  Optimal threshold (val F1): {thresh:.3f}")

    for split_name, X, y in [
        ("Validation", data["X_val"], data["y_val"]),
        ("Test",       data["X_test"], data["y_test"])
    ]:
        y_prob      = model.predict_proba(X)[:, 1]
        auc         = roc_auc_score(y, y_prob)

        for label, y_pred in [
            ("default (0.50)", (y_prob >= 0.5).astype(int)),
            (f"tuned  ({thresh:.2f})", (y_prob >= thresh).astype(int)),
        ]:
            acc = accuracy_score(y, y_pred) * 100
            f1  = f1_score(y, y_pred)
            print(f"\n  ── {split_name} | threshold {label} ──")
            print(f"  Accuracy : {acc:.2f}%   (paper: 92.03%)")
            print(f"  ROC-AUC  : {auc:.4f}    (paper: 0.97)")
            print(f"  F1 Score : {f1:.4f}")
            print(f"\n  Classification Report:")
            print(classification_report(y, y_pred, target_names=["No Diabetes", "Diabetes"]))
            cm = confusion_matrix(y, y_pred)
            print(f"  Confusion Matrix:")
            print(f"              Predicted No  Predicted Yes")
            print(f"  Actual No       {cm[0,0]:6d}        {cm[0,1]:6d}")
            print(f"  Actual Yes      {cm[1,0]:6d}        {cm[1,1]:6d}")

def save_model(model):
    print("\n[4/4] Saving model...")
    path = os.path.join(OUTPUTS_DIR, "lgbm_model.pkl")
    joblib.dump(model, path)
    print(f"      Saved to {path}")

def main():
    print("=" * 60)
    print("  Phase 3a: LightGBM Training")
    print("=" * 60)

    data, feature_names = load_arrays()
    model = train(data)
    evaluate(model, data, feature_names)
    save_model(model)

    print("\n✅ LightGBM training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
