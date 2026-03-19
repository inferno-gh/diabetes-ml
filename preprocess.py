"""
Phase 2: Preprocessing Pipeline
Paper: arXiv 2501.18071v2 — Diabetes Prediction with ML + XAI

Steps:
  1. Load raw data
  2. Median imputation (handles any NaN values)
  3. Stratified 70/15/15 split (train / val / test)
  4. SMOTE on train split ONLY
  5. StandardScaler fit on train, applied to all splits
  6. Save processed arrays to outputs/

Usage (local):
  python preprocess.py
  python preprocess.py --data path/to/your/diabetes.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_COL   = "Diabetes_binary"
RANDOM_STATE = 42
TEST_SIZE    = 0.15   # 15% test
VAL_SIZE     = 0.15   # 15% val  (from remaining 85%)
# After first split: 85% train+val, 15% test
# Val fraction of train+val = 0.15 / 0.85 ≈ 0.1765
VAL_FRAC_OF_TRAINVAL = VAL_SIZE / (1 - TEST_SIZE)

def load_data(path: str) -> pd.DataFrame:
    print(f"[1/5] Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}  |  Columns: {len(df.columns)}")
    print(f"      Missing values: {df.isnull().sum().sum()}")
    print(f"      Class distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(4)}")
    return df

def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation — matches paper's preprocessing."""
    print("[2/5] Median imputation...")
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns
    )
    print(f"      Missing values after imputation: {df_imputed.isnull().sum().sum()}")
    return df_imputed

def split(df: pd.DataFrame):
    """Stratified 70/15/15 split."""
    print("[3/5] Stratified 70/15/15 split...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    # Step 1: carve out 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    # Step 2: carve out 15% val from the remaining 85%
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=VAL_FRAC_OF_TRAINVAL,
        stratify=y_trainval,
        random_state=RANDOM_STATE
    )

    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        pct = ys.mean() * 100
        print(f"      {name:5s}: {len(ys):6d} rows  |  diabetic: {pct:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train):
    """
    SMOTE only on the training set.
    Key rule: NEVER apply SMOTE to val/test — that would contaminate evaluation.
    """
    print("[4/5] Applying SMOTE to training set only...")
    print(f"      Before — class 0: {(y_train==0).sum()}  class 1: {(y_train==1).sum()}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"      After  — class 0: {(y_train_res==0).sum()}  class 1: {(y_train_res==1).sum()}")
    print(f"      Train size grew: {len(y_train)} → {len(y_train_res)}")
    return X_train_res, y_train_res

def scale(X_train, X_val, X_test):
    """
    Fit StandardScaler on TRAIN only, then transform all splits.
    Fitting on val/test would be data leakage.
    """
    print("[5/5] Scaling features (StandardScaler fit on train)...")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    print(f"      Train mean (first 3 cols): {X_train_sc[:, :3].mean(axis=0).round(4)}")
    print(f"      Train std  (first 3 cols): {X_train_sc[:, :3].std(axis=0).round(4)}")
    return X_train_sc, X_val_sc, X_test_sc, scaler

def save_outputs(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, feature_names):
    os.makedirs(output_dir, exist_ok=True)
    arrays = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, arr in arrays.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"      Saved {path}  shape={arr.shape}")

    # Also save feature names for later use in XAI
    pd.Series(feature_names).to_csv(
        os.path.join(output_dir, "feature_names.csv"), index=False, header=False
    )
    print(f"      Saved feature_names.csv ({len(feature_names)} features)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(__file__),
            "data", "diabetes_binary_health_indicators_BRFSS2015.csv"
        ),
        help="Path to raw CSV file"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "outputs"),
        help="Directory to save processed arrays"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 2: Preprocessing Pipeline")
    print("=" * 60)

    df            = load_data(args.data)
    df            = impute(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split(df)
    X_train, y_train = apply_smote(X_train, y_train)
    X_train, X_val, X_test, scaler = scale(X_train, X_val, X_test)

    feature_names = [c for c in df.columns if c != TARGET_COL]
    save_outputs(args.output_dir, X_train, X_val, X_test,
                 y_train.values, y_val.values, y_test.values,
                 feature_names)

    print("\n✅ Preprocessing complete!")
    print(f"   Final shapes:")
    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print("=" * 60)

if __name__ == "__main__":
    main()