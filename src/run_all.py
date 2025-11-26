# src/run_all.py
"""
Improved run_all pipeline for Life Expectancy project.

Features added:
 - Canonicalize column names (lowercase, underscores)
 - Save cleaned CSV with canonical headers
 - Group-aware CV (GroupKFold) when 'country' present
 - Save model pipeline (preprocessor + RF) as joblib
 - Save model metadata (raw_feature_names, preprocessed_feature_names when available)
 - Generate and save visuals: distribution, trend, corr heatmap, feature importance
"""
import os
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import sys

# --- config paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "Life_expectancy_raw.csv"
DATA_CLEAN = ROOT / "data" / "cleaned" / "Life_expectancy_clean.csv"
VIS_DIR = ROOT / "visuals"
MODELS_DIR = ROOT / "models"

DROP_COUNTRIES = ["Vanuatu", "Tonga", "Togo", "Cabo Verde"]
TARGET_GUESS = "life expectancy"  # lowercase canonical guess

# create directories if not exist
for p in [DATA_CLEAN.parent, VIS_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def canonicalize_colname(s: str) -> str:
    """Lowercase, collapse whitespace, replace spaces/hyphens with underscores."""
    if not isinstance(s, str):
        return s
    s2 = re.sub(r'\s+', ' ', s).strip()
    s2 = s2.replace('-', ' ')
    s2 = s2.strip()
    s2 = s2.replace(' ', '_')
    return s2.lower()


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: dataset not found at: {path}")
        print("Put your CSV into:", path)
        sys.exit(1)
    df = pd.read_csv(path)
    # canonicalize column names
    df.columns = [canonicalize_colname(c) if isinstance(c, str) else c for c in df.columns]
    return df


def find_target_column(df: pd.DataFrame):
    # look for canonical forms containing 'life' and 'expect'
    for c in df.columns:
        try:
            if isinstance(c, str) and 'life' in c and 'expect' in c:
                return c
        except Exception:
            continue
    return None


def basic_clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # drop duplicates
    df = df.drop_duplicates().copy()

    # drop problematic countries if present (match original names in 'country' column)
    if 'country' in df.columns:
        present = [c for c in DROP_COUNTRIES if c.lower() in df['country'].astype(str).str.lower().unique()]
        if present:
            print("Dropping countries due to missingness:", present)
            mask = ~df['country'].astype(str).str.lower().isin([p.lower() for p in present])
            df = df[mask].copy()

    # ensure year is int
    if 'year' in df.columns:
        try:
            df['year'] = df['year'].astype(int)
        except Exception:
            pass

    # coerce target to numeric if needed
    if target_col and target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    # numeric imputation (median)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns detected:", num_cols)
    if num_cols:
        num_imp = SimpleImputer(strategy='median')
        df[num_cols] = num_imp.fit_transform(df[num_cols])

    # categorical impute (mode)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for c in cat_cols:
        if df[c].isnull().any():
            mode_val = df[c].mode().iloc[0] if not df[c].mode().empty else "Unknown"
            df[c] = df[c].fillna(mode_val)

    return df


def save_clean(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print("Saved cleaned data to:", path)


def run_eda(df: pd.DataFrame, target_col: str):
    tgt = target_col
    sns.set(style="whitegrid")

    # distribution
    try:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[tgt].dropna(), bins=30, kde=True)
        plt.title("Distribusi Life Expectancy")
        out1 = VIS_DIR / "dist_life_expectancy.png"
        plt.savefig(out1, bbox_inches='tight')
        plt.close()
        print("Saved:", out1)
    except Exception as e:
        print("Could not save distribution plot:", e)

    # trend by year
    try:
        if 'year' in df.columns:
            dfg = df.groupby('year')[tgt].mean().reset_index()
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=dfg, x='year', y=tgt, marker='o')
            plt.title("Rata-rata Life Expectancy per Tahun")
            out2 = VIS_DIR / "trend_life_expectancy.png"
            plt.savefig(out2, bbox_inches='tight')
            plt.close()
            print("Saved:", out2)
    except Exception as e:
        print("Could not save trend plot:", e)

    # correlation heatmap (numeric)
    try:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            plt.figure(figsize=(12, 10))
            sns.heatmap(num.corr(), cmap="RdBu_r", center=0)
            out3 = VIS_DIR / "corr_heatmap.png"
            plt.savefig(out3, bbox_inches='tight')
            plt.close()
            print("Saved:", out3)
    except Exception as e:
        print("Could not save corr heatmap:", e)


def train_model(df: pd.DataFrame, target_col: str):
    tgt = target_col
    # prepare X, y
    X = df.drop(columns=[tgt], errors='ignore')
    # drop country from features to avoid leakage
    if 'country' in X.columns:
        X = X.drop(columns=['country'])
    y = df[tgt]

    # define numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print("Training features - numeric:", num_cols, "categorical:", cat_cols)

    # save raw feature names (before preprocessing)
    raw_feature_names = list(X.columns)

    # pipelines
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]) if num_cols else None
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]) if cat_cols else None

    transformers = []
    if num_pipe:
        transformers.append(('num', num_pipe, num_cols))
    if cat_pipe:
        transformers.append(('cat', cat_pipe, cat_cols))

    pre = ColumnTransformer(transformers) if transformers else None

    steps = []
    if pre is not None:
        steps.append(('pre', pre))
    steps.append(('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)))
    model = Pipeline(steps)

    # if country present in original df, try group-aware CV
    if 'country' in df.columns:
        try:
            groups = df['country']
            gkf = GroupKFold(n_splits=5)
            print("Running GroupKFold cross-validation (by country)...")
            scores = -1 * cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='neg_mean_absolute_error', n_jobs=-1)
            print("Group CV MAE per fold:", scores)
            print("Group CV MAE mean:", scores.mean())
        except Exception as e:
            print("Group CV failed or not applicable, continuing to train. Reason:", e)

    # final holdout split (stratify not used)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Model results -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # save model pipeline
    model_path = MODELS_DIR / "rf_life_expectancy.joblib"
    joblib.dump(model, model_path)
    print("Saved model to:", model_path)

    # save metadata (raw_feature_names + preprocessed_feature_names if available)
    meta = {"raw_feature_names": raw_feature_names}
    try:
        if pre is not None:
            # Attempt to get transformed feature names (sklearn 1.0+)
            try:
                pre_feature_names = list(pre.get_feature_names_out())
            except Exception:
                # Fallback: try to construct names manually (num_cols + onehot names)
                pre_feature_names = []
                if num_cols:
                    pre_feature_names += num_cols
                if cat_cols:
                    try:
                        ohe = pre.named_transformers_['cat'].named_steps['ohe']
                        pre_feature_names += list(ohe.get_feature_names_out(cat_cols))
                    except Exception:
                        # last resort: approximate
                        for c in cat_cols:
                            uniques = list(X[c].astype(str).unique())
                            pre_feature_names += [f"{c}__{u}" for u in uniques]
            meta['preprocessed_feature_names'] = pre_feature_names
    except Exception as e:
        print("Could not derive preprocessed feature names:", e)

    # write meta json
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved model metadata to:", meta_path)

    # feature importance plotting (if RF present)
    try:
        rf = model.named_steps['rf']
        importances = rf.feature_importances_
        feat_names = meta.get('preprocessed_feature_names', meta.get('raw_feature_names', []))
        if len(importances) == len(feat_names):
            imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
            plt.figure(figsize=(9, 8))
            sns.barplot(x=imp_series.values, y=imp_series.index)
            plt.title("Top Feature Importances (Random Forest)")
            outfi = VIS_DIR / "feature_importance.png"
            plt.tight_layout()
            plt.savefig(outfi, bbox_inches='tight')
            plt.close()
            print("Saved feature importance to:", outfi)
        else:
            print("Warning: feature importance length mismatch (importances vs feature names). Skipping importance plot.")
    except Exception as e:
        print("Could not compute/save feature importances:", e)


def main():
    print("== Life Expectancy project - pipeline ==")
    df = load_data(DATA_RAW)
    print("Data loaded. Shape:", df.shape)
    target_col = find_target_column(df)
    if not target_col:
        print("ERROR: could not find target column (life expectancy) in CSV headers.")
        print("CSV headers are:", df.columns.tolist())
        sys.exit(1)
    print("Using target column:", target_col)
    df_clean = basic_clean(df, target_col)
    save_clean(df_clean, DATA_CLEAN)
    run_eda(df_clean, target_col)
    train_model(df_clean, target_col)
    print("== Pipeline finished ==")


if __name__ == "__main__":
    main()
