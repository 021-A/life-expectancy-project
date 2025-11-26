# src/app.py
"""
Streamlit app for Life Expectancy project
Features:
 - Load cleaned dataset and trained model
 - Searching, filtering, sorting the dataset (Country, Year, Status, and all numeric fields)
 - Bulk prediction for filtered rows or single-sample prediction
 - Display and (if needed) generate visuals: corr heatmap, distribution, feature importance, trend average per year
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data" / "cleaned" / "Life_expectancy_clean.csv"
MODEL_PATH = ROOT / "models" / "rf_life_expectancy.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
VIS_DIR = ROOT / "visuals"

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide")
st.title("Life Expectancy Predictor — WHO (2000-2015)")
st.markdown("Filter/search dataset, predict Life Expectancy (single or batch), and view visuals.")

# -- utilities ---------------------------------------------------------
def canonicalize_colname(col: str) -> str:
    """Lowercase + replace spaces/hyphens with underscores and collapse multiple spaces."""
    if not isinstance(col, str):
        return col
    s = " ".join(col.split())  # collapse whitespace
    s = s.replace("-", " ").replace(" ", "_")
    return s.lower()

def load_model_and_meta(model_path: Path, meta_path: Path):
    if not model_path.exists():
        return None, None
    model = joblib.load(model_path)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return model, meta

def ensure_visuals_dir():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

def generate_visuals(df: pd.DataFrame, model, meta: dict):
    """
    Generate and save visuals to VIS_DIR:
      - corr_heatmap.png
      - dist_life_expectancy.png
      - feature_importance.png (if available)
      - trend_life_expectancy.png
    """
    ensure_visuals_dir()
    # canonical target name guesses
    target_col_candidates = [c for c in df.columns if "life" in c and "expect" in c]
    if not target_col_candidates:
        return
    tgt = target_col_candidates[0]

    # 1) Distribution
    try:
        plt.figure(figsize=(8,5))
        sns.histplot(df[tgt].dropna(), bins=30, kde=True)
        plt.title("Distribusi Life Expectancy")
        (VIS_DIR / "dist_life_expectancy.png").write_bytes(plt_to_bytes(plt))
        plt.close()
    except Exception as e:
        print("Could not create distribution plot:", e)

    # 2) Trend average per year
    try:
        if 'year' in df.columns:
            dfg = df.groupby('year')[tgt].mean().reset_index()
            plt.figure(figsize=(10,5))
            sns.lineplot(data=dfg, x='year', y=tgt, marker='o')
            plt.title("Rata-rata Life Expectancy per Tahun")
            (VIS_DIR / "trend_life_expectancy.png").write_bytes(plt_to_bytes(plt))
            plt.close()
    except Exception as e:
        print("Could not create trend plot:", e)

    # 3) Correlation heatmap (numeric)
    try:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            plt.figure(figsize=(12,10))
            sns.heatmap(num.corr(), cmap="RdBu_r", center=0)
            (VIS_DIR / "corr_heatmap.png").write_bytes(plt_to_bytes(plt))
            plt.close()
    except Exception as e:
        print("Could not create corr heatmap:", e)

    # 4) Feature importances (if model exists & is pipeline)
    try:
        if model is not None:
            # try to extract feature importances from final estimator if available
            if hasattr(model, "named_steps") and 'rf' in model.named_steps:
                rf = model.named_steps['rf']
                # get transformed feature names if preprocessor present
                feature_names = []
                if 'pre' in model.named_steps:
                    pre = model.named_steps['pre']
                    try:
                        feature_names = list(pre.get_feature_names_out())
                    except Exception:
                        # fallback: use raw_feature_names from meta
                        feature_names = meta.get("raw_feature_names", [])
                else:
                    feature_names = meta.get("raw_feature_names", [])
                importances = rf.feature_importances_
                if len(importances) == len(feature_names):
                    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                    plt.figure(figsize=(8,6))
                    sns.barplot(x=imp_series.values, y=imp_series.index)
                    plt.title("Top Feature Importances (Random Forest)")
                    (VIS_DIR / "feature_importance.png").write_bytes(plt_to_bytes(plt))
                    plt.close()
    except Exception as e:
        print("Could not create feature importance plot:", e)

def plt_to_bytes(pltobj):
    """Helper: get png bytes from current plt figure"""
    import io
    buf = io.BytesIO()
    pltobj.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data

# -- load data & model -------------------------------------------------
st.sidebar.header("Data & Model")
st.sidebar.markdown("Make sure you ran `run_all.py` to produce cleaned CSV and model.")

# Load cleaned CSV if exists
if not DATA_CLEAN.exists():
    st.sidebar.error(f"Cleaned data not found at: {DATA_CLEAN}")
    st.stop()

df_raw = pd.read_csv(DATA_CLEAN)
# canonicalize column names (if not already)
df_raw.columns = [canonicalize_colname(c) for c in df_raw.columns]

# Load model & meta
model, meta = load_model_and_meta(MODEL_PATH, META_PATH)
if model is None:
    st.sidebar.warning("Trained model not found. Predictions will be disabled until you run training (run_all.py).")

# sidebar filters -----------------------------------------------------
st.sidebar.subheader("Filter / Search")
# Country filter
countries = sorted(df_raw['country'].dropna().unique().tolist()) if 'country' in df_raw.columns else []
sel_countries = st.sidebar.multiselect("Country (select one or more)", options=countries, default=countries[:10] if countries else [])

# Year range
if 'year' in df_raw.columns:
    min_y, max_y = int(df_raw['year'].min()), int(df_raw['year'].max())
    sel_years = st.sidebar.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y))
else:
    sel_years = None

# Status filter
if 'status' in df_raw.columns:
    statuses = sorted(df_raw['status'].dropna().unique().tolist())
    sel_status = st.sidebar.selectbox("Status", options=["All"] + statuses, index=0)
else:
    sel_status = "All"

# Column to search
all_cols = df_raw.columns.tolist()
search_col = st.sidebar.selectbox("Search text in column", options=["None"] + all_cols, index=0)
search_text = st.sidebar.text_input("Search text (substring match)")

# Sorting
sort_by = st.sidebar.selectbox("Sort by", options=["None"] + all_cols, index=0)
sort_asc = st.sidebar.checkbox("Ascending sort", value=True)

# Columns to display in table (default to a sensible subset)
default_display = [
    "country","year","status","life_expectancy","adult_mortality","infant_deaths","alcohol",
    "percentage_expenditure","hepatitis_b","measles","bmi","under_five_deaths","polio",
    "total_expenditure","diphtheria","hiv_aids","gdp","population","thinness_1_19_years",
    "thinness_5_9_years","income_composition_of_resources","schooling"
]
display_cols = [c for c in default_display if c in df_raw.columns]
# allow user to choose
display_cols = st.sidebar.multiselect("Columns to display", options=df_raw.columns.tolist(), default=display_cols)

# Apply filters
df = df_raw.copy()
if sel_countries:
    df = df[df['country'].isin(sel_countries)]
if sel_years and 'year' in df.columns:
    df = df[(df['year'] >= sel_years[0]) & (df['year'] <= sel_years[1])]
if sel_status and sel_status != "All":
    df = df[df['status'] == sel_status]
if search_col != "None" and search_text:
    # do substring match on stringified column
    df = df[df[search_col].astype(str).str.contains(search_text, case=False, na=False)]

if sort_by != "None":
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=sort_asc)

st.subheader(f"Dataset (rows: {df.shape[0]})")
st.markdown("You can download the current filtered table as CSV.")
csv_bytes = df[display_cols].to_csv(index=False).encode('utf-8')
st.download_button("Download CSV (filtered)", data=csv_bytes, file_name="life_expectancy_filtered.csv", mime="text/csv")

# show table
st.dataframe(df[display_cols], use_container_width=True)

# Prediction: single or batch
st.sidebar.subheader("Prediction")
pred_mode = st.sidebar.radio("Prediction mode", options=["Single sample input", "Predict for filtered rows"], index=1)

if pred_mode == "Single sample input":
    st.sidebar.markdown("Provide values for a single sample and click Predict.")
    # Build single-sample inputs; try to use meta raw_feature_names if available
    raw_features = meta.get("raw_feature_names", [])
    sample = {}
    # Build simple inputs for common features (with safe defaults)
    # Use canonical names used in training (e.g., 'year','adult_mortality', etc.)
    sample['year'] = st.sidebar.number_input("Year", min_value=2000, max_value=2015, value=int(df['year'].max()) if 'year' in df.columns else 2015)
    sample['status'] = st.sidebar.selectbox("Status (single)", options=sorted(df['status'].dropna().unique().tolist()) if 'status' in df.columns else ["Developing","Developed"])
    # numeric fields: try to create inputs only for those present in meta/raw_features
    common_numeric = [c for c in raw_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # ask user for a few important numeric features (if present)
    if 'adult_mortality' in df.columns:
        sample['adult_mortality'] = st.sidebar.number_input("Adult Mortality", value=float(df['adult_mortality'].median()))
    if 'gdp' in df.columns:
        sample['gdp'] = st.sidebar.number_input("GDP (per capita)", value=float(df['gdp'].median()))
    if 'bmi' in df.columns:
        sample['bmi'] = st.sidebar.number_input("BMI", value=float(df['bmi'].median()))
    # build sample_df aligned to raw_features when possible
    if raw_features:
        row = {f: sample.get(f, np.nan) for f in raw_features}
        sample_df = pd.DataFrame([row], columns=raw_features)
    else:
        sample_df = pd.DataFrame([sample])

    if st.sidebar.button("Predict single sample"):
        if model is None:
            st.error("Model not available. Run training first.")
        else:
            try:
                pred = model.predict(sample_df)[0]
                st.success(f"Predicted Life Expectancy: {pred:.2f} years")
            except Exception as e:
                st.error("Prediction error: " + str(e))

else:
    # Batch prediction for filtered rows
    st.sidebar.markdown("Predict Life Expectancy for all currently filtered rows and append a column with predictions.")
    if st.sidebar.button("Predict filtered rows"):
        if model is None:
            st.error("Model not available. Run training first.")
        else:
            raw_features = meta.get("raw_feature_names", None)
            if not raw_features:
                st.error("Model metadata not available; cannot determine which features to pass to model.")
            else:
                missing_features = [f for f in raw_features if f not in df.columns]
                if missing_features:
                    st.error("Filtered dataframe is missing features required by the model: " + ", ".join(missing_features))
                else:
                    X = df[raw_features].copy()
                    try:
                        preds = model.predict(X)
                        df_with_pred = df.copy()
                        df_with_pred['predicted_life_expectancy'] = preds
                        st.subheader("Predictions appended to table (first 10 rows):")
                        st.dataframe(df_with_pred[display_cols + ['predicted_life_expectancy']].head(20), use_container_width=True)
                        # allow download of predictions
                        st.download_button("Download predictions CSV", data=df_with_pred.to_csv(index=False).encode('utf-8'),
                                           file_name="predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

# Visuals area
st.markdown("---")
st.subheader("Visuals & Feature Importance")
col1, col2 = st.columns(2)
# helper to show or generate visuals
def show_image_if_exists(name, caption):
    p = VIS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_column_width=True)
        return True
    return False

# If visuals already exist, show them; otherwise generate then show
# Dist
with col1:
    if not show_image_if_exists("dist_life_expectancy.png", "Distribution of Life Expectancy"):
        st.info("Distribution plot not found. Generating now...")
        generate_visuals(df_raw, model, meta)
        if show_image_if_exists("dist_life_expectancy.png", "Distribution of Life Expectancy"):
            pass
with col2:
    if not show_image_if_exists("corr_heatmap.png", "Correlation heatmap"):
        st.info("Correlation heatmap not found. Generating now (if possible).")
        # generation already attempted above

# Next row: feature importance, trend
col3, col4 = st.columns(2)
with col3:
    if not show_image_if_exists("feature_importance.png", "Top Feature Importances"):
        st.info("Feature importance plot not found (requires trained RF model).")
with col4:
    if not show_image_if_exists("trend_life_expectancy.png", "Average Life Expectancy per Year / Trend"):
        st.info("Trend plot not found. Generating now...")
        # generate_visuals already attempted

st.markdown("**Notes**: If certain visuals require the trained model (feature importance), make sure you ran the training pipeline (`run_all.py`) so the model and metadata are available.")
