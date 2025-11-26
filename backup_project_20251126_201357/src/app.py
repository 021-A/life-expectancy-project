# src/app.py
"""
Streamlit enhanced UI for Life Expectancy Predictor
- English UI
- Interactive HTML table with DataTables (search/sort/pagination)
- Clean CSS styling embedded
- Visuals, single & batch prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data" / "cleaned" / "Life_expectancy_clean.csv"
MODEL_PATH = ROOT / "models" / "rf_life_expectancy.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
VIS_DIR = ROOT / "visuals"

st.set_page_config(page_title="Life Expectancy Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------- Custom CSS (header + cards) ----------
CUSTOM_CSS = """
<style>
:root{
  --accent:#6c5ce7;
  --muted:#6b7280;
  --card-bg: #ffffff;
  --bg: #f7f7fb;
}
body {
  background: var(--bg);
  font-family: "Inter", "Segoe UI", Roboto, Arial, sans-serif;
}
.header-title{
  font-size:32px;
  font-weight:700;
  color: #111827;
  margin-bottom:0;
}
.header-sub{
  color:var(--muted);
  margin-top:4px;
  margin-bottom:18px;
}
.kpi {
  background: linear-gradient(90deg, rgba(108,92,231,0.06), rgba(108,92,231,0.02));
  border-radius:12px;
  padding:12px;
  box-shadow: 0 1px 6px rgba(15,23,42,0.04);
}
.card {
  background: var(--card-bg);
  border-radius:10px;
  padding:14px;
  box-shadow: 0 1px 8px rgba(15,23,42,0.04);
}
.small-muted { color: var(--muted); font-size:13px; }
.dt-container { margin-top: 8px; }
.table-wrap { background:white; padding:12px; border-radius:8px; box-shadow:0 1px 6px rgba(0,0,0,0.04); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="header-title">Life Expectancy Predictor — WHO (2000–2015)</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Filter the dataset, predict life expectancy (single or batch), and explore visuals.</div>', unsafe_allow_html=True)

# ---------- Helpers ----------
def canonicalize_colname(col: str) -> str:
    if not isinstance(col, str):
        return col
    s = " ".join(col.split())
    s = s.replace("-", " ").replace(" ", "_")
    return s.lower()

def load_model_and_meta(model_path: Path, meta_path: Path):
    model = None
    meta = {}
    if model_path.exists():
        model = joblib.load(model_path)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return model, meta

def ensure_visuals_dir():
    VIS_DIR.mkdir(parents=True, exist_ok=True)

def plt_to_bytes(pltobj):
    import io
    buf = io.BytesIO()
    pltobj.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    data = buf.read()
    buf.close()
    return data

def generate_visuals(df: pd.DataFrame, model, meta: dict):
    ensure_visuals_dir()
    target_col_candidates = [c for c in df.columns if "life" in c and "expect" in c]
    if not target_col_candidates:
        return
    tgt = target_col_candidates[0]
    # Distribution
    try:
        plt.figure(figsize=(8,4.5))
        sns.histplot(df[tgt].dropna(), bins=30, kde=True)
        plt.title("Distribution of Life Expectancy")
        (VIS_DIR / "dist_life_expectancy.png").write_bytes(plt_to_bytes(plt))
        plt.close()
    except Exception:
        pass
    # Trend
    try:
        if 'year' in df.columns:
            dfg = df.groupby('year')[tgt].mean().reset_index()
            plt.figure(figsize=(8,4.5))
            sns.lineplot(data=dfg, x='year', y=tgt, marker='o')
            plt.title("Average Life Expectancy by Year")
            (VIS_DIR / "trend_life_expectancy.png").write_bytes(plt_to_bytes(plt))
            plt.close()
    except Exception:
        pass
    # Correlation
    try:
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] >= 2:
            plt.figure(figsize=(8,6))
            sns.heatmap(num.corr(), cmap="RdBu_r", center=0)
            (VIS_DIR / "corr_heatmap.png").write_bytes(plt_to_bytes(plt))
            plt.close()
    except Exception:
        pass
    # Feature importances (if available)
    try:
        if model is not None and hasattr(model, "named_steps") and 'rf' in model.named_steps:
            rf = model.named_steps['rf']
            feat_names = meta.get("raw_feature_names", []) or []
            importances = rf.feature_importances_
            if len(importances) == len(feat_names) and len(feat_names) > 0:
                imp_series = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(20)
                plt.figure(figsize=(6,5))
                sns.barplot(x=imp_series.values, y=imp_series.index)
                plt.title("Top Feature Importances (RF)")
                (VIS_DIR / "feature_importance.png").write_bytes(plt_to_bytes(plt))
                plt.close()
    except Exception:
        pass

# ---------- Load data & model ----------
st.sidebar.header("Data & Model")
st.sidebar.write("Run `run_all.py` to produce cleaned CSV and trained model if missing.")

if not DATA_CLEAN.exists():
    st.sidebar.error(f"Cleaned data not found: {DATA_CLEAN}")
    st.stop()

df_raw = pd.read_csv(DATA_CLEAN)
df_raw.columns = [canonicalize_colname(c) for c in df_raw.columns]

model, meta = load_model_and_meta(MODEL_PATH, META_PATH)
if model is None:
    st.sidebar.warning("Trained model not found — predictions disabled until training is run.")

# ---------- Sidebar filters ----------
st.sidebar.subheader("Filters")
countries = sorted(df_raw['country'].dropna().unique().tolist()) if 'country' in df_raw.columns else []
sel_countries = st.sidebar.multiselect("Countries", options=countries, default=countries[:8])

if 'year' in df_raw.columns:
    min_y, max_y = int(df_raw['year'].min()), int(df_raw['year'].max())
    sel_years = st.sidebar.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y))
else:
    sel_years = (None, None)

if 'status' in df_raw.columns:
    statuses = sorted(df_raw['status'].dropna().unique().tolist())
    sel_status = st.sidebar.selectbox("Status", options=["All"] + statuses, index=0)
else:
    sel_status = "All"

all_cols = df_raw.columns.tolist()
search_col = st.sidebar.selectbox("Search in column", options=["None"] + all_cols, index=0)
search_text = st.sidebar.text_input("Search text (substring)")

sort_by = st.sidebar.selectbox("Sort by", options=["None"] + all_cols, index=0)
sort_asc = st.sidebar.checkbox("Ascending", value=True)

default_display = [
    "country","year","status","life_expectancy","adult_mortality","infant_deaths","alcohol",
    "percentage_expenditure","hepatitis_b","measles","bmi","under_five_deaths","polio",
    "total_expenditure","diphtheria","hiv_aids","gdp","population","thinness_1_19_years",
    "thinness_5_9_years","income_composition_of_resources","schooling"
]
display_cols = [c for c in default_display if c in df_raw.columns]
display_cols = st.sidebar.multiselect("Display columns", options=df_raw.columns.tolist(), default=display_cols)

# ---------- Apply filters ----------
df = df_raw.copy()
if sel_countries:
    df = df[df['country'].isin(sel_countries)]
if sel_years and sel_years[0] is not None:
    df = df[(df['year'] >= sel_years[0]) & (df['year'] <= sel_years[1])]
if sel_status and sel_status != "All":
    df = df[df['status'] == sel_status]
if search_col != "None" and search_text:
    df = df[df[search_col].astype(str).str.contains(search_text, case=False, na=False)]
if sort_by != "None" and sort_by in df.columns:
    df = df.sort_values(by=sort_by, ascending=sort_asc)

st.markdown('<div class="kpi card"><strong>Dataset</strong> · rows: {}</div>'.format(df.shape[0]), unsafe_allow_html=True)

# fallback columns
if not display_cols:
    display_cols = df.columns.tolist()

# ---------- HTML interactive table (DataTables) ----------
def make_datatables_html(df_sub: pd.DataFrame, table_id="table1", max_height=450):
    # build HTML and include DataTables (CDN). This runs inside an iframe via components.html
    html_table = df_sub.to_html(index=False, classes="display compact", border=0)
    # ensure table has id
    html_table = html_table.replace("<table ", f"<table id='{table_id}' ")
    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
      <style> body{{ font-family: Inter, Arial, sans-serif; padding:8px; }} </style>
    </head>
    <body>
      <div class="table-wrap">{html_table}</div>
      <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
      <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
      <script>
        $(document).ready(function(){{
            $('#{table_id}').DataTable({{
                pageLength: 10,
                lengthMenu: [[10, 25, 50, -1],[10,25,50,"All"]],
                scrollY: '{max_height}px',
                scrollCollapse: true,
                responsive: true
            }});
        }});
      </script>
    </body>
    </html>
    """
    return html

st.markdown('<div class="card"><div class="small-muted">You can download the filtered dataset below.</div></div>', unsafe_allow_html=True)
csv_bytes = df[display_cols].to_csv(index=False).encode('utf-8')
st.download_button("Download CSV (filtered)", data=csv_bytes, file_name="life_expectancy_filtered.csv", mime="text/csv")

# embed DataTable
tbl_html = make_datatables_html(df[display_cols], table_id="life_table", max_height=420)
components.html(tbl_html, height=520, scrolling=True)

# ---------- Prediction controls ----------
st.sidebar.subheader("Prediction")
pred_mode = st.sidebar.radio("Mode", options=["Single sample", "Predict filtered rows"], index=1)

if pred_mode == "Single sample":
    st.sidebar.markdown("Enter a single sample to predict life expectancy.")
    raw_features = meta.get("raw_feature_names", []) if meta else []
    sample = {}
    sample['year'] = st.sidebar.number_input("Year", min_value=2000, max_value=2015, value=int(df['year'].max()) if 'year' in df.columns else 2015)
    sample['status'] = st.sidebar.selectbox("Status", options=sorted(df['status'].dropna().unique().tolist()) if 'status' in df.columns else ["Developing","Developed"])
    if 'adult_mortality' in df.columns:
        sample['adult_mortality'] = st.sidebar.number_input("Adult Mortality", value=float(df['adult_mortality'].median()))
    if 'gdp' in df.columns:
        sample['gdp'] = st.sidebar.number_input("GDP (per capita)", value=float(df['gdp'].median()))
    if 'bmi' in df.columns:
        sample['bmi'] = st.sidebar.number_input("BMI", value=float(df['bmi'].median()))
    if raw_features:
        row = {f: sample.get(f, np.nan) for f in raw_features}
        sample_df = pd.DataFrame([row], columns=raw_features)
    else:
        sample_df = pd.DataFrame([sample])
    if st.sidebar.button("Predict single sample"):
        if model is None:
            st.error("Model not available. Run training (run_all.py).")
        else:
            try:
                pred = model.predict(sample_df)[0]
                st.success(f"Predicted Life Expectancy: {pred:.2f} years")
            except Exception as e:
                st.error("Prediction error: " + str(e))
else:
    st.sidebar.markdown("Predict for the rows currently filtered above.")
    if st.sidebar.button("Predict filtered rows"):
        if model is None:
            st.error("Model not available. Run training (run_all.py).")
        else:
            raw_features = meta.get("raw_feature_names", None) if meta else None
            if not raw_features:
                st.error("Model metadata missing; cannot determine input features.")
            else:
                missing = [f for f in raw_features if f not in df.columns]
                if missing:
                    st.error("Missing features for model: " + ", ".join(missing))
                else:
                    X = df[raw_features].copy()
                    try:
                        preds = model.predict(X)
                        df_with_pred = df.copy()
                        df_with_pred['predicted_life_expectancy'] = preds
                        st.success("Predictions computed — see table below (first 20 rows).")
                        html_tbl = df_with_pred[display_cols + ['predicted_life_expectancy']].head(20).to_html(index=False)
                        st.markdown(html_tbl, unsafe_allow_html=True)
                        st.download_button("Download predictions CSV", data=df_with_pred.to_csv(index=False).encode('utf-8'),
                                           file_name="predictions.csv", mime="text/csv")
                    except Exception as e:
                        st.error("Prediction failed: " + str(e))

# ---------- Visuals ----------
st.markdown("---")
st.markdown('<div class="card"><strong>Visuals</strong><div class="small-muted">Distribution, correlation heatmap, feature importance and trend (if available).</div></div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
def show_image(name, caption, col):
    p = VIS_DIR / name
    if p.exists():
        with col:
            st.image(str(p), caption=caption, use_container_width=True)
        return True
    return False

# ensure visuals exist (generate if not)
generate_visuals(df_raw, model, meta)

show_image("dist_life_expectancy.png", "Distribution of Life Expectancy", col1)
show_image("corr_heatmap.png", "Correlation Heatmap", col2)
show_image("feature_importance.png", "Top Feature Importances (if model)", col1)
show_image("trend_life_expectancy.png", "Average Life Expectancy per Year", col2)

st.markdown("<div class='small-muted'>Notes: Feature importance requires trained model + metadata. Use <code>run_all.py</code> to train & build visuals.</div>", unsafe_allow_html=True)
