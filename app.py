import streamlit as st
import pandas as pd
from pathlib import Path

# Import components
from src.components.ui import inject_css, header, footer

# Import pages
from src.pages import home, explore, predict, about

# Configuration
st.set_page_config(
    page_title="Life Expectancy Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paths
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned" / "Life_expectancy_clean.csv"

# Load data
@st.cache_data
def load_data():
    """Load and cache dataset"""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        st.error(f"❌ Data tidak ditemukan di: {DATA_PATH}")
        st.info("💡 Pastikan file `Life_expectancy_clean.csv` ada di folder `data/cleaned/`")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

# Inject custom CSS
inject_css()

# Render header
header("Life Expectancy Predictor")

# Load dataset
df = load_data()

# Navigation
st.markdown("### 🧭 Navigation")
page = st.radio(
    "",
    options=["🏠 Home", "📊 Explore", "🔮 Predict", "ℹ️ About"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# Render selected page
if page == "🏠 Home":
    home.render(df)
elif page == "📊 Explore":
    explore.render(df)
elif page == "🔮 Predict":
    predict.render(df)
elif page == "ℹ️ About":
    about.render(df)

# Render footer
footer()