import streamlit as st
from pathlib import Path

# Path ke root project
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"

def inject_css():
    """Inject custom CSS ke aplikasi"""
    css_file = ASSETS / "styles.css"
    if css_file.exists():
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ CSS file tidak ditemukan")

def header(title="Life Expectancy Predictor"):
    """Render header aplikasi"""
    st.markdown(f"""
    <div class="app-header">
        <div class="app-title">🌍 {title}</div>
        <div class="app-subtitle">WHO Global Health Observatory (2000-2015) · Interactive Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

def footer():
    """Render footer aplikasi"""
    st.markdown("""
    <div class="app-footer">
        © 2025 · Life Expectancy Prediction Project<br>
        Built with ❤️ using Streamlit · Data from WHO & UN
    </div>
    """, unsafe_allow_html=True)

def card(title, content):
    """Render card component"""
    st.markdown(f"""
    <div class="custom-card">
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """, unsafe_allow_html=True)

def metric_card(label, value, delta=None):
    """Render metric card dengan styling custom"""
    st.metric(label=label, value=value, delta=delta)