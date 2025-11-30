import streamlit as st
from src.components.ui import card

def render(df):
    st.markdown("## 🏠 Selamat Datang")
    
    st.markdown("""
    Dashboard ini mengeksplorasi data **Life Expectancy** dari WHO (2000-2015) 
    dan menyediakan model prediksi berbasis Machine Learning.
    """)
    
    # Quick Stats
    st.markdown("### 📊 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌍 Negara", df['country'].nunique())
    with col2:
        st.metric("📅 Periode", f"{int(df['year'].min())}-{int(df['year'].max())}")
    with col3:
        st.metric("📈 Total Data", f"{len(df):,}")
    with col4:
        avg_life = df['life_expectancy'].mean()
        st.metric("⏱️ Rata-rata", f"{avg_life:.1f} tahun")
    
    st.markdown("---")
    
    # Key Insights
    col1, col2 = st.columns(2)
    
    with col1:
        card("🔍 Key Insights", """
        <ul style='margin: 0; padding-left: 20px;'>
            <li><b>GDP per capita</b> memiliki korelasi kuat dengan life expectancy</li>
            <li><b>Schooling</b> (pendidikan) adalah faktor penting</li>
            <li><b>Adult Mortality</b> berpengaruh negatif signifikan</li>
            <li><b>Income composition</b> menunjukkan hubungan positif</li>
        </ul>
        """)
    
    with col2:
        card("🎯 Fitur Dashboard", """
        <ul style='margin: 0; padding-left: 20px;'>
            <li><b>Explore:</b> Visualisasi data & analisis EDA</li>
            <li><b>Predict:</b> Prediksi life expectancy</li>
            <li><b>About:</b> Informasi project & dataset</li>
        </ul>
        """)
    
    st.info("💡 **Tip:** Gunakan navigasi di atas untuk menjelajahi fitur-fitur dashboard!")