import streamlit as st
from src.components.ui import card

def render(df):
    st.markdown("## ℹ️ Tentang Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Deskripsi Project
        
        Project ini menganalisis dan memprediksi **Life Expectancy** (harapan hidup) 
        menggunakan data dari WHO Global Health Observatory (2000-2015).
        
        ### 📊 Dataset
        
        - **Sumber:** WHO & United Nations
        - **Periode:** 2000-2015
        - **Jumlah Negara:** 193 negara
        - **Total Records:** 2,938 data points
        - **Variabel:** 22 features (ekonomi, kesehatan, sosial)
        
        ### 🤖 Model Machine Learning
        
        - **Algorithm:** Random Forest Regressor
        - **Features:** GDP, Schooling, Adult Mortality, BMI, dll.
        - **Performance:** R² score > 0.80
        
        ### 🛠️ Technology Stack
        
        - **Framework:** Streamlit
        - **ML Library:** Scikit-learn
        - **Visualization:** Matplotlib, Seaborn
        - **Language:** Python 3.x
        """)
    
    with col2:
        card("👨‍💻 Developer", """
        <b>Dedi Firmansyah</b><br>
        📧 dedifirmansyah34770@gmail.com<br><br>
        
        <b>GitHub Repository:</b><br>
        <a href="https://github.com/021-A/life-expectancy-project" target="_blank">
        github.com/021-A/life-expectancy-project
        </a>
        """)
        
        card("📚 References", """
        <ul style='margin: 0; padding-left: 20px;'>
            <li>WHO Global Health Observatory</li>
            <li>United Nations (GDP data)</li>
            <li>Kaggle Dataset</li>
        </ul>
        """)
        
        card("📝 License", """
        MIT License © 2025<br>
        Open source project
        """)
    
    st.markdown("---")
    
    # Acknowledgments
    st.markdown("### 🙏 Acknowledgments")
    st.markdown("""
    Terima kasih kepada:
    - **WHO** untuk data kesehatan global
    - **United Nations** untuk data ekonomi
    - **Kaggle Community** untuk dataset dan insights
    - **Streamlit** untuk framework yang powerful
    """)