import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from src.components.ui import card

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "rf_life_expectancy.joblib"
META_PATH = ROOT / "models" / "model_meta.json"

def render(df):
    st.markdown("## 🔮 Prediksi Life Expectancy")
    
    # Load model
    model = None
    meta = {}
    
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            st.success("✅ Model berhasil dimuat!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
    else:
        st.warning("⚠️ Model tidak ditemukan. Pastikan file model ada di folder `models/`")
        st.info(f"📁 Expected path: {MODEL_PATH}")
    
    if META_PATH.exists():
        try:
            with open(META_PATH, 'r') as f:
                meta = json.load(f)
        except:
            pass
    
    st.markdown("---")
    
    # Model Info
    if meta:
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Model", meta.get('model_type', 'Random Forest'))
        col2.metric("📊 R² Score", f"{meta.get('r2_score', 0):.4f}")
        col3.metric("📉 RMSE", f"{meta.get('rmse', 0):.2f}")
    
    st.markdown("---")
    
    # Prediction Form
    st.markdown("### 🎯 Single Sample Prediction")
    
    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            year = st.number_input(
                "Year", 
                min_value=int(df['year'].min()), 
                max_value=int(df['year'].max()) + 15, 
                value=int(df['year'].max())
            )
            
            country = st.selectbox(
                "Country", 
                options=sorted(df['country'].unique()),
                index=list(sorted(df['country'].unique())).index('Indonesia') if 'Indonesia' in df['country'].values else 0
            )
        
        with col2:
            status = st.selectbox(
                "Status", 
                options=sorted(df['status'].unique())
            )
            
            adult_mortality = st.number_input(
                "Adult Mortality", 
                min_value=0, 
                max_value=1000, 
                value=int(df['adult_mortality'].median()),
                help="Number of deaths per 1000 population"
            )
        
        with col3:
            infant_deaths = st.number_input(
                "Infant Deaths", 
                min_value=0, 
                max_value=2000, 
                value=int(df['infant_deaths'].median()),
                help="Number of infant deaths per 1000 population"
            )
            
            alcohol = st.number_input(
                "Alcohol", 
                min_value=0.0, 
                max_value=20.0, 
                value=float(df['alcohol'].median()),
                help="Alcohol consumption (liters per capita)"
            )
        
        with col4:
            bmi = st.number_input(
                "BMI", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(df['bmi'].median()),
                help="Average Body Mass Index"
            )
            
            gdp = st.number_input(
                "GDP", 
                min_value=0, 
                max_value=200000, 
                value=int(df['gdp'].median()) if df['gdp'].median() > 0 else 5000,
                help="GDP per capita (USD)"
            )
        
        # Additional features
        st.markdown("#### 📊 Additional Features (Optional)")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            hepatitis_b = st.number_input(
                "Hepatitis B (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(df['hepatitis_b'].median()) if df['hepatitis_b'].median() > 0 else 80.0
            )
            
            polio = st.number_input(
                "Polio (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(df['polio'].median()) if df['polio'].median() > 0 else 85.0
            )
        
        with col6:
            diphtheria = st.number_input(
                "Diphtheria (%)", 
                min_value=0.0, 
                max_value=100.0, 
                value=float(df['diphtheria'].median()) if df['diphtheria'].median() > 0 else 85.0
            )
            
            hiv_aids = st.number_input(
                "HIV/AIDS", 
                min_value=0.0, 
                max_value=50.0, 
                value=float(df['hiv/aids'].median())
            )
        
        with col7:
            schooling = st.number_input(
                "Schooling (years)", 
                min_value=0.0, 
                max_value=25.0, 
                value=float(df['schooling'].median())
            )
            
            income_composition = st.number_input(
                "Income Composition", 
                min_value=0.0, 
                max_value=1.0, 
                value=float(df['income_composition_of_resources'].median()),
                help="Human Development Index"
            )
        
        with col8:
            population = st.number_input(
                "Population", 
                min_value=0, 
                max_value=1500000000, 
                value=int(df['population'].median()) if df['population'].median() > 0 else 10000000
            )
            
            total_expenditure = st.number_input(
                "Total Expenditure (%)", 
                min_value=0.0, 
                max_value=20.0, 
                value=float(df['total_expenditure'].median()) if df['total_expenditure'].median() > 0 else 5.0
            )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Prediksi Sekarang", use_container_width=True)
    
    # Handle submission
    if submitted:
        if model is None:
            st.error("❌ Model tidak tersedia! Pastikan file model ada di `models/rf_life_expectancy.joblib`")
        else:
            try:
                # Prepare input data
                input_data = {
                    'year': year,
                    'country': country,
                    'status': status,
                    'adult_mortality': adult_mortality,
                    'infant_deaths': infant_deaths,
                    'alcohol': alcohol,
                    'percentage_expenditure': 0,  # Default value
                    'hepatitis_b': hepatitis_b,
                    'measles': 0,  # Default value
                    'bmi': bmi,
                    'under_five_deaths': 0,  # Default value
                    'polio': polio,
                    'total_expenditure': total_expenditure,
                    'diphtheria': diphtheria,
                    'hiv/aids': hiv_aids,
                    'gdp': gdp,
                    'population': population,
                    'thinness_1_19_years': 0,  # Default value
                    'thinness_5_9_years': 0,  # Default value
                    'income_composition_of_resources': income_composition,
                    'schooling': schooling
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Predict
                prediction = model.predict(input_df)[0]
                
                # Display result
                st.markdown("---")
                st.markdown("### 🎉 Hasil Prediksi")
                
                # Big result display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
                        <h1 style='color: white; font-size: 3rem; margin: 0;'>{prediction:.1f}</h1>
                        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 0;'>tahun</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Interpretation
                if prediction < 60:
                    interpretation = "Rendah - Perlu perhatian serius pada faktor kesehatan dan ekonomi"
                    color = "🔴"
                    bg_color = "#e74c3c"
                elif prediction < 70:
                    interpretation = "Sedang - Ada ruang untuk perbaikan, fokus pada pendidikan dan GDP"
                    color = "🟡"
                    bg_color = "#f39c12"
                else:
                    interpretation = "Tinggi - Kondisi baik, pertahankan dan tingkatkan"
                    color = "🟢"
                    bg_color = "#27ae60"
                
                st.markdown(f"""
                <div style='padding: 1rem; background-color: {bg_color}22; border-left: 4px solid {bg_color}; border-radius: 8px;'>
                    <h3>{color} Interpretasi</h3>
                    <p style='margin: 0;'>{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show input summary
                with st.expander("📋 Lihat Input Data"):
                    st.json(input_data)
                
            except Exception as e:
                st.error(f"❌ Error saat prediksi: {e}")
                st.info("💡 Debug info:")
                st.code(str(e))
    
    # Tips
    st.markdown("---")
    card("💡 Tips Penggunaan", """
    <ul style='margin: 0; padding-left: 20px;'>
        <li>Isi semua field dengan data yang valid</li>
        <li>Gunakan data historis sebagai referensi</li>
        <li>Klik tombol <b>"🚀 Prediksi Sekarang"</b> untuk melihat hasil</li>
        <li>Faktor yang paling berpengaruh: GDP, Schooling, Adult Mortality</li>
    </ul>
    """)