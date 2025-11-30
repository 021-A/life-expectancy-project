import streamlit as st
from src.components.ui import card
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VISUALS = ROOT / "visuals"

# Set style untuk matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def render(df):
    st.markdown("## 📊 Exploratory Data Analysis")
    
    # Dataset Overview
    st.markdown("### 📋 Dataset Overview")
    
    # Display first 10 rows
    st.markdown("**Preview Data (10 baris pertama):**")
    preview_html = df.head(10).to_html(index=False, classes='dataframe', border=0)
    st.markdown(
        f'<div style="overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">{preview_html}</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📏 Dimensi", f"{df.shape[0]} × {df.shape[1]}")
    with col2:
        st.metric("❓ Missing Values", df.isnull().sum().sum())
    with col3:
        st.metric("🔢 Numeric Columns", df.select_dtypes(include='number').shape[1])
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📈 Visualisasi Data")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "🔥 Correlation", "📉 Trends", "🌍 By Country"])
    
    with tab1:
        st.markdown("#### Distribusi Life Expectancy")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            dist_img = VISUALS / "dist_life_expectancy.png"
            if dist_img.exists():
                st.image(str(dist_img), caption="Distribution of Life Expectancy", use_container_width=True)
            else:
                # Create distribution plot
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#2C3E50')
                ax.set_facecolor('#ECF0F1')
                
                # Histogram
                n, bins, patches = ax.hist(df['life_expectancy'].dropna(), bins=30, 
                                          color='#F39C12', edgecolor='#2C3E50', alpha=0.8)
                
                # Add KDE line
                from scipy import stats
                kde = stats.gaussian_kde(df['life_expectancy'].dropna())
                x_range = np.linspace(df['life_expectancy'].min(), df['life_expectancy'].max(), 100)
                kde_values = kde(x_range)
                ax2 = ax.twinx()
                ax2.plot(x_range, kde_values, color='#2C3E50', linewidth=2, label='KDE')
                ax2.set_ylabel('Density', fontsize=11, color='#2C3E50', fontweight='bold')
                ax2.tick_params(axis='y', labelcolor='#2C3E50')
                
                ax.set_xlabel('Life Expectancy (years)', fontsize=12, fontweight='bold', color='#2C3E50')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold', color='#2C3E50')
                ax.set_title('Distribution of Life Expectancy', fontsize=14, fontweight='bold', 
                           color='white', pad=20)
                ax.grid(axis='y', alpha=0.3, color='#2C3E50')
                ax.tick_params(colors='#2C3E50')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        with col2:
            # Calculate statistics
            life_mean = df['life_expectancy'].mean()
            life_median = df['life_expectancy'].median()
            life_min = df['life_expectancy'].min()
            life_max = df['life_expectancy'].max()
            life_std = df['life_expectancy'].std()
            
            card("Distribution Analysis", f"""
            <div style='font-size: 14px;'>
            <b>Central Tendency:</b>
            <ul style='margin: 5px 0; padding-left: 20px;'>
                <li>Mean: <b style='color: #F39C12;'>{life_mean:.1f}</b> tahun</li>
                <li>Median: <b style='color: #F39C12;'>{life_median:.1f}</b> tahun</li>
            </ul>
            <b>Variability:</b>
            <ul style='margin: 5px 0; padding-left: 20px;'>
                <li>Std Dev: <b>{life_std:.1f}</b> tahun</li>
                <li>Range: <b>{life_max - life_min:.1f}</b> tahun</li>
            </ul>
            <b>Extremes:</b>
            <ul style='margin: 5px 0; padding-left: 20px;'>
                <li>Min: <b>{life_min:.1f}</b> tahun</li>
                <li>Max: <b>{life_max:.1f}</b> tahun</li>
            </ul>
            </div>
            """)
    
    with tab2:
        st.markdown("#### Correlation Heatmap")
        corr_img = VISUALS / "corr_heatmap.png"
        if corr_img.exists():
            st.image(str(corr_img), caption="Correlation Heatmap", use_container_width=True)
        else:
            # Create correlation heatmap
            numeric_df = df.select_dtypes(include='number')
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10), facecolor='#2C3E50')
            
            # Plot heatmap
            sns.heatmap(corr_matrix, annot=False, cmap='RdYlGn', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       ax=ax)
            
            ax.set_title('Correlation Matrix - All Features', fontsize=14, 
                        fontweight='bold', color='white', pad=20)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Show top correlations
        numeric_df = df.select_dtypes(include='number')
        if 'life_expectancy' in numeric_df.columns:
            correlations = numeric_df.corr()['life_expectancy'].sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔼 Top 5 Positive Correlations:**")
                top_pos = correlations[1:6]
                for feat, corr in top_pos.items():
                    bar_width = int(corr * 100)
                    st.markdown(f"""
                    <div style='margin: 8px 0;'>
                        <div style='font-size: 13px; margin-bottom: 3px;'><b>{feat}</b>: {corr:.3f}</div>
                        <div style='background: linear-gradient(to right, #27ae60 {bar_width}%, #ecf0f1 {bar_width}%); 
                                    height: 8px; border-radius: 4px;'></div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🔽 Top 5 Negative Correlations:**")
                top_neg = correlations[-5:]
                for feat, corr in top_neg.items():
                    bar_width = int(abs(corr) * 100)
                    st.markdown(f"""
                    <div style='margin: 8px 0;'>
                        <div style='font-size: 13px; margin-bottom: 3px;'><b>{feat}</b>: {corr:.3f}</div>
                        <div style='background: linear-gradient(to right, #e74c3c {bar_width}%, #ecf0f1 {bar_width}%); 
                                    height: 8px; border-radius: 4px;'></div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### Life Expectancy Trends Over Time")
        trend_img = VISUALS / "trend_life_expectancy.png"
        
        if trend_img.exists():
            st.image(str(trend_img), caption="Life Expectancy Trends", use_container_width=True)
        else:
            # Create trend plot
            yearly_avg = df.groupby('year')['life_expectancy'].mean()
            yearly_std = df.groupby('year')['life_expectancy'].std()
            
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='#2C3E50')
            ax.set_facecolor('#ECF0F1')
            
            # Plot line with confidence interval
            ax.plot(yearly_avg.index, yearly_avg.values, 
                   marker='o', linewidth=3, color='#F39C12', 
                   markersize=8, markerfacecolor='#2C3E50', 
                   markeredgecolor='#F39C12', markeredgewidth=2,
                   label='Average Life Expectancy')
            
            # Add confidence interval
            ax.fill_between(yearly_avg.index, 
                           yearly_avg.values - yearly_std.values,
                           yearly_avg.values + yearly_std.values,
                           alpha=0.2, color='#F39C12', label='± 1 Std Dev')
            
            ax.set_xlabel('Year', fontsize=12, fontweight='bold', color='white')
            ax.set_ylabel('Average Life Expectancy (years)', fontsize=12, fontweight='bold', color='white')
            ax.set_title('Life Expectancy Trends Over Time (2000-2015)', 
                        fontsize=14, fontweight='bold', color='white', pad=20)
            ax.grid(True, alpha=0.3, color='#2C3E50')
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
            ax.tick_params(colors='white')
            
            # Add annotations for min and max
            min_year = yearly_avg.idxmin()
            max_year = yearly_avg.idxmax()
            ax.annotate(f'Lowest\n{yearly_avg[min_year]:.1f}', 
                       xy=(min_year, yearly_avg[min_year]),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='#e74c3c', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='white'),
                       color='white', fontweight='bold', fontsize=9)
            ax.annotate(f'Highest\n{yearly_avg[max_year]:.1f}', 
                       xy=(max_year, yearly_avg[max_year]),
                       xytext=(10, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='#27ae60', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', color='white'),
                       color='white', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Show yearly data in expandable table
        with st.expander("📋 Lihat Data Tabel Lengkap"):
            yearly_stats = df.groupby('year').agg({
                'life_expectancy': ['mean', 'median', 'std', 'min', 'max']
            }).round(2)
            yearly_stats.columns = ['Mean', 'Median', 'Std Dev', 'Min', 'Max']
            yearly_html = yearly_stats.to_html()
            st.markdown(yearly_html, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("#### Life Expectancy by Country")
        
        # Country selector
        all_countries = sorted(df['country'].unique())
        default_countries = all_countries[:10] if len(all_countries) >= 10 else all_countries
        
        selected_countries = st.multiselect(
            "🌍 Pilih negara untuk perbandingan (max 15):",
            options=all_countries,
            default=default_countries,
            max_selections=15
        )
        
        if selected_countries:
            filtered_df = df[df['country'].isin(selected_countries)]
            country_avg = filtered_df.groupby('country')['life_expectancy'].mean().sort_values(ascending=True)
            
            st.markdown("**Rata-rata Life Expectancy per Negara:**")
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, max(6, len(selected_countries) * 0.5)), 
                                  facecolor='#2C3E50')
            ax.set_facecolor('#ECF0F1')
            
            colors = ['#F39C12' if val >= country_avg.median() else '#e74c3c' 
                     for val in country_avg.values]
            bars = ax.barh(country_avg.index, country_avg.values, color=colors, 
                          edgecolor='#2C3E50', linewidth=1.5)
            
            ax.set_xlabel('Average Life Expectancy (years)', fontsize=12, 
                         fontweight='bold', color='white')
            ax.set_ylabel('Country', fontsize=12, fontweight='bold', color='white')
            ax.set_title('Life Expectancy by Country', fontsize=14, 
                        fontweight='bold', color='white', pad=20)
            ax.grid(axis='x', alpha=0.3, color='#2C3E50')
            ax.tick_params(colors='white')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, country_avg.values)):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=9, 
                       fontweight='bold', color='white')
            
            # Add median line
            median_val = country_avg.median()
            ax.axvline(median_val, color='white', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Median: {median_val:.1f}')
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show detailed statistics
            with st.expander("📊 Lihat Statistik Detail"):
                country_detail = filtered_df.groupby('country').agg({
                    'life_expectancy': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2)
                country_detail.columns = ['N', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
                country_detail = country_detail.sort_values('Mean', ascending=False)
                country_html = country_detail.to_html()
                st.markdown(country_html, unsafe_allow_html=True)
        else:
            st.info("💡 Pilih minimal 1 negara untuk melihat visualisasi dan statistik")
    
    # Column Information
    with st.expander("📋 Informasi Kolom Dataset"):
        st.markdown("**Daftar Lengkap Kolom:**")
        
        col_data = []
        for i, col in enumerate(df.columns, 1):
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            col_data.append({
                'No': i,
                'Nama Kolom': col,
                'Tipe Data': str(df[col].dtype),
                'Missing': df[col].isnull().sum(),
                'Missing %': f"{missing_pct:.1f}%",
                'Unique': df[col].nunique(),
                'Sample': str(df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'N/A')
            })
        
        info_df = pd.DataFrame(col_data)
        info_html = info_df.to_html(index=False)
        st.markdown(info_html, unsafe_allow_html=True)