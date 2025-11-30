# 🌍 Life Expectancy Predictor

Interactive dashboard untuk prediksi dan analisis life expectancy berbasis data WHO (2000-2015).

## 🚀 Live Demo

**[🔗 Try the App Here!](https://life-expectancy-project-38qz7rkqbjdski8xukmph5.streamlit.app/)**

## ✨ Features

- 📊 **Exploratory Data Analysis** - Visualisasi distribusi, korelasi, dan trends
- 🔮 **Life Expectancy Prediction** - Model Random Forest untuk prediksi
- 🌍 **Country Comparison** - Perbandingan antar negara
- 🎨 **Modern UI** - Custom theme dengan color palette yang menarik

## 🎨 Color Theme

- **Primary:** `#2C3E50` (Dark Blue) - Background utama, header, footer
- **Accent:** `#F39C12` (Orange) - CTA buttons, highlights
- **Secondary:** `#ECF0F1` (Light Gray) - Cards, forms, text areas

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn, Scipy
- **ML:** Scikit-learn, Random Forest
- **Data Processing:** Pandas, NumPy

## 📊 Dataset

- **Source:** WHO Global Health Observatory (2000-2015)
- **Records:** 2,874 entries
- **Countries:** 193 countries
- **Features:** 22 variables (economic, health, social indicators)

## 📁 Project Structure
```
Life-Expectancy-Project/
├── .streamlit/          # Streamlit configuration
├── assets/              # CSS styles
├── data/cleaned/        # Dataset
├── src/
│   ├── components/      # Reusable UI components
│   └── pages/           # Multi-page modules
├── app.py               # Main application
└── requirements.txt     # Dependencies
```

## 🚀 Local Development
```bash
# Clone repository
git clone https://github.com/021-A/life-expectancy-project.git
cd life-expectancy-project

# Create virtual environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 📸 Screenshots

### Home Page
Dashboard overview dengan quick statistics dan key insights.

### Explore Page
Interactive data exploration dengan visualisasi distribusi, korelasi, dan trends.

### Predict Page
Form prediksi life expectancy berdasarkan berbagai faktor kesehatan dan ekonomi.

## 👨‍💻 Author

**Dedi Firmansyah**
- 📧 Email: dedifirmansyah34770@gmail.com
- 🐙 GitHub: [@021-A](https://github.com/021-A)
- 🌐 Live Demo: [Streamlit App](https://life-expectancy-project-38qz7rkqbjdski8xukmph5.streamlit.app/)

## 🙏 Acknowledgments

- **WHO** - Global Health Observatory data
- **United Nations** - Economic data (GDP)
- **Kaggle Community** - Dataset and insights
- **Streamlit** - Amazing framework for data apps

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

⭐ **Don't forget to star this repo if you find it useful!** ⭐