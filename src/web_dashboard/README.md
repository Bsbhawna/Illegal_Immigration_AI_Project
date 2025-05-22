# 🛂 Immigration Monitoring Dashboard (Streamlit App)

This is a private full-stack AI-powered immigration monitoring dashboard built using **Streamlit**. It provides real-time monitoring, alert analysis, and visualization features for illegal immigration activities using various AI modules and data sources.

## 📊 Features

- **📍 Alert Dashboard:** View MongoDB-based real-time alerts with keyword filters.
- **🌐 Heatmaps:** Visualize illegal immigration hotspots (HTML maps).
- **📈 Statistical Reports:** Analyze sentiment from Reddit, Twitter, and Dark Web.
- **🧠 RAG Chatbot:** Ask any immigration-related question with context-aware responses.
- **📷 CCTV Overlay:** AI-based video surveillance with object/person detection.
- **📄 Fake Document Checker:** Upload docs and detect authenticity (via AI model).
- **✍️ Signature Verifier:** Check if a signature is genuine (Siamese Neural Net).
- **🛰️ Satellite Intrusion Detection:** Border surveillance using CNN over satellite data.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

### 2. Install Requirements

```bash
pip install -r requirements.txt

###3. Run the App
streamlit run dashboard.py


Note on Security
This repository contains no sensitive files. Models, secrets, and private datasets are excluded using .gitignore.

For enterprise deployment, connect the app securely to:

✅ Snowflake Data Warehouse

✅ MongoDB (for real-time alerts)

✅ Secure AI model endpoints (optional)

Tech Stack
-Streamlit
-Pandas, Matplotlib, Plotly
-MongoDB (pymongo)
-Scikit-learn, PyTorch, OpenCV
-LangChain, FAISS (for RAG)
-Google Earth Engine, Folium

Project Structure
.
├── dashboard.py
├── requirements.txt
├── README.md
├── .gitignore
└── sample_data/

 License
 -This project is intended for internal demo and research purposes. Do not use in production without proper security audit
