# 🚗 AutoInsight AI

AutoInsight AI is a Machine Learning powered vehicle valuation and market intelligence platform that predicts realistic used car prices and categorizes vehicles into market segments using data-driven models.

The project demonstrates a complete end-to-end Machine Learning workflow — from data preprocessing and model training to deployment using Streamlit.

---

## 🌐 Live Application

👉 https://autoinsight-ai.streamlit.app

---

## ✨ Features

- Predict realistic used car prices using trained ML models
- Classify vehicles into Budget / Midrange / Premium segments
- Interactive dataset exploration dashboard
- Clean and responsive Streamlit interface
- End-to-end ML pipeline deployment

---

## 🧠 Machine Learning Overview

The system uses multiple machine learning approaches:

- **Regression Model**  
  Random Forest Regressor for continuous price prediction.

- **Classification Model**  
  Random Forest Classifier for vehicle price category prediction.

- **Clustering Model**  
  K-Means clustering for market segmentation analysis.

---

## 📊 Model Performance

The models were trained on real-world used vehicle data and evaluated using standard machine learning metrics.

- Classification Accuracy: ~83%
- Regression R² Score: ~0.82

Due to the high variability in used car pricing (condition, location, demand, and seller behavior), the model focuses on providing realistic price ranges rather than exact price estimation.

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib

---

## 📁 Project Structure

```
AutoInsight-AI/
│
├── app/            # Streamlit application
├── model/          # Trained ML pipelines
├── notebooks/      # Research & experimentation
├── src/            # Model training scripts
├── data/           # Sample dataset (for demo)
├── requirements.txt
└── README.md
```

---

## 🚀 Running Locally

```bash
git clone https://github.com/Saksham2404/AutoInsight-AI.git
cd AutoInsight-AI
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 👨‍💻 Author

**Saksham Malhotra**

Machine Learning & Data Science student focused on building practical AI applications and data-driven systems.

- LinkedIn: https://www.linkedin.com/in/saksham02

---

## 📌 Note

The original dataset is not included due to size limitations.  
A sample dataset is provided for demonstration and deployment purposes.
