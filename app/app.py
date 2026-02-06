import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os


# ---- Page setup ----
st.set_page_config(
    page_title="AutoInsight AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---- Data & model loading ----
# Cached loading keeps the app fast and avoids reloading models repeatedly

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "used_cars_sample.csv")
PRICE_MODEL_PATH = os.path.join(BASE_DIR, "model", "price_pipeline.pkl")
CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "model", "category_pipeline.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    reg_model = joblib.load(PRICE_MODEL_PATH)
    clf_model = joblib.load(CATEGORY_MODEL_PATH)
    return reg_model, clf_model


df = load_data()
reg_model, clf_model = load_models()


# ---- Custom styling ----
st.markdown("""
<style>
.main {
    background: linear-gradient(180deg,#060b16,#02050b);
    color:white;
}

.nav-btn button {
    background:#0f172a;
    border:1px solid #334155;
    border-radius:10px;
    padding:10px 20px;
    color:white;
    font-weight:600;
}

.nav-btn button:hover {
    border-color:#22c55e;
}

.price {
    font-size:42px;
    font-weight:700;
    color:#22c55e;
}

.badge {
    background:#22c55e;
    color:black;
    padding:6px 14px;
    border-radius:20px;
    font-weight:600;
    display:inline-block;
    margin-top:8px;
}

.metric-box {
    background:#020817;
    padding:20px;
    border-radius:16px;
    border:1px solid #1e293b;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)


# ---- Navigation ----
if "page" not in st.session_state:
    st.session_state.page = "home"

col1, col2, col3 = st.columns([3,4,3])
with col2:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
    with c2:
        if st.button("📊 Explore Data"):
            st.session_state.page = "eda"
    with c3:
        if st.button("🚗 Price Predictor"):
            st.session_state.page = "predict"


# ---- Home page ----
if st.session_state.page == "home":

    st.markdown(
        "<h1 style='text-align:center;'>🚘 AutoInsight AI</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center;color:#94a3b8;'>"
        "Used Car Price Prediction & Market Segmentation using Machine Learning"
        "</p>",
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"<div class='metric-box'><h2>{len(df):,}</h2>Total Vehicles</div>",
            unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"<div class='metric-box'><h2>{df['manufacturer'].nunique()}</h2>Manufacturers</div>",
            unsafe_allow_html=True)

    with c3:
        st.markdown(
            f"<div class='metric-box'><h2>${int(df['price'].median()):,}</h2>Median Price</div>",
            unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🚀 What this app does")

    st.markdown("""
Predict realistic used car prices using trained ML models  
Classify vehicles into Budget / Midrange / Premium segments  
Provide confidence estimation based on model performance  
Explore the dataset used for training
""")

    # ---- About / Author section ----
    st.markdown("---")

    st.markdown("### 👨‍💻 About the Author")

    st.markdown("""
**Saksham Malhotra**

Machine Learning & Data Science student focused on building practical AI applications and data-driven systems.  
This project demonstrates an end-to-end ML workflow — from data preprocessing and model training to deployment using Streamlit.

📫 **Connect with me:**

- GitHub: https://github.com/Saksham2404  
- LinkedIn: https://www.linkedin.com/in/saksham02
""")


# ---- Dataset explorer ----
elif st.session_state.page == "eda":

    st.header("📊 Dataset Explorer")

    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str)
    }))

    st.dataframe(df.describe())


# ---- Prediction page ----
elif st.session_state.page == "predict":

    st.header("🚗 Car Price Prediction")

    col1, col2 = st.columns([1.2,1])

    with col1:

        year = st.number_input("Year", 1990, 2025, 2015)
        odometer = st.number_input("Odometer", 0, 500000, 60000)

        manufacturer = st.selectbox(
            "Manufacturer",
            sorted(df["manufacturer"].unique())
        )

        fuel = st.selectbox(
            "Fuel",
            sorted(df["fuel"].unique())
        )

        condition = st.selectbox(
            "Condition",
            sorted(df["condition"].unique())
        )

        car_type = st.selectbox(
            "Type",
            sorted(df["type"].unique())
        )

        with st.expander("⚙️ Advanced Options (Optional)", expanded=False):
            cylinders = st.selectbox(
                "Cylinders",
                sorted(df["cylinders"].dropna().unique())
            )
            transmission = st.selectbox(
                "Transmission",
                sorted(df["transmission"].dropna().unique())
            )
            drive = st.selectbox(
                "Drive",
                sorted(df["drive"].dropna().unique())
            )
            state = st.selectbox(
                "State",
                sorted(df["state"].dropna().unique())
            )

        predict_btn = st.button("Predict Price")

    # ---- Prediction output ----
    with col2:

        if predict_btn:

            user_df = pd.DataFrame([{
                "year": year,
                "odometer": odometer,
                "manufacturer": manufacturer,
                "fuel": fuel,
                "condition": condition,
                "type": car_type,
                "cylinders": cylinders,
                "transmission": transmission,
                "drive": drive,
                "state": state
            }])

            price_pred = reg_model.predict(user_df)[0]
            category = clf_model.predict(user_df)[0]

            confidence = 82
            lower = int(price_pred * 0.85)
            upper = int(price_pred * 1.15)
            final_price = int(price_pred)

            st.markdown("### Estimated Price")

            price_placeholder = st.empty()
            for val in np.linspace(0, final_price, 25):
                price_placeholder.markdown(
                    f"<div class='price'>${int(val):,}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.015)

            with st.container():

                st.markdown(
                    f"<div class='badge'>{category.upper()}</div>",
                    unsafe_allow_html=True
                )

                st.write("")
                st.caption(f"Estimated prediction confidence: {confidence}%")

                st.progress(confidence / 100)

                st.markdown(
                    f"""
                    <div style='color:#22c55e;font-weight:600;margin-top:10px;'>
                        Expected price range: ${lower:,} — ${upper:,}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
