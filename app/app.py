import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os


# ---- Page configuration ----
st.set_page_config(
    page_title="AutoInsight AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---- Data & model loading ----
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


# ---- Styling ----
st.markdown("""
<style>
.main {
    background: linear-gradient(180deg,#060b16,#02050b);
    color:white;
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

.footer {
    margin-top:60px;
    padding-top:20px;
    border-top:1px solid #1e293b;
    color:#94a3b8;
    font-size:14px;
    display:flex;
    justify-content:space-between;
}
</style>
""", unsafe_allow_html=True)


# ---- Navigation ----
if "page" not in st.session_state:
    st.session_state.page = "home"

col1, col2, col3 = st.columns([3, 4, 3])
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


# ---- HOME PAGE ----
if st.session_state.page == "home":

    st.markdown("<h1 style='text-align:center;'>🚘 AutoInsight AI</h1>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center;color:#94a3b8;'>Machine Learning powered vehicle valuation and market intelligence platform</p>",
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<div class='metric-box'><h2>{len(df):,}</h2>Total Vehicles</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='metric-box'><h2>{df['manufacturer'].nunique()}</h2>Manufacturers</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='metric-box'><h2>${int(df['price'].median()):,}</h2>Median Price</div>", unsafe_allow_html=True)

    st.markdown("### 🚀 What this app does")

    st.markdown("""
Predict realistic used car prices using trained ML models  
Classify vehicles into Budget / Midrange / Premium segments  
Explore the dataset used for training
""")


# ---- EDA PAGE ----
elif st.session_state.page == "eda":

    st.header("📊 Dataset Explorer")

    st.dataframe(df.head())
    st.write("Shape:", df.shape)

    st.dataframe(pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str)
    }))

    st.dataframe(df.describe())


# ---- PREDICTION PAGE ----
elif st.session_state.page == "predict":

    st.header("🚗 Car Price Prediction")

    col1, col2 = st.columns([1.2, 1])

    with col1:

        year = st.number_input("Year", 1990, 2025, 2015)
        odometer = st.number_input("Odometer", 0, 500000, 60000)

        manufacturer = st.selectbox("Manufacturer", sorted(df["manufacturer"].unique()))
        fuel = st.selectbox("Fuel", sorted(df["fuel"].unique()))
        condition = st.selectbox("Condition", sorted(df["condition"].unique()))
        car_type = st.selectbox("Type", sorted(df["type"].unique()))

        with st.expander("⚙️ Advanced Options (Optional)", expanded=False):
            cylinders = st.selectbox("Cylinders", sorted(df["cylinders"].dropna().unique()))
            transmission = st.selectbox("Transmission", sorted(df["transmission"].dropna().unique()))
            drive = st.selectbox("Drive", sorted(df["drive"].dropna().unique()))
            state = st.selectbox("State", sorted(df["state"].dropna().unique()))

        predict_btn = st.button("Predict Price")

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

            final_price = int(price_pred)
            lower = int(price_pred * 0.85)
            upper = int(price_pred * 1.15)

            st.markdown("### Estimated Price")

            price_placeholder = st.empty()
            for val in np.linspace(0, final_price, 25):
                price_placeholder.markdown(
                    f"<div class='price'>${int(val):,}</div>",
                    unsafe_allow_html=True
                )
                time.sleep(0.015)

            st.markdown(f"<div class='badge'>{category.upper()}</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div style='color:#22c55e;font-weight:600;margin-top:10px;'>Expected price range: ${lower:,} — ${upper:,}</div>",
                unsafe_allow_html=True
            )

            # ---- Market Comparison (Dynamic) ----
            st.markdown("### 📊 Market Comparison")

            similar_cars = df[
                (df["manufacturer"] == manufacturer) &
                (df["type"] == car_type) &
                (df["fuel"] == fuel)
            ]

            if len(similar_cars) > 10:
                avg_price = int(similar_cars["price"].mean())
                median_price = int(similar_cars["price"].median())

                st.write(f"Average price of similar cars: ${avg_price:,}")
                st.write(f"Median market price: ${median_price:,}")

                if final_price > avg_price:
                    st.write("This prediction is above average market pricing.")
                else:
                    st.write("This prediction is within typical market pricing.")
            else:
                st.info("Not enough similar cars available for comparison.")

            # ---- Market Position ----
            st.markdown("### 💡 Market Position")

            if final_price > df["price"].quantile(0.75):
                st.success("This vehicle falls in the higher price range of the market.")
            elif final_price < df["price"].quantile(0.25):
                st.warning("This vehicle falls in the lower price range of the market.")
            else:
                st.info("This vehicle is priced within the typical market range.")


# ---- Footer ----
st.markdown("""
<div class="footer">
<div>Built by Saksham Malhotra</div>
<div>
<a href="https://www.linkedin.com/in/saksham02" target="_blank">LinkedIn</a> |
<a href="https://github.com/Saksham2404" target="_blank">GitHub</a>
</div>
<div>© 2026 AutoInsight AI</div>
</div>
""", unsafe_allow_html=True)
