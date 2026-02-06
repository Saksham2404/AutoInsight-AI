# ==========================================================
# TRAIN MODEL SCRIPT (FAST + PRODUCTION READY)
# Used Car Price Prediction Project
# ==========================================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ==========================================================
# CONFIG
# ==========================================================

DATA_PATH = "used_cars_clean.csv"

# Use sample for faster training
SAMPLE_SIZE = 40000


# ==========================================================
# HELPER
# ==========================================================

def categorize_price(price):
    if price <= 10000:
        return "budget"
    elif price <= 30000:
        return "midrange"
    else:
        return "premium"


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

df["price_category"] = df["price"].apply(categorize_price)

# drop high-cardinality column
if "model" in df.columns:
    df = df.drop(columns=["model"])


# ==========================================================
# SAMPLE DATA (IMPORTANT FOR SPEED)
# ==========================================================

if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)

print("Training on rows:", len(df))


# ==========================================================
# ================= REGRESSION MODEL =======================
# ==========================================================

print("Training regression pipeline...")

X_reg = df.drop(columns=["price", "price_category"])
y_reg = df["price"]

cat_cols = X_reg.select_dtypes(include=["object"]).columns
num_cols = X_reg.select_dtypes(exclude=["object"]).columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

reg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=50,      # reduced for speed
        max_depth=12,
        random_state=42,
        n_jobs=2
    ))
])

reg_pipeline.fit(X_reg, y_reg)

pickle.dump(reg_pipeline, open("price_pipeline.pkl", "wb"))
print("✅ Saved price_pipeline.pkl")


# ==========================================================
# ================= CLASSIFICATION MODEL ===================
# ==========================================================

print("Training classification pipeline...")

X_clf = df.drop(columns=["price", "price_category"])
y_clf = df["price_category"]

clf_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        random_state=42,
        n_jobs=2
    ))
])

clf_pipeline.fit(X_clf, y_clf)

pickle.dump(clf_pipeline, open("category_pipeline.pkl", "wb"))
print("✅ Saved category_pipeline.pkl")


print("\n✅ Training complete. Models ready for Streamlit.")
