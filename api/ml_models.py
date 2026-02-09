# api/ml_models.py

import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

price_pipeline = joblib.load(
    os.path.join(BASE_DIR, "model", "price_pipeline.pkl")
)

category_pipeline = joblib.load(
    os.path.join(BASE_DIR, "model", "category_pipeline.pkl")
)
