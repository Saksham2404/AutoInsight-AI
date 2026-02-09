import pandas as pd
from ..ml_models import price_pipeline, category_pipeline


def predict_price(data: dict):
    df = pd.DataFrame([data])

    price = price_pipeline.predict(df)[0]
    category = category_pipeline.predict(df)[0]

    return {
        "predicted_price": float(price),
        "category": str(category)
    }
