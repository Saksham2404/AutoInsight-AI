from unittest import result
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from .database import engine, SessionLocal
from .models import Base, Prediction
from sqlalchemy.orm import Session
from api.schemas import PredictionInput, PredictionOutput
from api.services.predictor import predict_price

app = FastAPI()
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (ok for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# price_pipeline = joblib.load(
#     os.path.join(BASE_DIR, "model", "price_pipeline.pkl")
# )

# category_pipeline = joblib.load(
#     os.path.join(BASE_DIR, "model", "category_pipeline.pkl")
# )



@app.get("/")
def home():
    return {"message": "AutoInsight AI API running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):

    result = predict_price(data)

    price = result["predicted_price"]
    category = result["category"]



    # ---- SAVE TO DATABASE ----
    db = SessionLocal()

    new_prediction = Prediction(
        manufacturer=data["manufacturer"],
        year=data["year"],
        odometer=data["odometer"],
        predicted_price=float(price),
        category=str(category)
    )

    db.add(new_prediction)
    db.commit()
    db.close()

    return {
        "predicted_price": float(price),
        "category": str(category)
    }
@app.get("/history")
def get_history():

    db = SessionLocal()

    predictions = db.query(Prediction).order_by(
        Prediction.id.desc()
    ).limit(20).all()

    db.close()

    result = []

    for p in predictions:
        result.append({
            "manufacturer": p.manufacturer,
            "year": p.year,
            "odometer": p.odometer,
            "predicted_price": p.predicted_price,
            "category": p.category
        })

    return result
