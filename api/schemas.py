from pydantic import BaseModel


# ---- INPUT SCHEMA ----
class PredictionInput(BaseModel):
    year: int
    odometer: int
    manufacturer: str
    fuel: str
    condition: str
    type: str
    cylinders: str
    transmission: str
    drive: str
    state: str


# ---- OUTPUT SCHEMA ----
class PredictionOutput(BaseModel):
    predicted_price: float
    category: str
