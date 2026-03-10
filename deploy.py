from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model_weights.pkl")
scaler = joblib.load("scaler_weights.pkl")

class HouseData(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: float
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float
    month: float
    year: float

@app.get("/")
def home():
    return {"message": "House Price Prediction API Running"}

@app.post("/predict")
def predict(data: HouseData):

    input_data = np.array([[
        data.bedrooms,
        data.bathrooms,
        data.sqft_living,
        data.sqft_lot,
        data.floors,
        data.waterfront,
        data.view,
        data.condition,
        data.grade,
        data.sqft_above,
        data.sqft_basement,
        data.yr_built,
        data.yr_renovated,
        data.zipcode,
        data.lat,
        data.long,
        data.sqft_living15,
        data.sqft_lot15,
        data.month,
        data.year
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    return {"predicted_price": float(prediction[0])}