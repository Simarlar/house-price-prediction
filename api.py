from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load model + preprocessor
model = joblib.load("artifacts/model.pkl")
preprocessor = joblib.load("artifacts/preprocessor.pkl")


# Input schema
class HouseData(BaseModel):
    Square_Footage: int
    Num_Bedrooms: int
    Num_Bathrooms: int
    Year_Built: int
    Lot_Size: float
    Garage_Size: int
    Neighborhood_Quality: int


@app.get("/")
def home():
    return {"message": "House Price Prediction API Running"}

@app.post("/predict")
def predict(data: HouseData):

    input_dict = {
        "Square_Footage": data.Square_Footage,
        "Num_Bedrooms": data.Num_Bedrooms,
        "Num_Bathrooms": data.Num_Bathrooms,
        "Year_Built": data.Year_Built,
        "Lot_Size": data.Lot_Size,
        "Garage_Size": data.Garage_Size,
        "Neighborhood_Quality": data.Neighborhood_Quality
    }

    input_df = pd.DataFrame([input_dict])

    # ✅ FEATURE ENGINEERING (ADD THIS)
    input_df["House_Age"] = 2026 - input_df["Year_Built"]
    input_df["SqFt_Per_Bedroom"] = input_df["Square_Footage"] / (input_df["Num_Bedrooms"] + 1)
    input_df["House_Density"] = input_df["Square_Footage"] / (input_df["Lot_Size"] + 1)
    input_df["Luxury_Score"] = input_df["Square_Footage"] * input_df["Neighborhood_Quality"]

    # Transform
    transformed = preprocessor.transform(input_df)

    prediction = model.predict(transformed)[0]

    return {
        "prediction": float(prediction)
    }