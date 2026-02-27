from fastapi import FastAPI
from app.schema import CustomerData, ChurnInput
from app.prediction import PredictionService

app = FastAPI(title="Telcom churn Production API")

prediction_service = PredictionService()

@app.get("/")
def home():
    return {"message": "Telcom Churn production API running"}


@app.post("/predict")
def predict(data : ChurnInput):
    result = prediction_service.predict(data.dict())

    return result